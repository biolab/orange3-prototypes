import contextlib
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from Orange.widgets.utils.colorpalettes import LimitedDiscretePalette
from scipy import sparse
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

from Orange.base import Model
from Orange.data import Table, Domain
from Orange.util import dummy_callback, wrap_callback
from shap import KernelExplainer, TreeExplainer
from shap.common import DenseData, SHAPError, sample

RGB_LOW = [0, 137, 229]
RGB_HIGH = [255, 0, 66]


@contextlib.contextmanager
def temp_seed(seed):
    """
    This function provides an environment with a custom random seed. It reset
    it back to normal when exiting the environment.
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def kmeans(X, k, round_values=True):
    """
    This function should be imported from shap.kmeans. Remove it when they
    merge and release the following changes:
    https://github.com/slundberg/shap/pull/1135
    """
    group_names = [str(i) for i in range(X.shape[1])]
    if str(type(X)).endswith("'pandas.core.frame.DataFrame'>"):
        group_names = X.columns
        X = X.values

    # in case there are any missing values in data impute them
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    X = imp.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

    if round_values:
        for i in range(k):
            for j in range(X.shape[1]):
                xj = X[:, j].toarray().flatten() if issparse(X) else X[:, j]
                ind = np.argmin(np.abs(xj - kmeans.cluster_centers_[i, j]))
                kmeans.cluster_centers_[i, j] = X[ind, j]
    return DenseData(
        kmeans.cluster_centers_,
        group_names,
        None,
        1.0 * np.bincount(kmeans.labels_),
    )


def _subsample_data(data: Table, n_samples: int) -> Tuple[Table, np.array]:
    """
    Randomly subsample rows in data to n_samples
    """
    if len(data) > n_samples:
        idx = np.random.choice(len(data), n_samples, replace=False)
        # first make mask since idx not sorted - sampling with idx mix data
        mask_array = np.zeros(len(data), dtype=bool)
        mask_array[idx] = True
        data_sample = data[mask_array]
    else:
        data_sample = data
        mask_array = np.ones(len(data), dtype=bool)
    return data_sample, mask_array


def _join_shap_values(
    shap_values: List[Union[List[np.ndarray], np.ndarray]]
) -> List[np.ndarray]:
    """
    Since the explanation algorithm is called multiple times we need to join
    results back together. When explaining classification, the result is a list
    of lists with the np.ndarray for each class, when explaining regression,
    the result is the list of one np.ndarrays.
    """
    if isinstance(shap_values[0], np.ndarray):
        # regression
        return [np.vstack(shap_values)]
    else:
        # classification
        return [np.vstack(s) for s in zip(*shap_values)]


def _explain_trees(
    model: Model,
    transformed_data: Table,
    transformed_reference_data: Table,
    progress_callback: Callable,
) -> Tuple[
    Optional[List[np.ndarray]], Optional[np.ndarray], Optional[np.ndarray]
]:
    """
    Computes and returns SHAP values for learners that are explained by
    TreeExplainer: all sci-kit models based on trees. In case that explanation
    with TreeExplainer is not possible it returns None
    """
    if sparse.issparse(transformed_data.X):
        # sparse not supported by TreeExplainer, KernelExplainer can handle it
        return None, None, None
    try:
        explainer = TreeExplainer(
            model.skl_model, data=sample(transformed_reference_data.X, 100),
        )
    except (SHAPError, AttributeError):
        return None, None, None

    # TreeExplaner cannot explain in normal time more cases than 1000
    data_sample, sample_mask = _subsample_data(transformed_data, 1000)
    num_classes = (
        len(model.domain.class_var.values)
        if model.domain.class_var.is_discrete
        else None
    )

    # this method will work in batches since explaining only one attribute
    # at time the processing timed doubles comparing to batch size 10
    shap_values = []
    batch_size = 1  # currently set to 1 to minimize widget blocking
    for i in range(0, len(data_sample), batch_size):
        progress_callback(i / len(data_sample))
        batch = data_sample.X[i : i + batch_size]
        shap_values.append(
            explainer.shap_values(batch, check_additivity=False)
        )

    shap_values = _join_shap_values(shap_values)
    base_value = explainer.expected_value
    # when in training phase one class value was missing skl_model do not
    # output probability for it. For other models it is handled by Orange
    if num_classes is not None:
        missing_d = num_classes - len(shap_values)
        shap_values += [
            np.zeros(shap_values[0].shape) for _ in range(missing_d)
        ]
        base_value = np.hstack((base_value, np.zeros(missing_d)))

    return shap_values, sample_mask, base_value


def _explain_other_models(
    model: Model,
    transformed_data: Table,
    transformed_reference_data: Table,
    progress_callback: Callable,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Computes SHAP values for any learner with KernelExplainer.
    """
    # 1000 is a number that for normal data and model do not take so long
    data_sample, sample_mask = _subsample_data(transformed_data, 1000)

    try:
        ref = kmeans(transformed_reference_data.X, k=10)
    except ValueError:
        # k-means fails with value error when it cannot produce enough clusters
        # in this case we will use sample instead of clusters
        ref = sample(transformed_reference_data.X, nsamples=100)

    explainer = KernelExplainer(
        lambda x: (
            model(x)
            if model.domain.class_var.is_continuous
            else model(x, model.Probs)
        ),
        ref,
    )

    shap_values = []
    for i, row in enumerate(data_sample.X):
        progress_callback(i / len(data_sample))
        shap_values.append(
            explainer.shap_values(row, nsamples=100, silent=True, l1_reg=False)
        )
    return (
        _join_shap_values(shap_values),
        sample_mask,
        explainer.expected_value,
    )


def compute_shap_values(
    model: Model,
    data: Table,
    reference_data: Table,
    progress_callback: Callable = None,
) -> Tuple[List[np.ndarray], Table, np.ndarray, np.ndarray]:
    """
    Compute SHAP values - explanation for a model. And also give a transformed
    data table.

    Parameters
    ----------
    model
        Model which is explained.
    data
        Data to be explained
    reference_data
        Background data for perturbation purposes
    progress_callback
        The callback for reporting the progress.

    Returns
    -------
    shap_values
        Shapely values for each data item computed by the SHAP library. The
        result is a list of SHAP values for each class - the class order is
        taken from values in the class_var. Each array in the list has shape
        (num cases x num attributes) - explanation for the contribution of each
         attribute to the final prediction.
    data_transformed
        The table on which explanation was made: table preprocessed by models
        preprocessors
    sample_mask
        SHAP values are computed just for a data sample. It is a boolean mask
        that tells which rows in data_transformed are explained.
    base_value
        The base value (average prediction on dataset) for each class.
    """
    # ensure that sampling and SHAP value calculation is same for same data
    with temp_seed(0):
        if progress_callback is None:
            progress_callback = dummy_callback
        progress_callback(0, "Computing explanation ...")

        data_transformed = model.data_to_model_domain(data)
        reference_data_transformed = model.data_to_model_domain(reference_data)

        shap_values, sample_mask, base_value = _explain_trees(
            model,
            data_transformed,
            reference_data_transformed,
            progress_callback,
        )
        if shap_values is None:
            shap_values, sample_mask, base_value = _explain_other_models(
                model,
                data_transformed,
                reference_data_transformed,
                progress_callback,
            )

        # for regression return array with one value
        if not isinstance(base_value, np.ndarray):
            base_value = np.array([base_value])

        progress_callback(1)
    return shap_values, data_transformed, sample_mask, base_value


def _get_min_max(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute min and max bases for each column in the data. This are not real
    min or max values but values at 5/95 percentile.
    """
    vmin = np.nanpercentile(values, 5, axis=0)
    vmax = np.nanpercentile(values, 95, axis=0)

    # fix where equal
    equals = vmin == vmax
    vmin[equals] = np.nanpercentile(values[:, equals], 1, axis=0)
    vmax[equals] = np.nanpercentile(values[:, equals], 1, axis=0)

    # fix where still equal
    equals = vmin == vmax
    vmin[equals] = np.min(values[:, equals], axis=0)
    vmax[equals] = np.max(values[:, equals], axis=0)

    # fix where vmin higher than vmax - rare numerical precision issues
    greater = vmin > vmax
    vmin[greater] = vmax[greater]

    assert vmin.shape == (values.shape[1],)
    assert vmax.shape == (values.shape[1],)
    return vmin, vmax


def compute_colors(data: Table) -> np.ndarray:
    """
    Compute colors which represent how high is a value comparing to other
    values for each feature. For discrete features return a vale's color.

    Parameters
    ----------
    data
        Data for which function computes color values

    Returns
    -------
   Colors for each data instance and each feature. The shape of the matrix is
   M x N x C, where M is a number of instances, N is a number of features, and
   C is 3 (one value for each RGB channel).
    """

    def continuous_colors(x):
        min_, max_ = _get_min_max(x)
        x = x.copy()
        for i in range(len(max_)):
            x[x[:, i] > max_[i], i] = max_[i]
            x[x[:, i] < min_[i], i] = min_[i]
        normalized_x = (x - min_) / (max_ - min_)
        # when max_ and min_ completely same use average color
        normalized_x[:, max_ == min_] = 0.5

        v = [normalized_x * (b - a) + a for (a, b) in zip(RGB_LOW, RGB_HIGH)]
        cont_colors = np.dstack(v)

        # missing values are imputed as gray
        return cont_colors

    def discrete_colors(x, attributes):
        disc_colors = np.zeros(x.shape + (3,))
        for i, a in enumerate(attributes):
            nonnan = ~np.isnan(x[:, i])
            colors = (
                a.colors
                if hasattr(a, "colors")
                else LimitedDiscretePalette(len(a.values)).palette
            )
            disc_colors[nonnan, i] = colors[x[nonnan, i].astype(int)]
        return disc_colors

    # the final array is dense and we do not expect huge matrices here
    values = data.X.toarray() if sparse.issparse(data.X) else data.X
    colors = np.zeros(values.shape + (3,))
    is_discrete = np.array(
        [a.is_discrete for a in data.domain.attributes], dtype=bool
    )

    colors[:, ~is_discrete] = continuous_colors(values[:, ~is_discrete])
    colors[:, is_discrete] = discrete_colors(
        values[:, is_discrete],
        [a for a in data.domain.attributes if a.is_discrete],
    )

    colors[np.isnan(colors)] = 100
    return colors


def get_shap_values_and_colors(
    model: Model, data: Table, progress_callback: Callable = None
) -> Tuple[List[np.ndarray], List[str], np.ndarray, np.ndarray]:
    """
    Compute SHAP values and colors that represent how high is the feature value
    comparing to other values for the same feature. This function provides all
    required components for explain model widget.

    Parameters
    ----------
    model
        Model which predictions are explained explained.
    data
        Data, which's prediction is explained.
    progress_callback
        The callback for reporting the progress.

    Returns
    -------
    shap_values
        Shapely values for each data item computed by the SHAP library. The
        result is a list of SHAP values for each class - the class order is
        taken from values in the class_var. Each array in the list has shape
        (num cases x num attributes) - explanation for the contribution of each
         attribute to the final prediction.
    attributes
        The attributes from table on which explanation was made: table
        preprocessed by models preprocessors
    sample_mask
        SHAP values are computed just for a data sample. It is a boolean mask
        that tells which rows in data are explained.
    colors
        Colors for each data instance and each feature. The shape of the matrix
        is M x N x C, where M is a number of instances, N is a number of
        features, and C is 3 (one value for each RGB channel).
    """
    if progress_callback is None:
        progress_callback = dummy_callback
    cb = wrap_callback(progress_callback, end=0.9)

    shap_values, transformed_data, sample_mask, _ = compute_shap_values(
        model, data, data, progress_callback=cb
    )

    colors = compute_colors(transformed_data[sample_mask])
    attributes = [t.name for t in transformed_data.domain.attributes]
    progress_callback(1)

    return shap_values, attributes, sample_mask, colors


def explain_predictions(
    model: Model,
    data: Table,
    background_data: Table,
    progress_callback: Callable = None,
) -> Tuple[List[np.ndarray], np.ndarray, Table, np.ndarray]:
    """
    Compute SHAP values and predictions for each item in data.
    This function provides all required components for explaining the
    prediction widget.

    Parameters
    ----------
    model
        Model which prediction is explained
    data
        Data, which's prediction is explained.
    background_data
        Data which are used as a background data for explanation process.
        SHAP used them in the perturbation process.
    progress_callback
        Callback to report progress.

    Returns
    -------
    shap_values
        Shapely values for each data item computed by the SHAP library. The
        result is a list of SHAP values for each class - the class order is
        taken from values in the class_var. Each array in the list has shape
        (num cases x num attributes) - explanation for the contribution of each
         attribute to the final prediction.
    predictions
        num_cases X num_classes array with predictions/probabilities from the
        model for each case. In the case of classification, each column is a
        probability for each class. For regression, the number of columns is 1
        and the result is prediction itself.
    transformed_data
        Table on which explanation was made: table preprocessed by models
        preprocessors
    base_value
        The base value (average prediction on dataset) for each class.
    """
    if progress_callback is None:
        progress_callback = dummy_callback
    progress_callback(0)

    # prediction happens independent from the class -
    # same than for prediction widget
    classless_data = data.transform(
        Domain(data.domain.attributes, None, data.domain.metas)
    )
    predictions = model(
        classless_data, model.Probs if model.domain.class_var.is_discrete else model.Value
    )
    # for regression - predictions array is 1d transform it shape N x 1
    if predictions.ndim == 1:
        predictions = predictions[:, None]

    shap_values, transformed_data, _, base_value = compute_shap_values(
        model, data, background_data, progress_callback
    )
    return shap_values, predictions, transformed_data, base_value


def _compute_segments(
    shap_valus: np.ndarray, prediction: float
) -> List[Tuple[float, float]]:
    """
    Compute starting and ending point of the segment on y-asis.
    """
    curr = prediction
    segments = []
    for sh in shap_valus:
        segments.append((curr, curr - sh))
        curr -= sh
    return segments


def prepare_force_plot_data(
    shap_values: List[np.ndarray],
    transformed_data: Table,
    predictions: np.ndarray,
    target_class: int,
    top_n_features: Optional[int] = None,
) -> Tuple[List[Tuple], List[Tuple], List[Tuple], List[Tuple]]:
    """
    Prepare data for a force plot. It select top_n_features most important
    features.

    Parameters
    ----------
    shap_values
        SHAP values to be plotted
    transformed_data
        Data for which SHAP values are plotted
    predictions
        Predictions for each data item.
    target_class
        Target class for a force plot.
    top_n_features
        Number of features for which we will show importance

    Returns
    -------
    selected_shap_values
        SHAP values for the most important features. It is a list with a tuple
        for each data item. Each tuple contains two arrays First are positive
        SHAP values sorted from most to least important, second contains
        negative SHAP values sorted from most to least important.
    segments
        Value on y-axis to start/end segment. It is a list with a tuple for
        each data item. Each tuple contains two arrays First are segments for
        positive SHAP values sorted from most to least important, second
        contains segments for negative SHAP values sorted from most to least
        important. Each segment is a tuple with starting and the ending point.
    selected_labels
        Corresponding attributes and values. It is a list with a tuple for each
        data item (row in data). Each tuple contains two arrays. In first are
        attributes and values with positive SHAP values sorted from most to
        least important, second contains attributes with negative SHAP values
        sorted from most to least important.
    ranges
        Range of the graph for each data item.
    """
    shap_values = shap_values[target_class]
    if top_n_features is None:
        top_n_features = shap_values.shape[1]

    top_features_idx = np.fliplr(np.argsort(np.abs(shap_values), axis=1),)[
        :, :top_n_features
    ]

    assert top_features_idx.shape == (
        len(shap_values),
        min(shap_values.shape[1], top_n_features),
    )

    data_attributes = transformed_data.domain.attributes

    selected_shap_values = []
    selected_labels = []
    segments = []
    ranges = []
    for sv, tfi, dat, pred in zip(
        shap_values, top_features_idx, transformed_data, predictions
    ):
        positive_idx = tfi[sv[tfi] > 0]
        negative_ids = tfi[sv[tfi] < 0]
        selected_shap_values.append(
            (sv[positive_idx].tolist(), sv[negative_ids].tolist())
        )
        selected_labels.append(
            (
                [(data_attributes[i].name, dat.x[i]) for i in positive_idx],
                [(data_attributes[i].name, dat.x[i]) for i in negative_ids],
            )
        )
        pos_segments = _compute_segments(sv[positive_idx], pred[target_class])
        neg_segments = _compute_segments(sv[negative_ids], pred[target_class])
        segments.append((pos_segments, neg_segments))
        ranges.append(
            (
                pos_segments[-1][1]
                if len(pos_segments)
                else pred[target_class],
                neg_segments[-1][1]
                if len(neg_segments)
                else pred[target_class],
            )
        )
    return selected_shap_values, segments, selected_labels, ranges


if __name__ == "__main__":
    from Orange.classification import LogisticRegressionLearner

    data_ = Table.from_file("heart_disease.tab")
    learner = LogisticRegressionLearner()
    model_ = learner(data_)

    shap_val, transformed_domain, mask, colors_ = get_shap_values_and_colors(
        model_, data_
    )
