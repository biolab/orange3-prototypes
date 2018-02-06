import sys
from contextlib import contextmanager
from typing import Tuple, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype

from scipy.stats import (
    hypergeom, chisquare, ttest_1samp, fligner,
    mannwhitneyu, gumbel_l, gumbel_r,
)

from joblib import Parallel, delayed


@contextmanager
def patch(obj, attr, value):
    old = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, old)


PVALUE_LABEL = 'p-value'
CORRECTED_LABEL = 'Corrected p-value (Šidák)'
COLUMN_RENAMES = {
    'pval': PVALUE_LABEL,
    'sum': 'count | class',
}


def _check_Xy(X: pd.DataFrame,
              y: pd.Series, *,
              norm_y=False) -> Tuple[pd.Series, pd.Series]:
    if np.ndim(X) == 1:
        X = pd.Series(X).to_frame()
    elif np.ndim(X) == 2:
        X = pd.DataFrame(X)

    assert X.ndim == 2
    assert np.ndim(y) == 1
    assert len(X) == len(y)

    valid = ~X.isnull().any(1).values
    X = pd.Series(list(zip(*X.values[valid].T)),
                  name=tuple(X.columns)).astype('category')
    y = pd.Series(y).reset_index(drop=True)[valid]

    if is_object_dtype(y):
        y = pd.Categorical(y)

    if norm_y:
        assert is_numeric_dtype(y)
        y = (y - y.mean()) / y.std()

    return X, y


# This is class so that it's picklable for joblib
class _PermTestChi2:
    def __init__(self, values):
        self.f_exp = values.value_counts()

    def __call__(self, values):
        f_obs = pd.Series(values).value_counts().reindex(self.f_exp.index).fillna(0)
        return chisquare(f_obs.values, self.f_exp.values)[0]


def correction_dunn_sidak(pvalues):
    return 1 - (1 - pvalues)**len(pvalues)


def _groupby_agg(x, y, agg, min_count=5, func: callable = lambda x: None):
    assert x.ndim == 1
    agg = list(agg) if isinstance(agg, Sequence) and not isinstance(agg, str) else [agg]
    df = y.groupby(x.values).agg(['count'] + agg)  # type: pd.DataFrame
    func(df)

    # Make p-values two-tailed by reversing the high-end
    pv = df['pval']
    df['pval'] = pv = pv.where(pv < .5, 1 - pv)
    assert (pv.fillna(0) <= .5).all()

    df[CORRECTED_LABEL] = correction_dunn_sidak(pv)
    df.rename(columns=COLUMN_RENAMES, inplace=True)

    df = df[df['count'] >= min_count]
    df.dropna(inplace=True)
    df.index.name = x.name
    return df


def perm_test(X, y, *, statistic='mean', n_iter=300, n_jobs=1,
              min_count=5, exact_sample_size=False, verbose=False,
              callback=None):
    x, y = _check_Xy(X, y, norm_y=statistic != 'chi2')
    min_count = max(min_count, 5)

    STATISTICS = {
        'mean': pd.Series.mean,
        'median': pd.Series.median,
        'var': pd.Series.var,
        'min': pd.Series.min,
        'max': pd.Series.max,
        'chi2': _PermTestChi2(y),
    }
    assert statistic in STATISTICS or callable(statistic)
    statistic_func = STATISTICS.get(statistic) or statistic

    # Compute the p-value by integrating the discrete tail directly
    def pv_cdf(score, scores):
        return (np.sum(scores <= score) + 1) / (scores.size + 1)

    cache = {}
    parallel = Parallel(n_jobs=n_jobs, backend='threading')

    if callback:
        assert callable(callback)
        _n = 0

        def callback(callback=callback, _n_groups = x.nunique()):
            nonlocal _n
            _n += 1
            callback(_n, _n_groups)

    def pval(group):
        nonlocal y, pv_cdf, statistic_func, exact_sample_size, cache, parallel, callback
        n = len(group)

        if callback:
            callback()

        if n < min_count:
            return np.nan

        # Round n to order of magnitude for more cache hits
        if not exact_sample_size:
            n = round(n, -int(np.log10(n)))

        # Clip n to y size to avoid 'larger sample than population'
        # error in pd.Series.sample()
        n = min(n, len(y))

        sample_distance = statistic_func(group)

        distances = cache.get(n)
        print('' if distances is None else n, sep='', end='.')
        if distances is None:
            # Stage 1: Compute only 100 iterations
            N1_ITER = 50
            distances, n_iter_remaining = [], n_iter
            if n_iter >= 2*N1_ITER:
                n_iter_remaining -= N1_ITER
                distances = np.array(
                    parallel(delayed(statistic_func)(y.sample(n))
                             for _ in range(N1_ITER)))

                # If p-value of first stage iterations is not significant,
                # do not continue
                pv = pv_cdf(sample_distance, distances)
                if pv > .2 or pv < .8:
                    return pv

            # Stage 2: Compute the rest n_iter iterations
            cache[n] = distances = np.r_[
                distances,
                parallel(delayed(statistic_func)(y.sample(n))
                         for _ in range(n_iter_remaining))
            ]

        return pv_cdf(sample_distance, distances)

    with patch(sys, 'stdout', sys.stdout if verbose else None):
        res = _groupby_agg(x, y, pval, min_count=min_count)
        print()
        return res


def chi2_test(X, y, *, ddof=0, min_count=5):
    x, y = _check_Xy(X, y)
    f_exp = y.value_counts()
    min_count = max(min_count, 5)

    def pval(grp, _f_exp=f_exp, _ddof=ddof):
        if grp.size < min_count:
            return np.nan
        f_obs = grp.value_counts().reindex(_f_exp.index)
        f_obs.fillna(0, inplace=True)
        if not (f_obs >= 5).all():
            return np.nan
        return chisquare(f_obs.values, _f_exp.values, ddof=_ddof)[1]

    if not (f_exp >= 5).all():
        def pval(_):
            return np.nan

    return _groupby_agg(x, y, pval, min_count=min_count)


def hyper_test(X, y, *, min_count=5):
    x, y = _check_Xy(X, y)
    assert y.dtype == bool
    min_count = max(min_count, 5)

    # N, n, K, k as in https://en.wikipedia.org/wiki/Hypergeometric_distribution#Definition
    N, K = y.size, y.sum()

    def pval(grp):
        if grp.size < min_count:
            return np.nan
        return hypergeom(M=N, n=K, N=len(grp), loc=1).sf(x=grp.sum())

    def enrichment(grp):
        return (grp.sum() / grp.size) / (K / N)

    return _groupby_agg(x, y, ['sum', enrichment, pval], min_count=min_count)


def t_test(X, y, min_count=5):
    x, y = _check_Xy(X, y, norm_y=True)
    min_count = max(min_count, 5)

    def pval(grp, _popmean=y.mean()):
        if grp.size < min_count:
            return np.nan
        return ttest_1samp(grp, popmean=_popmean, nan_policy='omit')[1]

    return _groupby_agg(x, y, pval, min_count=min_count)


def fligner_killeen_test(X, y, min_count=5):
    x, y = _check_Xy(X, y, norm_y=True)
    min_count = max(min_count, 5)

    def pval(grp):
        if grp.size < min_count:
            return np.nan
        return fligner(grp.values, y.values)[1]

    return _groupby_agg(x, y, pval, min_count=min_count)


def mannwhitneyu_test(X, y, min_count=20):
    x, y = _check_Xy(X, y, norm_y=True)
    min_count = max(min_count, 20)

    def pval(grp, _y_sorted=y.sort_values().values):
        if grp.size < min_count:
            return np.nan
        return mannwhitneyu(grp.values, _y_sorted)[1]

    return _groupby_agg(x, y, pval, min_count=min_count)


def gumbel_min_test(X, y, min_count=5):
    x, y = _check_Xy(X, y, norm_y=True)
    min_count = max(min_count, 5)
    return _groupby_agg(x, y, 'min', min_count=min_count,
                        func=lambda df: df.__setitem__('pval', gumbel_l.cdf(df.pop('min').values)))


def gumbel_max_test(X, y, min_count=5):
    x, y = _check_Xy(X, y, norm_y=True)
    min_count = max(min_count, 5)
    return _groupby_agg(x, y, 'max', min_count=min_count,
                        func=lambda df: df.__setitem__('pval', gumbel_r.cdf(df.pop('max').values)))


if __name__ == '__main__':
    N = 50
    y = np.r_[np.random.random(N), np.random.random(N) * 10]
    X = np.c_[list('A' * N) + list('B' * N)]
    # print(X)
    # print(y)
    print(t_test(X, y))
    print(fligner_killeen_test(X, y))
    print(hyper_test(X, y > 5))
    print(chi2_test(X, y > 5))
    print(y > 5)
    print(perm_test(X, y, statistic='var', min_count=1, n_jobs=-1))
    print(perm_test(X, y > 7, statistic='chi2', min_count=1, n_jobs=-1))
