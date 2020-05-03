import unittest

import numpy as np

from Orange.classification import (
    LogisticRegressionLearner,
    RandomForestLearner,
    SGDClassificationLearner,
    SVMLearner,
    TreeLearner,
)
from Orange.data import Table, Domain
from Orange.regression import LinearRegressionLearner
from Orange.tests.test_classification import LearnerAccessibility
from Orange.tests.test_regression import TestRegression
from Orange.widgets.data import owcolor
from orangecontrib.prototypes.explanation.explainer import (
    compute_colors,
    compute_shap_values,
    explain_predictions,
    get_shap_values_and_colors,
    prepare_force_plot_data,
)


class TestExplainer(unittest.TestCase):
    def setUp(self) -> None:
        self.iris = Table.from_file("iris")
        self.housing = Table.from_file("housing")[:100, -10:]
        self.titanic = Table("titanic")

    def test_tree_explainer(self):
        learner = RandomForestLearner()
        model = learner(self.iris)

        shap_values, _, sample_mask, base_value = compute_shap_values(
            model, self.iris, self.iris
        )

        self.assertEqual(len(shap_values), 3)
        self.assertTupleEqual(shap_values[0].shape, self.iris.X.shape)
        self.assertTupleEqual(shap_values[1].shape, self.iris.X.shape)
        self.assertTupleEqual(shap_values[2].shape, self.iris.X.shape)
        self.assertIsInstance(shap_values, list)
        self.assertIsInstance(shap_values[0], np.ndarray)
        # number of cases to short to be subsampled
        self.assertEqual(len(shap_values[0]), sample_mask.sum())
        self.assertTupleEqual(
            (len(self.iris.domain.class_var.values),), base_value.shape
        )

        # test with small dataset
        shap_values, _, sample_mask, base_value = compute_shap_values(
            model, self.iris[:1], self.iris[:5]
        )
        self.assertEqual(len(shap_values), 3)
        self.assertTupleEqual(shap_values[0].shape, (1, 4))
        self.assertTupleEqual(shap_values[1].shape, (1, 4))
        self.assertTupleEqual(shap_values[2].shape, (1, 4))

    def test_kernel_explainer(self):
        learner = LogisticRegressionLearner()
        model = learner(self.iris)

        shap_values, _, sample_mask, base_value = compute_shap_values(
            model, self.iris, self.iris
        )

        self.assertEqual(len(shap_values), 3)
        self.assertTupleEqual(shap_values[0].shape, self.iris.X.shape)
        self.assertTupleEqual(shap_values[1].shape, self.iris.X.shape)
        self.assertTupleEqual(shap_values[2].shape, self.iris.X.shape)
        self.assertIsInstance(shap_values, list)
        self.assertIsInstance(shap_values[0], np.ndarray)
        # number of cases to short to be subsampled
        self.assertEqual(len(shap_values[0]), sample_mask.sum())
        self.assertTupleEqual(
            (len(self.iris.domain.class_var.values),), base_value.shape
        )

        # test with small dataset
        shap_values, _, sample_mask, base_value = compute_shap_values(
            model, self.iris[:1], self.iris[:5]
        )
        self.assertEqual(len(shap_values), 3)
        self.assertTupleEqual(shap_values[0].shape, (1, 4))
        self.assertTupleEqual(shap_values[1].shape, (1, 4))
        self.assertTupleEqual(shap_values[2].shape, (1, 4))

    def test_kernel_explainer_sgd(self):
        learner = SGDClassificationLearner()
        model = learner(self.titanic)
        np.random.shuffle(self.titanic.X)

        shap_values, _, sample_mask, _ = compute_shap_values(
            model, self.titanic[:200], self.titanic[:200]
        )

    def test_explain_regression(self):
        learner = LinearRegressionLearner()
        model = learner(self.housing)

        shap_values, _, sample_mask, base_value = compute_shap_values(
            model, self.housing, self.housing
        )

        self.assertEqual(len(shap_values), 1)
        self.assertTupleEqual(shap_values[0].shape, self.housing.X.shape)
        self.assertIsInstance(shap_values, list)
        self.assertIsInstance(shap_values[0], np.ndarray)
        # number of cases to short to be subsampled
        self.assertEqual(len(shap_values[0]), sample_mask.sum())
        self.assertTupleEqual((1,), base_value.shape)

    def test_class_not_predicted(self):
        """
        This is a case where one class is missing in the data. In this case
        skl learners output probabilities with only two classes. Orange models
        adds a zero probability for a missing class. In case where we work
        directly with skl learners - all tree-like learners it is added
        manually and tested here.
        """
        learner = RandomForestLearner()
        model = learner(self.iris[:100])

        shap_values, _, _, base_value = compute_shap_values(
            model, self.iris[:100], self.iris[:100]
        )

        self.assertEqual(len(shap_values), 3)
        self.assertTupleEqual((3,), base_value.shape)
        self.assertTrue(np.any(shap_values[0]))
        self.assertTrue(np.any(shap_values[1]))
        # missing class has all shap values 0
        self.assertFalse(np.any(shap_values[2]))

        # for one class SHAP returns only array (not list of arrays) -
        # must be handled
        learner = RandomForestLearner()
        model = learner(self.iris[:50])

        shap_values, _, _, base_value = compute_shap_values(
            model, self.iris[:100], self.iris[:100]
        )
        self.assertEqual(len(shap_values), 3)
        self.assertTupleEqual((3,), base_value.shape)

        # for Logistic regression Orange handle that - test anyway
        learner = LogisticRegressionLearner()
        model = learner(self.iris[:100])

        shap_values, _, _, base_value = compute_shap_values(
            model, self.iris[:100], self.iris[:100]
        )
        self.assertEqual(len(shap_values), 3)
        self.assertTupleEqual((3,), base_value.shape)
        self.assertNotEqual(shap_values[0].sum(), 0)
        self.assertNotEqual(shap_values[1].sum(), 0)
        # missing class has all shap values 0
        self.assertTrue(not np.any(shap_values[2].sum()))

    @unittest.skip("Enable when learners fixed")
    def test_all_classifiers(self):
        """ Test explanation for all classifiers """
        for learner in LearnerAccessibility.all_learners(None):
            with self.subTest(learner.name):
                model = learner(self.iris)
                shap_values, _, _, _ = compute_shap_values(
                    model, self.iris, self.iris
                )
                self.assertEqual(len(shap_values), 3)
                for i in range(3):
                    self.assertTupleEqual(
                        self.iris.X.shape, shap_values[i].shape
                    )

    def test_all_regressors(self):
        """ Test explanation for all regressors """
        for learner in TestRegression.all_learners(None):
            with self.subTest(learner.name):
                model = learner()(self.housing)
                shap_values, _, _, _ = compute_shap_values(
                    model, self.housing, self.housing
                )
                self.assertEqual(len(shap_values), 1)
                self.assertTupleEqual(
                    self.housing.X.shape, shap_values[0].shape
                )

    def test_sparse(self):
        sparse_iris = self.iris.to_sparse()
        learner = LogisticRegressionLearner()
        model = learner(sparse_iris)

        shap_values, _, _, _ = compute_shap_values(
            model, sparse_iris, sparse_iris
        )
        self.assertTupleEqual(shap_values[0].shape, sparse_iris.X.shape)
        self.assertTupleEqual(shap_values[1].shape, sparse_iris.X.shape)
        self.assertTupleEqual(shap_values[2].shape, sparse_iris.X.shape)

        learner = RandomForestLearner()
        model = learner(sparse_iris)

        shap_values, _, _, _ = compute_shap_values(
            model, sparse_iris, sparse_iris
        )
        self.assertTupleEqual(shap_values[0].shape, sparse_iris.X.shape)
        self.assertTupleEqual(shap_values[1].shape, sparse_iris.X.shape)
        self.assertTupleEqual(shap_values[2].shape, sparse_iris.X.shape)

    def test_missing_values(self):
        heart_disease = Table("heart_disease.tab")
        learner = TreeLearner()
        model = learner(heart_disease)
        shap_values, _, _, _ = compute_shap_values(
            model, heart_disease, heart_disease
        )
        self.assertEqual(len(shap_values), 2)
        self.assertTupleEqual(shap_values[0].shape, heart_disease.X.shape)
        self.assertTupleEqual(shap_values[1].shape, heart_disease.X.shape)

    def test_compute_colors(self):
        heart_disease = Table.from_file("heart_disease.tab")
        colors = compute_colors(heart_disease)
        self.assertTupleEqual(colors.shape, heart_disease.X.shape + (3,))

        # the way to add colors to attributes
        [owcolor.DiscAttrDesc(a) for a in heart_disease.domain.attributes]

        colors = compute_colors(heart_disease)
        self.assertTupleEqual(colors.shape, heart_disease.X.shape + (3,))

        titanic = Table("titanic")
        model = SVMLearner()(titanic)
        titanic_proc = model.data_to_model_domain(titanic)

        colors = compute_colors(titanic_proc)
        self.assertTupleEqual(colors.shape, titanic_proc.X.shape + (3,))

    def test_subsample(self):
        titanic = Table("titanic")
        learner = LogisticRegressionLearner()
        model = learner(titanic)

        shap_values, _, sample_mask, _ = compute_shap_values(
            model, titanic, titanic
        )
        self.assertTupleEqual((1000, 8), shap_values[0].shape)
        self.assertTupleEqual((2201,), sample_mask.shape)

        # sample mask should match due to same random seed
        _, _, sample_mask_new, _ = compute_shap_values(model, titanic, titanic)
        np.testing.assert_array_equal(sample_mask, sample_mask_new)

    def test_shap_random_seed(self):
        model = LogisticRegressionLearner()(self.iris)

        shap_values, _, _, _ = compute_shap_values(model, self.iris, self.iris)
        shap_values_new, _, _, _ = compute_shap_values(
            model, self.iris, self.iris
        )
        np.testing.assert_array_equal(shap_values, shap_values_new)

        model = RandomForestLearner()(self.iris)

        shap_values, _, _, _ = compute_shap_values(model, self.iris, self.iris)
        shap_values_new, _, _, _ = compute_shap_values(
            model, self.iris, self.iris
        )
        np.testing.assert_array_equal(shap_values, shap_values_new)

    def test_get_shap_values_and_colors(self):
        model = LogisticRegressionLearner()(self.iris)

        shap_values1, transformed_data, mask1, _ = compute_shap_values(
            model, self.iris, self.iris
        )
        colors1 = compute_colors(transformed_data)

        shap_values2, attributes, mask2, colors2 = get_shap_values_and_colors(
            model, self.iris
        )

        np.testing.assert_array_equal(shap_values1, shap_values2)
        np.testing.assert_array_equal(colors1, colors2)
        self.assertListEqual(
            list(map(lambda x: x.name, transformed_data.domain.attributes)),
            attributes,
        )
        np.testing.assert_array_equal(mask1, mask2)

    def test_explain_predictions(self):
        model = LogisticRegressionLearner()(self.iris)

        shap_values, predictions, _, _ = explain_predictions(
            model, self.iris[:3], self.iris
        )

        self.assertEqual(3, len(shap_values))
        self.assertTupleEqual((3, self.iris.X.shape[1]), shap_values[0].shape)
        self.assertTupleEqual((3, self.iris.X.shape[1]), shap_values[1].shape)
        self.assertTupleEqual((3, self.iris.X.shape[1]), shap_values[2].shape)

        self.assertTupleEqual(
            (3, len(self.iris.domain.class_var.values)), predictions.shape
        )

        # regression
        model = LinearRegressionLearner()(self.housing)
        shap_values, predictions, _, _ = explain_predictions(
            model, self.housing[:3], self.housing
        )

        self.assertEqual(1, len(shap_values))
        self.assertTupleEqual(
            (3, self.housing.X.shape[1]), shap_values[0].shape
        )
        self.assertTupleEqual((3, 1), predictions.shape)

    def test_prepare_force_plot_data_target_0(self):
        shap_values = [
            np.array([[1, -2, 6, 5], [-2, -3, -1, -5], [1, 2, 4, 5]]),
            np.random.random((3, 4)),
        ]
        predictions = np.array([[2, 1], [3, 1], [4, 1]])

        shaps, segments, labels, ranges = prepare_force_plot_data(
            shap_values, self.iris[:4], predictions, 0, top_n_features=3
        )
        self.assertListEqual(
            [([6, 5], [-2]), ([], [-5, -3, -2]), ([5, 4, 2], [])], shaps
        )
        self.assertListEqual(
            [
                ([(2, -4), (-4, -9)], [(2, 4)]),
                ([], [(3, 8), (8, 11), (11, 13)]),
                ([(4, -1), (-1, -5), (-5, -7)], []),
            ],
            segments,
        )
        self.assertListEqual(
            [
                (
                    [("petal length", 1.4), ("petal width", 0.2)],
                    [("sepal width", 3.5)],
                ),
                (
                    [],
                    [
                        ("petal width", 0.2),
                        ("sepal width", 3.0),
                        ("sepal length", 4.9),
                    ],
                ),
                (
                    [
                        ("petal width", 0.2),
                        ("petal length", 1.3),
                        ("sepal width", 3.2),
                    ],
                    [],
                ),
            ],
            labels,
        ),
        self.assertListEqual([(-9, 4), (3, 13), (-7, 4)], ranges)

    def test_prepare_force_plot_data_target_1(self):
        # for target class 1
        shap_values = [
            np.random.random((3, 4)),
            np.array([[1, -2, 6, 5], [-2, -3, -1, -5], [1, 2, 4, 5]]),
        ]
        predictions = np.array([[1, 2], [1, 3], [1, 4]])

        shaps, segments, labels, ranges = prepare_force_plot_data(
            shap_values, self.iris[:4], predictions, 1, top_n_features=3
        )
        self.assertListEqual(
            [([6, 5], [-2]), ([], [-5, -3, -2]), ([5, 4, 2], [])], shaps
        )
        self.assertListEqual(
            [
                ([(2, -4), (-4, -9)], [(2, 4)]),
                ([], [(3, 8), (8, 11), (11, 13)]),
                ([(4, -1), (-1, -5), (-5, -7)], []),
            ],
            segments,
        )
        self.assertListEqual(
            [
                (
                    [("petal length", 1.4), ("petal width", 0.2)],
                    [("sepal width", 3.5)],
                ),
                (
                    [],
                    [
                        ("petal width", 0.2),
                        ("sepal width", 3.0),
                        ("sepal length", 4.9),
                    ],
                ),
                (
                    [
                        ("petal width", 0.2),
                        ("petal length", 1.3),
                        ("sepal width", 3.2),
                    ],
                    [],
                ),
            ],
            labels,
        )
        self.assertListEqual([(-9, 4), (3, 13), (-7, 4)], ranges)

    def test_prepare_force_plot_less_attributes(self):
        # for target class 1
        shap_values = [
            np.random.random((3, 4)),
            np.array([[1, -2, 6, 5], [-2, -3, -1, -5], [1, 2, 4, 5]]),
        ]
        predictions = np.array([[1, 2], [1, 3], [1, 4]])

        shaps, segments, labels, ranges = prepare_force_plot_data(
            shap_values, self.iris[:4], predictions, 1, top_n_features=5
        )

        self.assertEqual(len(shaps), 3)
        self.assertEqual(len(shaps[0][0]), 3)
        self.assertEqual(len(shaps[0][1]), 1)
        self.assertEqual(len(shaps[1][0]), 0)
        self.assertEqual(len(shaps[1][1]), 4)
        self.assertEqual(len(shaps[2][0]), 4)
        self.assertEqual(len(shaps[2][1]), 0)

        self.assertEqual(len(segments), 3)
        self.assertEqual(len(segments[0][0]), 3)
        self.assertEqual(len(segments[0][1]), 1)
        self.assertEqual(len(segments[1][0]), 0)
        self.assertEqual(len(segments[1][1]), 4)
        self.assertEqual(len(segments[2][0]), 4)
        self.assertEqual(len(segments[2][1]), 0)

        self.assertEqual(len(labels), 3)
        self.assertEqual(len(labels[0][0]), 3)
        self.assertEqual(len(labels[0][1]), 1)
        self.assertEqual(len(labels[1][0]), 0)
        self.assertEqual(len(labels[1][1]), 4)
        self.assertEqual(len(labels[2][0]), 4)
        self.assertEqual(len(labels[2][1]), 0)

        self.assertEqual(len(ranges), 3)

    def test_prepare_force_plot_no_top_n_features(self):
        shap_values = [
            np.random.random((3, 4)),
            np.array([[1, -2, 6, 5], [-2, -3, -1, -5], [1, 2, 4, 5]]),
        ]
        predictions = np.array([[1, 2], [1, 3], [1, 4]])

        shaps, segments, labels, ranges = prepare_force_plot_data(
            shap_values, self.iris[:4], predictions, 1
        )

        self.assertEqual(len(shaps), 3)
        self.assertEqual(len(shaps[0][0]), 3)
        self.assertEqual(len(shaps[0][1]), 1)
        self.assertEqual(len(shaps[1][0]), 0)
        self.assertEqual(len(shaps[1][1]), 4)
        self.assertEqual(len(shaps[2][0]), 4)
        self.assertEqual(len(shaps[2][1]), 0)

        self.assertEqual(len(segments), 3)
        self.assertEqual(len(segments[0][0]), 3)
        self.assertEqual(len(segments[0][1]), 1)
        self.assertEqual(len(segments[1][0]), 0)
        self.assertEqual(len(segments[1][1]), 4)
        self.assertEqual(len(segments[2][0]), 4)
        self.assertEqual(len(segments[2][1]), 0)

        self.assertEqual(len(labels), 3)
        self.assertEqual(len(labels[0][0]), 3)
        self.assertEqual(len(labels[0][1]), 1)
        self.assertEqual(len(labels[1][0]), 0)
        self.assertEqual(len(labels[1][1]), 4)
        self.assertEqual(len(labels[2][0]), 4)
        self.assertEqual(len(labels[2][1]), 0)

        self.assertEqual(len(ranges), 3)

    def test_prepare_force_plot_data_zero_shap(self):
        """
        prepare_force_plot_data should remove all values and variables that
        have SHAP values 0. Test if it works
        """
        shap_values = [
            np.random.random((3, 4)),
            np.array([[1, -2, 6, 0], [-2, -3, 0, -5], [1, 0, 4, 5]]),
        ]
        predictions = np.array([[1, 2], [1, 3], [1, 4]])

        shaps, segments, labels, ranges = prepare_force_plot_data(
            shap_values, self.iris[:4], predictions, 1
        )
        self.assertListEqual(
            [([6, 1], [-2]), ([], [-5, -3, -2]), ([5, 4, 1], [])], shaps
        )
        self.assertListEqual(
            [
                ([(2, -4), (-4, -5)], [(2, 4)]),
                ([], [(3, 8), (8, 11), (11, 13)]),
                ([(4, -1), (-1, -5), (-5, -6)], []),
            ],
            segments,
        )
        self.assertListEqual(
            [
                (
                    [("petal length", 1.4), ("sepal length", 5.1)],
                    [("sepal width", 3.5)],
                ),
                (
                    [],
                    [
                        ("petal width", 0.2),
                        ("sepal width", 3.0),
                        ("sepal length", 4.9),
                    ],
                ),
                (
                    [
                        ("petal width", 0.2),
                        ("petal length", 1.3),
                        ("sepal length", 4.7),
                    ],
                    [],
                ),
            ],
            labels,
        )
        self.assertListEqual([(-5, 4), (3, 13), (-6, 4)], ranges)

    def test_no_class(self):
        iris_no_class = Table.from_table(
            Domain(self.iris.domain.attributes), self.iris
        )

        # tree
        model = RandomForestLearner()(self.iris)
        shap_values, _, sample_mask, _ = compute_shap_values(
            model, iris_no_class, iris_no_class
        )

        self.assertTupleEqual(self.iris.X.shape, shap_values[0].shape)
        self.assertTupleEqual((len(self.iris),), sample_mask.shape)

        # kernel
        model = LogisticRegressionLearner()(self.iris)
        shap_values, _, sample_mask, _ = compute_shap_values(
            model, iris_no_class, iris_no_class
        )

        self.assertTupleEqual(self.iris.X.shape, shap_values[0].shape)
        self.assertTupleEqual((len(self.iris),), sample_mask.shape)


if __name__ == "__main__":
    unittest.main()
