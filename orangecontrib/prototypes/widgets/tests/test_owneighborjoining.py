# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, protected-access
import unittest
import warnings

import numpy as np

import Orange.misc
from Orange.clustering.hierarchical import leaves
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.distance import Euclidean
from Orange.misc import DistMatrix
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin

from orangecontrib.prototypes.widgets.owneighborjoining import (
    MAX_ITEMS,
    OWNeighborJoining,
)


class TestOWNeighborJoining(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.distances = Euclidean(cls.data)
        cls.distances_cols = Euclidean(cls.data, axis=0)
        cls.signal_name = OWNeighborJoining.Inputs.distances
        cls.signal_data = cls.distances
        cls.same_input_output_domain = False

    def setUp(self):
        self.widget = self.create_widget(OWNeighborJoining)

    def _select_first_non_leaf_cluster(self, widget=None):
        widget = widget or self.widget
        clusters = [
            node for node in widget.dendrogram._items
            if not node.is_leaf
        ]
        self.assertTrue(clusters)
        cluster = widget.dendrogram.item(clusters[0])
        widget.dendrogram.set_selected_items([cluster])
        widget.commit.now()
        return [leaf.value.index for leaf in leaves(cluster.node)]

    def _select_data(self):
        return self._select_first_non_leaf_cluster()

    def test_selection_box_output(self):
        self.send_signal(self.widget.Inputs.distances, self.distances)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.annotated_data))

        self.widget.selection_box.buttons[1].click()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.selected_data))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.annotated_data))

        self.widget.selection_box.buttons[2].click()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.selected_data))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.annotated_data))

    def test_data_input(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertFalse(self.widget.Error.distance_computation_error.is_shown())
        self.assertFalse(self.widget.Error.no_numeric_features.is_shown())
        self.assertIsNotNone(self.widget.root)
        self.assertIsNotNone(self.get_output(self.widget.Outputs.annotated_data))

    def test_discrete_data_input(self):
        domain = Domain([
            DiscreteVariable("a", values=("x", "y")),
            DiscreteVariable("b", values=("u", "v")),
        ])
        data = Table.from_numpy(domain, X=np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]))

        self.send_signal(self.widget.Inputs.data, data)

        self.assertFalse(self.widget.Error.no_numeric_features.is_shown())
        self.assertIsNotNone(self.widget.root)

    def test_no_input_features(self):
        data = Table.from_numpy(Domain([], None), X=np.empty((3, 0)))

        self.send_signal(self.widget.Inputs.data, data)

        self.assertTrue(self.widget.Error.no_numeric_features.is_shown())
        self.assertIsNone(self.widget.root)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        self.assertIsNone(self.get_output(self.widget.Outputs.annotated_data))

    def test_all_zero_inputs(self):
        d = Orange.misc.DistMatrix(np.zeros((10, 10)))
        self.send_signal(self.widget.Inputs.distances, d)
        self.assertFalse(self.widget.Error.tree_construction_error.is_shown())

    def test_annotation_settings_retrieval(self):
        widget = self.widget

        dist_names = Orange.misc.DistMatrix(
            np.zeros((4, 4)), self.data, axis=0)
        dist_no_names = Orange.misc.DistMatrix(np.zeros((10, 10)), axis=1)

        self.send_signal(self.widget.Inputs.distances, self.distances)
        self.assertEqual(widget.annotation, self.data.domain.class_var)

        var2 = self.data.domain[2]
        widget.annotation = var2

        self.send_signal(self.widget.Inputs.distances, dist_no_names)
        self.assertEqual(widget.annotation, "Enumeration")
        widget.annotation = "None"

        self.send_signal(self.widget.Inputs.distances, self.distances)
        self.assertIs(widget.annotation, var2)
        self.send_signal(self.widget.Inputs.distances, dist_no_names)
        self.assertEqual(widget.annotation, "None")

        self.send_signal(self.widget.Inputs.distances, dist_names)
        self.assertEqual(widget.annotation, "Name")
        widget.annotation = "Enumeration"

        self.send_signal(self.widget.Inputs.distances, self.distances)
        self.assertIs(widget.annotation, var2)
        self.send_signal(self.widget.Inputs.distances, dist_no_names)
        self.assertEqual(widget.annotation, "None")
        self.send_signal(self.widget.Inputs.distances, dist_names)
        self.assertEqual(widget.annotation, "Enumeration")
        self.send_signal(self.widget.Inputs.distances, dist_no_names)
        self.assertEqual(widget.annotation, "None")

    def test_domain_loses_class(self):
        self.send_signal(self.widget.Inputs.distances, self.distances)
        data = self.data[:, :4]
        distances = Euclidean(data)
        self.send_signal(self.widget.Inputs.distances, distances)
        self.assertIsNotNone(self.widget.root)

    def test_infinite_distances(self):
        table = Table.from_list(
            Domain(
                [ContinuousVariable("a")],
                [DiscreteVariable("b", values=("y", ))]),
            list(zip([1.79e308, -1e120], "yy"))
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*", RuntimeWarning)
            distances = Euclidean(table)
        self.assertFalse(self.widget.Error.not_finite_distances.is_shown())
        self.send_signal(self.widget.Inputs.distances, distances)
        self.assertTrue(self.widget.Error.not_finite_distances.is_shown())
        self.send_signal(self.widget.Inputs.distances, self.distances)
        self.assertFalse(self.widget.Error.not_finite_distances.is_shown())

    def test_not_symmetric(self):
        w = self.widget
        self.send_signal(w.Inputs.distances, DistMatrix([[1, 2, 3], [4, 5, 6]]))
        self.assertTrue(w.Error.not_symmetric.is_shown())
        self.send_signal(w.Inputs.distances, None)
        self.assertFalse(w.Error.not_symmetric.is_shown())

    def test_empty_matrix(self):
        w = self.widget
        self.send_signal(w.Inputs.distances, DistMatrix([[]]))
        self.assertTrue(w.Error.empty_matrix.is_shown())
        self.send_signal(w.Inputs.distances, None)
        self.assertFalse(w.Error.empty_matrix.is_shown())

    def test_too_many_items(self):
        matrix = DistMatrix(np.zeros((MAX_ITEMS + 1, MAX_ITEMS + 1)))

        self.send_signal(self.widget.Inputs.distances, matrix)

        self.assertTrue(self.widget.Error.too_many_items.is_shown())
        self.assertIsNone(self.widget.root)

    def test_manual_selection_output(self):
        self.send_signal(self.widget.Inputs.distances, self.distances)

        self._select_first_non_leaf_cluster()

        selected = self.get_output(self.widget.Outputs.selected_data)
        annotated = self.get_output(self.widget.Outputs.annotated_data)
        self.assertIsNotNone(selected)
        self.assertIsNotNone(annotated)
        self.assertLessEqual(len(selected), len(self.data))
        self.assertEqual(annotated.domain.variables, selected.domain.variables)

    def test_column_distances(self):
        self.send_signal(self.widget.Inputs.distances, self.distances_cols)

        self._select_first_non_leaf_cluster()

        annotated = self.get_output(self.widget.Outputs.annotated_data)
        self.assertIsNotNone(annotated)
        self.assertEqual(len(annotated.domain.attributes),
                         len(self.data.domain.attributes))
        self.assertTrue(any(
            "cluster" in attr.attributes
            for attr in annotated.domain.attributes
        ))

    def test_many_values_warning(self):
        w = self.widget

        self.send_signal(self.widget.Inputs.distances, self.distances)
        w.top_n = 21
        w.selection_box.buttons[2].click()
        self.assertTrue(w.Warning.many_clusters.is_shown())

        w.top_n = 20
        w.selection_box.buttons[2].click()
        self.assertFalse(w.Warning.many_clusters.is_shown())

        w.top_n = 21
        w.selection_box.buttons[2].click()
        self.assertTrue(w.Warning.many_clusters.is_shown())

        self.send_signal(self.widget.Inputs.distances, None)
        self.assertFalse(w.Warning.many_clusters.is_shown())

    def test_pruning_keeps_widget_operational(self):
        self.send_signal(self.widget.Inputs.distances, self.distances)

        self.widget.pruning = 1
        self.widget.max_depth = 3
        self.widget._invalidate_pruning()

        self.assertIsNotNone(self.widget.root)
        self.assertIsNotNone(self.widget._displayed_root)
        self._select_first_non_leaf_cluster()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.annotated_data))


if __name__ == "__main__":
    unittest.main()
