import unittest
from itertools import chain

import numpy as np

from orangecontrib.prototypes.neighbor_joining import (
    TreeNode,
    get_leaves,
    neighbor_joining_core,
    reorder_children,
    to_newick,
)

from orangecontrib.prototypes.neighbor_joining_adapter import (
    treenode_to_orange_tree,
)

def flatten(seq):
    return chain(*seq)


def lower_to_square(lower):
    size = len(lower)
    expected = size * (size - 1) // 2
    if len(list(flatten(lower))) != expected:
        raise ValueError("lower diagonal must contain n * (n - 1) / 2 values")

    matrix = np.zeros((size, size), dtype=float)
    for i, row in enumerate(lower):
        if len(row) != i:
            raise ValueError("lower diagonal rows must have lengths 0..n-1")
        matrix[i, :i] = row
        matrix[:i, i] = row
    return matrix


def walk(node):
    yield node
    for child in node.children:
        yield from walk(child)


def leaf_depths(root):
    depths = {}

    def collect(node, depth):
        if node.is_leaf():
            depths[node.name] = depth
        for child in node.children:
            collect(child, depth + child.length)

    collect(root, 0.0)
    return depths


def pairwise_leaf_distances(root):
    parent = {}
    depth = {}

    def collect(node, distance):
        depth[node] = distance
        for child in node.children:
            parent[child] = node
            collect(child, distance + child.length)

    collect(root, 0.0)
    nodes_by_name = {
        node.name: node for node in depth
        if node.is_leaf()
    }

    def common_ancestor(left, right):
        ancestors = set()
        node = left
        while node is not None:
            ancestors.add(node)
            node = parent.get(node)
        node = right
        while node not in ancestors:
            node = parent.get(node)
        return node

    distances = {}
    labels = sorted(nodes_by_name)
    for i, left_label in enumerate(labels):
        for right_label in labels[i + 1:]:
            left = nodes_by_name[left_label]
            right = nodes_by_name[right_label]
            ancestor = common_ancestor(left, right)
            distances[left_label, right_label] = (
                depth[left] + depth[right] - 2 * depth[ancestor]
            )
    return distances


def orange_preorder(tree):
    yield tree
    for branch in tree.branches:
        yield from orange_preorder(branch)


def orange_leaves(tree):
    if tree.is_leaf:
        return [tree]
    return [
        leaf
        for branch in tree.branches
        for leaf in orange_leaves(branch)
    ]


class NeighborJoiningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.four_labels = ["A", "B", "C", "D"]
        cls.four_lower = [
            [],
            [5],
            [9, 10],
            [9, 10, 8],
        ]
        cls.four_distances = lower_to_square(cls.four_lower)

        cls.clustered_labels = ["A", "B", "C", "D", "E", "F"]
        cls.clustered_lower = [
            [],
            [2],
            [4, 4],
            [4, 4, 2],
            [7, 7, 7, 7],
            [7, 7, 7, 7, 4],
        ]
        cls.clustered_distances = lower_to_square(cls.clustered_lower)

        cls.four_tree = neighbor_joining_core(
            cls.four_distances, cls.four_labels)
        cls.clustered_tree = neighbor_joining_core(
            cls.clustered_distances, cls.clustered_labels)


class TestNeighborJoiningCore(NeighborJoiningTest):
    def test_two_items_are_joined_by_split_edge(self):
        distances = lower_to_square([[], [4]])

        root = neighbor_joining_core(distances, ["A", "B"])

        self.assertEqual(root.name, "Root")
        self.assertEqual([child.name for child in root.children], ["A", "B"])
        self.assertEqual([child.length for child in root.children], [2, 2])
        self.assertEqual(to_newick(root), "(A:2.000000,B:2.000000);")

    def test_known_four_taxa_example(self):
        self.assertEqual(
            to_newick(self.four_tree),
            "((A:2.000000,B:3.000000)Node1:1.500000,"
            "(C:4.000000,D:4.000000)Node2:1.500000);",
        )
        self.assertEqual(pairwise_leaf_distances(self.four_tree), {
            ("A", "B"): 5.0,
            ("A", "C"): 9.0,
            ("A", "D"): 9.0,
            ("B", "C"): 10.0,
            ("B", "D"): 10.0,
            ("C", "D"): 8.0,
        })

    def test_all_labels_are_preserved(self):
        self.assertCountEqual(get_leaves(self.clustered_tree),
                              self.clustered_labels)
        self.assertEqual(
            sum(node.is_leaf() for node in walk(self.clustered_tree)),
            len(self.clustered_labels),
        )

    def test_topology_groups_nearest_neighbors(self):
        clades = {
            frozenset(get_leaves(node))
            for node in walk(self.clustered_tree)
            if not node.is_leaf()
        }

        self.assertIn(frozenset({"A", "B"}), clades)
        self.assertIn(frozenset({"C", "D"}), clades)
        self.assertIn(frozenset({"E", "F"}), clades)

    def test_reorder_children_prefers_closer_leaf_boundaries(self):
        root = neighbor_joining_core(
            self.clustered_distances, self.clustered_labels)

        reorder_children(
            root,
            self.clustered_distances,
            dict(zip(self.clustered_labels, range(len(self.clustered_labels)))),
        )

        ordered_leaves = get_leaves(root)
        self.assertEqual(set(ordered_leaves), set(self.clustered_labels))
        for left, right in (("A", "B"), ("C", "D"), ("E", "F")):
            self.assertEqual(
                abs(ordered_leaves.index(left) - ordered_leaves.index(right)),
                1,
            )

    def test_to_newick_sorts_children_for_deterministic_output(self):
        root = TreeNode("Root")
        root.add_child(TreeNode("B", 2))
        root.add_child(TreeNode("A", 1))

        self.assertEqual(to_newick(root), "(A:1.000000,B:2.000000);")

    def test_invalid_input_raises(self):
        with self.assertRaisesRegex(ValueError, "square"):
            neighbor_joining_core(np.zeros((2, 3)), ["A", "B"])

        with self.assertRaisesRegex(ValueError, "len"):
            neighbor_joining_core(np.zeros((2, 2)), ["A"])

    def test_deterministic(self):
        first = to_newick(neighbor_joining_core(
            self.clustered_distances, self.clustered_labels))
        second = to_newick(neighbor_joining_core(
            self.clustered_distances, self.clustered_labels))

        self.assertEqual(first, second)


@unittest.skipIf(treenode_to_orange_tree is None, "Orange is not installed")
class TestNeighborJoiningAdapter(NeighborJoiningTest):
    def test_converts_to_orange_tree_with_leaf_metadata(self):
        tree = treenode_to_orange_tree(
            self.four_tree,
            dict(zip(self.four_labels, range(len(self.four_labels)))),
        )

        leaf_values = [leaf.value for leaf in orange_leaves(tree)]
        self.assertEqual(tree.value.first, 0)
        self.assertEqual(tree.value.last, len(self.four_labels))
        self.assertEqual(tree.value.members, tuple(range(len(self.four_labels))))
        self.assertCountEqual([value.index for value in leaf_values], range(4))
        self.assertTrue(all(value.last == value.first + 1
                            for value in leaf_values))
        self.assertTrue(all(node.value.height >= 0
                            for node in orange_preorder(tree)))

    def test_orange_heights_match_distance_to_furthest_leaf(self):
        depths = leaf_depths(self.four_tree)
        max_depth = max(depths.values())

        tree = treenode_to_orange_tree(
            self.four_tree,
            dict(zip(self.four_labels, range(len(self.four_labels)))),
        )

        self.assertAlmostEqual(tree.value.height, max_depth)
        for leaf in orange_leaves(tree):
            label = self.four_labels[leaf.value.index]
            self.assertAlmostEqual(
                leaf.value.height,
                max_depth - depths[label],
            )


if __name__ == "__main__":
    unittest.main()