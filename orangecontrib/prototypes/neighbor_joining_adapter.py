"""
Adapter: Neighbor Joining TreeNode -> Orange hierarchical.Tree

This module converts a NJ tree expressed with TreeNode (branch lengths)
into Orange's hierarchical.Tree with absolute heights, suitable for
DendrogramWidget (including non-ultrametric trees).

Intended usage:
    nj_root = neighbor_joining_core(D, labels)
    orange_tree = treenode_to_orange_tree(nj_root)
"""

from dataclasses import dataclass
from typing import Tuple

from Orange.clustering.hierarchical import Tree


@dataclass(frozen=True)
class ClusterValue:
    height: float
    first: int          # leaf-order position (inclusive)
    last: int           # leaf-order position (exclusive)
    index: int          # original row index for leaves, -1 for internal nodes
    range: Tuple[int, int]
    members: Tuple[int, ...]  # original row indices contained in this subtree


def _compute_root_distances(node, current=0.0):
    node._dist_from_root = current
    for ch in node.children:
        _compute_root_distances(ch, current + ch.length)


def _max_root_distance(node) -> float:
    if not node.children:
        return node._dist_from_root
    return max(_max_root_distance(ch) for ch in node.children)


def _assign_orange_heights(node, max_height):
    node._abs_height = max_height - node._dist_from_root
    for ch in node.children:
        _assign_orange_heights(ch, max_height)


def treenode_to_orange_tree(root, label_to_index) -> Tree:
    # Orange dendrograms use height above the leaf baseline. NJ trees are not
    # ultrametric, so leaves can have positive heights after this conversion.
    _compute_root_distances(root)
    max_h = _max_root_distance(root)
    _assign_orange_heights(root, max_h)
    leaf_pos = 0

    def _build(node) -> Tree:
        nonlocal leaf_pos
        if not node.children:  # true leaf
            orig_idx = label_to_index[node.name]
            pos = leaf_pos
            leaf_pos += 1

            val = ClusterValue(
                height=node._abs_height,
                first=pos,
                last=pos + 1,
                index=orig_idx,
                range=(pos, pos + 1),
                members=(orig_idx,),
            )
            return Tree(val, ())

        children = tuple(_build(ch) for ch in node.children)
        first = min(ch.value.first for ch in children)
        last = max(ch.value.last for ch in children)
        members = tuple(i for ch in children for i in ch.value.members)

        val = ClusterValue(
            height=node._abs_height,
            first=first,
            last=last,
            index=-1,
            range=(first, last),
            members=members,
        )
        return Tree(val, children)

    return _build(root)


__all__ = [
    "ClusterValue",
    "treenode_to_orange_tree"
]
