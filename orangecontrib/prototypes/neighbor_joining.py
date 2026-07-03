import numpy as np

from typing import List, Optional


class TreeNode:
    """
    Minimal tree node for phylogenetic trees.

    name   : taxon or internal node name
    length : branch length to parent (root usually 0)
    children : list of child nodes (empty for leaves)
    """

    __slots__ = ("name", "length", "children", "_abs_height", "_dist_from_root")

    def __init__(self, name: Optional[str] = None, length: float = 0.0):
        self.name: Optional[str] = name
        self.length: float = float(length)
        self.children: List["TreeNode"] = []

    def is_leaf(self) -> bool:
        return not self.children

    def add_child(self, child: "TreeNode"):
        self.children.append(child)


def neighbor_joining_core(D: np.ndarray, labels: List[str]) -> TreeNode:
    """
    Core implementation of Neighbor Joining.

    Parameters
    ----------
    D : np.ndarray (n x n)
        Symmetric distance matrix with zeros on the diagonal.
    labels : list[str]
        Taxon labels of length n.

    Returns
    -------
    root : TreeNode
        Root of the reconstructed (artificially rooted) tree.

    Notes
    -----
    NJ is inherently unrooted. We root the final tree by splitting
    the last remaining edge in half to obtain a binary rooted tree
    suitable for Newick export and comparison.
    """
    D = np.array(D, dtype=float)
    n = D.shape[0]
    if D.shape[1] != n:
        raise ValueError("D must be a square matrix")
    if len(labels) != n:
        raise ValueError("len(labels) must match matrix size")

    nodes: List[TreeNode] = [
        TreeNode(name=str(labels[i]), length=0.0) for i in range(n)
    ]

    next_internal_id = 1

    while n > 2:
        r = D.sum(axis=1)

        Q = (n - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)

        idx_min = np.argmin(Q)
        u, v = divmod(idx_min, n)
        if u > v:
            u, v = v, u

        delta = (r[u] - r[v]) / (n - 2)
        limb_u = 0.5 * (D[u, v] + delta)
        limb_v = D[u, v] - limb_u

        m_name = f"Node{next_internal_id}"
        next_internal_id += 1
        m = TreeNode(name=m_name, length=0.0)

        node_u = nodes[u]
        node_v = nodes[v]
        node_u.length = limb_u
        node_v.length = limb_v
        m.add_child(node_u)
        m.add_child(node_v)

        idxs = [i for i in range(n) if i not in (u, v)]
        d_u = D[u, idxs]
        d_v = D[v, idxs]
        d_uv = D[u, v]
        d_m = 0.5 * (d_u + d_v - d_uv)

        n_new = n - 1
        D_new = np.zeros((n_new, n_new), dtype=float)

        D_reduced = D[np.ix_(idxs, idxs)]  # (n-2, n-2)
        D_new[:n_new - 1, :n_new - 1] = D_reduced

        last_idx = n_new - 1
        D_new[last_idx, last_idx] = 0.0
        D_new[last_idx, :last_idx] = d_m
        D_new[:last_idx, last_idx] = d_m

        D = D_new
        n = n_new

        new_nodes = [nodes[i] for i in idxs]
        new_nodes.append(m)
        nodes = new_nodes

    # Final two clusters: create root in the middle of the last edge
    d_last = D[0, 1]
    root = TreeNode(name="Root", length=0.0)

    nodes[0].length = d_last / 2.0
    nodes[1].length = d_last / 2.0
    root.add_child(nodes[0])
    root.add_child(nodes[1])

    return root


def get_leaves(node: TreeNode) -> List[str]:
    """
    Get the list of leaf names under a given node.
    """
    if node.is_leaf():
        return [node.name]
    leaves = []
    for child in node.children:
        leaves.extend(get_leaves(child))
    return leaves


def reorder_children(node, D, label_to_index):
    """
    Recursively reorder children of internal nodes.
    """
    if node.is_leaf():
        return

    for child in node.children:
        reorder_children(child, D, label_to_index)

    if len(node.children) == 2:
        left, right = node.children
        left_leaves = get_leaves(left)
        right_leaves = get_leaves(right)

        d_current = D[
            label_to_index[left_leaves[-1]], label_to_index[right_leaves[0]]
        ]
        d_swapped = D[
            label_to_index[left_leaves[0]], label_to_index[right_leaves[0]]
        ]

        if d_swapped < d_current:
            node.children[0], node.children[1] = right, left


def to_newick(node: TreeNode) -> str:
    """
    Convert a rooted TreeNode tree to a Newick string.
    """

    def _rec(n: TreeNode) -> str:
        if n.is_leaf():
            name = n.name if n.name is not None else ""
            return f"{name}:{n.length:.6f}"
        else:
            parts = []
            # deterministic order by child name if present
            for ch in sorted(n.children,
                             key=lambda x: "" if x.name is None else x.name):
                parts.append(_rec(ch))
            inner = ",".join(parts)
            if n.name is None or n.name == "Root":
                return f"({inner})"
            else:
                return f"({inner}){n.name}:{n.length:.6f}"

    return _rec(node) + ";"
