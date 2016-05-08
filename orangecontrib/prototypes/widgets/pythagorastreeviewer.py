"""
Pythagoras tree viewer for visualizing tree strucutres.

The pythagoras tree viewer widget is a widget that can be plugged into any
existing widget given a tree adapter instance. It is simply a canvas that takes
and input tree adapter and takes care of all the drawing.

Types
-----
Square : namedtuple (center, length, angle)
    Since Pythagoras trees deal only with squares (they also deal with
    rectangles in the generalized form, but are completely unreadable), this
    is what all the squares are stored as.
Point : namedtuple (x, y)
    Self exaplanatory.

Classes
-------
PythagorasTreeViewer
SquareGraphicsItem

"""
from collections import namedtuple, defaultdict, deque
from functools import lru_cache
from math import pi, sqrt, cos, sin, degrees

import numpy as np
from Orange.preprocess.transformation import Indicator
from PyQt4 import QtCore, QtGui

# Please note that all angles are in radians
from PyQt4.QtCore import Qt

# z index range, increase if needed
Z_STEP = 5000000

Square = namedtuple('Square', ['center', 'length', 'angle'])
Point = namedtuple('Point', ['x', 'y'])


class PythagorasTreeViewer(QtGui.QGraphicsWidget):
    """Pythagoras tree viewer graphics widget.

    Simply pass in a tree adapter instance and a valid scene object, and the
    pythagoras tree will be added.

    Examples
    --------
    Pass tree through constructor.
    >>> tree_view = PythagorasTreeViewer(parent=scene, adapter=tree_adapter)

    Pass tree later through method.
    >>> tree_view = PythagorasTreeViewer(parent=scene)
    >>> tree_view.set_tree(tree_adapter)

    Both these examples set the appropriate tree and add all the squares to the
    widget instance.

    Parameters
    ----------
    parent : QGraphicsItem, optional
        The parent object that the graphics widget belongs to. Should be a
        scene.
    adapter : TreeAdapter, optional
        Any valid tree adapter instance.
    interacitive : bool, optional,
        Specify whether the widget should have an interactive display. This
        means special hover effects, selectable boxes. Default is true.

    """

    def __init__(self, parent=None, adapter=None, depth_limit=0, **kwargs):
        super().__init__(parent)

        # Instance variables
        # The tree adapter parameter will be handled at the end of init
        self._tree_adapter = None
        # The root tree node instance which is calculated inside the class
        self._tree = None

        # Necessary settings that need to be set from the outside
        self._depth_limit = depth_limit
        # Provide a nice green default in case no color function is provided
        self._calc_node_color = lambda _: QtGui.QColor('#297A1F')
        self._get_tooltip = lambda _: 'Tooltip'
        self._interactive = kwargs.get('interactive', True)

        self._square_objects = {}
        self._drawn_nodes = deque()
        self._frontier = deque()

        # Store the items in a item group container
        self._item_group = QtGui.QGraphicsWidget(self)

        # If a tree adapter was passed, set and draw the tree
        if adapter is not None:
            self.set_tree(adapter)

    def set_tree(self, tree_adapter):
        """Pass in a new tree adapter instance and perform updates to canvas.

        Parameters
        ----------
        tree_adapter : TreeAdapter
            The new tree adapter that is to be used.

        Returns
        -------

        """
        self.clear()
        self._tree_adapter = tree_adapter

        if self._tree_adapter is not None:
            self._tree = self._calculate_tree(self._tree_adapter)
            self.set_depth_limit(tree_adapter.max_depth)
            self._draw_tree(self._tree)

    def set_depth_limit(self, depth):
        """Update the drawing depth limit.

        The drawing stops when the depth is GT the limit. This means that at
        depth 0, the root node will be drawn.

        Parameters
        ----------
        depth : int
            The maximum depth at which the nodes can still be drawn.

        Returns
        -------

        """
        self._depth_limit = depth
        self._draw_tree(self._tree)

    def set_node_color_func(self, func):
        """Set the function that will be used to calculate the node colors.

        The function must accept one parameter that represents the label of a
        given node and return the appropriate QColor object that should be used
        for the node.

        Parameters
        ----------
        func : Callable
            func :: label -> QtGui.QColor

        Returns
        -------

        """
        self._calc_node_color = func

    def set_tooltip_func(self, func):
        self._get_tooltip = func

    def target_class_has_changed(self):
        self._update_node_colors()
        self._update_node_tooltips()

    def tooltip_has_changed(self):
        self._update_node_tooltips()

    def _update_node_colors(self):
        """Update all the node colors.

        Should be called when the color method is changed and the nodes need to
        be drawn with the new colors.

        Returns
        -------

        """
        for square in self._get_scene_squares():
            square.setBrush(self._calc_node_color(square.tree_node))

    def _update_node_tooltips(self):
        """Update all the tooltips for the squares."""
        for square in self._get_scene_squares():
            square.setToolTip(self._get_tooltip(square.tree_node))

    def clear(self):
        """Clear the widget state."""
        self._tree_adapter = None
        self._tree = None

        self._clear_scene()

    @staticmethod
    def _calculate_tree(tree_adapter):
        """Actually calculate the tree squares"""
        tree_builder = PythagorasTree()
        return tree_builder.pythagoras_tree(
            tree_adapter, tree_adapter.root, Square(Point(0, 0), 200, -pi / 2)
        )

    def _draw_tree(self, root):
        """Efficiently draw the tree with regards to the depth.

        If using a recursive approach, the tree had to be redrawn every time
        the depth was changed, which was very impractical for larger trees,
        since everything got very slow, very fast.

        In this approach, we use two queues to represent the tree frontier and
        the nodes that have already been drawn. We also store the depth. This
        way, when the max depth is increased, we do not redraw the whole tree
        but only iterate throught the frontier and draw those nodes, and update
        the frontier accordingly.
        When decreasing the max depth, we reverse the process, we clear the
        frontier, and remove nodes from the drawn nodes, and append those with
        depth max_depth + 1 to the frontier, so the frontier doesn't get
        cluttered.

        Parameters
        ----------
        root : TreeNode
            The root tree node.

        Returns
        -------

        """
        if self._tree is None:
            return
        # if this is the first time drawing the tree begin with root
        if not self._drawn_nodes:
            self._frontier.appendleft((0, root))
        # if the depth was decreased, we can clear the frontier, otherwise
        # frontier gets cluttered with non-frontier nodes
        was_decreased = self._depth_was_decreased()
        if was_decreased:
            self._frontier.clear()
        # remove nodes from drawn and add to frontier if limit is decreased
        while self._drawn_nodes:
            depth, node = self._drawn_nodes.pop()
            # check if the node is in the allowed limit
            if depth <= self._depth_limit:
                self._drawn_nodes.append((depth, node))
                break
            if depth == self._depth_limit + 1:
                self._frontier.appendleft((depth, node))

            if node.label in self._square_objects:
                self._square_objects[node.label].hide()

        # add nodes to drawn and remove from frontier if limit is increased
        while self._frontier:
            depth, node = self._frontier.popleft()
            # check if the depth of the node is outside the allowed limit
            if depth > self._depth_limit:
                self._frontier.appendleft((depth, node))
                break
            self._drawn_nodes.append((depth, node))
            self._frontier.extend((depth + 1, c) for c in node.children)

            if node.label in self._square_objects:
                self._square_objects[node.label].show()
            else:
                square_obj = InteractiveSquareGraphicsItem \
                    if self._interactive else SquareGraphicsItem
                self._square_objects[node.label] = square_obj(
                    node,
                    parent=self._item_group,
                    brush=QtGui.QBrush(self._calc_node_color(node)),
                    tooltip=self._get_tooltip(node),
                    zvalue=depth,
                )

    def _depth_was_decreased(self):
        if not self._drawn_nodes:
            return False
        # checks if the max depth was increased from the last change
        depth, node = self._drawn_nodes.pop()
        self._drawn_nodes.append((depth, node))
        # if the right most node in drawn nodes has appropriate depth, it must
        # have been increased
        return depth > self._depth_limit

    def _get_scene_squares(self):
        return filter(lambda i: isinstance(i, SquareGraphicsItem),
                      self.scene().items())

    def _clear_scene(self):
        for item in self._get_scene_squares():
            if item.scene() is self.scene() and self.scene() is not None:
                self.scene().removeItem(item)
        self._frontier.clear()
        self._drawn_nodes.clear()
        self._square_objects.clear()


class SquareGraphicsItem(QtGui.QGraphicsRectItem):
    """Square Graphics Item.

    Square component to draw as components for the non-interactive Pythagoras
    tree.

    Parameters
    ----------
    tree_node : TreeNode
        The tree node the square represents.
    brush : QColor, optional
        The brush to be used as the backgound brush.
    pen : QPen, optional
        The pen to be used for the border.

    """

    def __init__(self, tree_node, parent=None, **kwargs):
        self.tree_node = tree_node
        self.tree_node.graphics_item = self

        center, length, angle = tree_node.square
        self._center_point = center
        self.center = QtCore.QPointF(*center)
        self.length = length
        self.angle = angle
        super().__init__(self._get_rect_attributes(), parent)
        self.setTransformOriginPoint(self.boundingRect().center())
        self.setRotation(degrees(angle))

        self.setBrush(kwargs.get('brush', QtGui.QColor('#297A1F')))
        self.setPen(kwargs.get('pen', QtGui.QPen(QtGui.QColor('#000'))))
        self.setToolTip(kwargs.get('tooltip', 'Tooltip'))

        self.setAcceptHoverEvents(True)
        self.setZValue(kwargs.get('zvalue', 0))
        self.z_step = Z_STEP

        # calculate the correct z values based on the parent
        if self.tree_node.parent != -1:
            p = self.tree_node.parent
            # override root z step
            num_children = len(p.children)
            own_index = [1 if c.label == self.tree_node.label else 0
                         for c in p.children].index(1)

            self.z_step = int(p.graphics_item.z_step / num_children)
            base_z = p.graphics_item.zValue()

            self.setZValue(base_z + own_index * self.z_step)

    def _get_rect_attributes(self):
        """Get the rectangle attributes requrired to draw item.

        Compute the QRectF that a QGraphicsRect needs to be rendered with the
        data passed down in the constructor.
        """
        height = width = self.length
        x = self.center.x() - self.length / 2
        y = self.center.y() - self.length / 2
        return QtCore.QRectF(x, y, height, width)

    def paint(self, painter, option, widget=None):
        # Override the default selected appearance
        if self.isSelected():
            option.state ^= QtGui.QStyle.State_Selected
            rect = self.rect()
            # this must render before overlay due to order in which it's drawn
            super().paint(painter, option, widget)
            painter.save()
            pen = QtGui.QPen(QtGui.QColor(75, 134, 204, 200))
            pen.setWidth(2)
            pen.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(75, 134, 204, 100)))
            painter.drawRect(rect.adjusted(1, 1, -1, -1))
            painter.restore()
        else:
            super().paint(painter, option, widget)


class InteractiveSquareGraphicsItem(SquareGraphicsItem):
    timer = QtCore.QTimer()

    """Interactive square graphics items.

    This is different from the base square graphics item so that it is
    selectable, and it can handle and react to hover events (highlight and
    focus own branch).

    Parameters
    ----------
    tree_node : TreeNode
        The tree node the square represents.
    brush : QColor, optional
        The brush to be used as the backgound brush.
    pen : QPen, optional
        The pen to be used for the border.

    """
    def __init__(self, tree_node, parent=None, **kwargs):
        super().__init__(tree_node, parent, **kwargs)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, True)

        InteractiveSquareGraphicsItem.timer.setSingleShot(True)

    def hoverEnterEvent(self, ev):
        InteractiveSquareGraphicsItem.timer.stop()

        def fnc(graphics_item):
            graphics_item.setZValue(graphics_item.zValue() + Z_STEP)
            graphics_item.setOpacity(1.)

        def other_fnc(graphics_item):
            graphics_item.setOpacity(.1)

        self._propagate_z_values(self, fnc, other_fnc)

    def hoverLeaveEvent(self, ev):

        def fnc(graphics_item):
            graphics_item.setZValue(graphics_item.zValue() - Z_STEP)

        def other_fnc(graphics_item):
            graphics_item.setOpacity(1.)

        InteractiveSquareGraphicsItem.timer.timeout.connect(
            lambda: self._propagate_z_values(self, fnc, other_fnc)
        )
        InteractiveSquareGraphicsItem.timer.start(250)

    def _propagate_z_values(self, graphics_item, fnc, other_fnc):
        self._propagate_to_children(graphics_item, fnc)
        self._propagate_to_parents(graphics_item, fnc, other_fnc)

    def _propagate_to_children(self, graphics_item, fnc):
        # propagate function that handles graphics item to appropriate children
        fnc(graphics_item)
        for c in graphics_item.tree_node.children:
            self._propagate_to_children(c.graphics_item, fnc)

    def _propagate_to_parents(self, graphics_item, fnc, other_fnc):
        # propagate function that handles graphics item to appropriate parents
        if graphics_item.tree_node.parent != -1:
            parent = graphics_item.tree_node.parent.graphics_item
            # handle the non relevant children nodes
            for c in parent.tree_node.children:
                if c != graphics_item.tree_node:
                    self._propagate_to_children(c.graphics_item, other_fnc)
            # handle the parent node
            fnc(parent)
            # propagate up the tree
            self._propagate_to_parents(parent, fnc, other_fnc)


class TreeNode:
    """A node in the tree structure used to represent the tree adapter

    Parameters
    ----------
    label : int
        The label of the tree node, can be looked up in the original tree.
    square : Square
        The square the represents the tree node.
    parent : TreeNode
        The parent of the current node.
    children : tuple of TreeNode, optional
        All the children that belong to this node.

    """

    def __init__(self, label, square, parent, children=()):
        self.label = label
        self.square = square
        self.parent = parent
        self.children = children
        self.graphics_item = None


class PythagorasTree:
    """Pythagoras tree.

    Contains all the logic that converts a given tree adapter to a tree
    consisting of node classes.

    """

    def __init__(self):
        # store the previous angles of each square children so that slopes can
        # be computed
        self._slopes = defaultdict(list)

    def pythagoras_tree(self, tree, node, square):
        """Get the Pythagoras tree representation in a graph like view.

        Constructs a graph using TreeNode into a tree structure. Each node in
        graph contains the information required to plot the the tree.

        Parameters
        ----------
        tree : TreeAdapter
            A tree adapter instance where the original tree is stored.
        node : int
            The node label, the root node is denoted with 0.
        square : Square
            The initial square which will represent the root of the tree.

        Returns
        -------
        TreeNode
            The root node which contains the rest of the tree.

        """
        # make sure to clear out any old slopes if we are drawing a new tree
        if node == tree.root:
            self._slopes.clear()

        children = tuple(
            self._compute_child(tree, square, child)
            for child in tree.children(node)
        )
        # make sure to pass a reference to parent to each child
        obj = TreeNode(node, square, tree.parent(node), children)
        # mutate the existing data stored in the created tree node
        for c in children:
            c.parent = obj
        return obj

    def _compute_child(self, tree, parent_square, node):
        """Compute all the properties for a single child.

        Parameters
        ----------
        tree : TreeAdapter
            A tree adapter instance where the original tree is stored.
        parent_square : Square
            The parent square of the given child.
        node : int
            The node label of the child.

        Returns
        -------
        TreeNode
            The tree node representation of the given child with the computed
            subtree.

        """
        weight = tree.weight(node)
        # the angle of the child from its parent
        alpha = weight * pi
        # the child side length
        length = parent_square.length * sin(alpha / 2)
        # the sum of the previous anlges
        prev_angles = sum(self._slopes[parent_square])

        center = self._compute_center(
            parent_square, length, alpha, prev_angles
        )
        # the angle of the square is dependent on the parent, the current
        # angle and the previous angles. Subtract PI/2 so it starts drawing at
        # 0rads.
        angle = parent_square.angle - pi / 2 + prev_angles + alpha / 2
        square = Square(center, length, angle)

        self._slopes[parent_square].append(alpha)

        return self.pythagoras_tree(tree, node, square)

    def _compute_center(self, initial_square, length, alpha, base_angle=0):
        """Compute the central point of a child square.

        Parameters
        ----------
        initial_square : Square
            The parent square representation where we will be drawing from.
        length : float
            The length of the side of the new square (the one we are computing
            the center for).
        alpha : float
            The angle that defines the size of our new square (in radians).
        base_angle : float, optional
            If the square we want to find the center for is not the first child
            i.e. its edges does not touch the base square, then we need the
            initial angle that will act as the starting point for the new
            square.

        Returns
        -------
        Point
            The central point to the new square.

        """
        parent_center, parent_length, parent_angle = initial_square
        # get the point on the square side that will be the rotation origin
        t0 = self._get_point_on_square_edge(
            parent_center, parent_length, parent_angle)
        # get the edge point that we will rotate around t0
        square_diagonal_length = sqrt(2 * parent_length ** 2)
        edge = self._get_point_on_square_edge(
            parent_center, square_diagonal_length, parent_angle - pi / 4)
        # if the new square is not the first child, we need to rotate the edge
        if base_angle != 0:
            edge = self._rotate_point(edge, t0, base_angle)

        # rotate the edge point to the correct spot
        t1 = self._rotate_point(edge, t0, alpha)

        # calculate the middle point between the rotated point and edge
        t2 = Point((t1.x + edge.x) / 2, (t1.y + edge.y) / 2)
        # calculate the slope of the new square
        slope = parent_angle - pi / 2 + alpha / 2
        # using this data, we can compute the square center
        return self._get_point_on_square_edge(t2, length, slope + base_angle)

    @staticmethod
    def _rotate_point(point, around, alpha):
        """Rotate a point around another point by some angle.

        Parameters
        ----------
        point : Point
            The point to rotate.
        around : Point
            The point to perform rotation around.
        alpha : float
            The angle to rotate by (in radians).

        Returns
        -------
        Point:
            The rotated point.

        """
        temp = Point(point.x - around.x, point.y - around.y)
        temp = Point(
            temp.x * cos(alpha) - temp.y * sin(alpha),
            temp.x * sin(alpha) + temp.y * cos(alpha)
        )
        return Point(temp.x + around.x, temp.y + around.y)

    @staticmethod
    def _get_point_on_square_edge(center, length, angle):
        """Calculate the central point on the drawing edge of the given square.

        Parameters
        ----------
        center : Point
            The square center point.
        length : float
            The square side length.
        angle : float
            The angle of the square.

        Returns
        -------
        Point
            A point on the center of the drawing edge of the given square.

        """
        return Point(
            center.x + length / 2 * cos(angle),
            center.y + length / 2 * sin(angle)
        )


class TreeAdapter:
    """Base class for tree representation.

    Any subclass should implement the methods listed in this base class. Note
    that some simple methods do not need to reimplemented e.g. is_leaf since
    it that is the opposite of has_children.

    """

    def weight(self, node):
        """Get the weight of the given node.

        The weights of the children always sum up to 1.

        Parameters
        ----------
        node : object
            The label of the node.

        Returns
        -------
        float
            The weight of the node relative to its siblings.

        """
        raise NotImplemented()

    def num_samples(self, node):
        """Get the number of samples that a given node contains.

        Parameters
        ----------
        node : object
            A unique identifier of a node.

        Returns
        -------
        int

        """
        raise NotImplemented()

    def parent(self, node):
        """Get the parent of a given node. Return -1 if the node is the root.

        Parameters
        ----------
        node : object

        Returns
        -------
        object

        """
        raise NotImplemented()

    def has_children(self, node):
        """Check if the given node has any children.

        Parameters
        ----------
        node : object

        Returns
        -------
        bool

        """
        raise NotImplemented()

    def is_leaf(self, node):
        """Check if the given node is a leaf node.

        Parameters
        ----------
        node : object

        Returns
        -------
        object

        """
        return not self.has_children(node)

    def children(self, node):
        """Get all the children of a given node.

        Parameters
        ----------
        node : object

        Returns
        -------
        Iterable[object]
            A iterable object containing the labels of the child nodes.

        """
        raise NotImplemented()

    def get_distribution(self, node):
        """Get the distribution of types for a given node.

        This may be the number of nodes that belong to each different classe in
        a node.

        Parameters
        ----------
        node : object

        Returns
        -------
        Iterable[int, ...]
            The return type is an iterable with as many fields as there are
            different classes in the given node. The values of the fields are
            the number of nodes that belong to a given class inside the node.

        """
        raise NotImplemented()

    def get_impurity(self, node):
        """Get the impurity of a given node.

        Parameters
        ----------
        node : object

        Returns
        -------
        object

        """
        raise NotImplemented()

    def rules(self, node):
        """Get a list of rules that define the given node.

        Parameters
        ----------
        node : object

        Returns
        -------
        Iterable
            A list of tuples in format (attr, relation, value) e.g
            (width, =, 5).

        """
        raise NotImplemented()

    def attribute(self, node):
        """Get the attribute that splits the given tree.

        Parameters
        ----------
        node

        Returns
        -------

        """
        raise NotImplemented()

    def is_root(self, node):
        """Check if a given node is the root node.

        Parameters
        ----------
        node

        Returns
        -------

        """
        return node == self.root

    def leaves(self, node):
        """Get all the leavse that belong to the subtree of a given node.

        Parameters
        ----------
        node

        Returns
        -------

        """
        raise NotImplemented()


    @property
    def max_depth(self):
        """Get the maximum depth that the tree reaches.

        Returns
        -------
        int

        """
        raise NotImplemented()

    @property
    def num_nodes(self):
        """Get the total number of nodes that the tree contains.

        This does not mean the number of samples inside the entire tree, just
        the number of nodes.

        Returns
        -------
        int

        """
        raise NotImplemented()

    @property
    def root(self):
        """Get the label of the root node.

        Returns
        -------
        object

        """
        raise NotImplemented()

    @property
    def domain(self):
        """Get the domain of the given tree.

        The domain contains information about the classes what the tree
        represents.

        Returns
        -------

        """
        raise NotImplemented()


class SklTreeAdapter(TreeAdapter):
    """SklTreeAdapter Class.

    An abstraction on top of the scikit learn classification tree.

    Parameters
    ----------
    model : sklearn.tree._tree.Tree
        The raw sklearn classification tree.
    adjust_weight : function, optional
        If you want finer control over the weights of individual nodes you can
        pass in a function that takes the existsing weight and modifies it.
        The given function must have signture :: Number -> Number

    """

    def __init__(self, model, adjust_weight=lambda x: x):
        self._tree = model.skl_model.tree_
        self._domain = model.domain
        self._adjust_weight = adjust_weight

        # clear memoized functions
        self.weight.cache_clear()
        self._adjusted_child_weight.cache_clear()
        self.parent.cache_clear()

    @lru_cache()
    def weight(self, node):
        return self._adjust_weight(self.num_samples(node)) / \
               self._adjusted_child_weight(self.parent(node))

    @lru_cache()
    def _adjusted_child_weight(self, node):
        """Helps when dealing with adjusted weights.

        It is needed when dealing with non linear weights e.g. when calculating
        the log weight, the sum of logs of all the children will not be equal
        to the log of all the data instances.
        A simple example: log(2) + log(2) != log(4)

        Parameters
        ----------
        node : int
            The label of the node.

        Returns
        -------
        float
            The sum of all of the weights of the children of a given node.

        """
        return sum(self._adjust_weight(self.num_samples(c))
                   for c in self.children(node)) \
            if self.has_children(node) else 0

    def num_samples(self, node):
        return self._tree.n_node_samples[node]

    @lru_cache()
    def parent(self, node):
        for children in (self._tree.children_left, self._tree.children_right):
            try:
                return (children == node).nonzero()[0][0]
            except IndexError:
                continue
        return -1

    def has_children(self, node):
        return self._tree.children_left[node] != -1 \
               or self._tree.children_right[node] != -1

    def children(self, node):
        if self.has_children(node):
            return self._left_child(node), self._right_child(node)
        return ()

    def _left_child(self, node):
        return self._tree.children_left[node]

    def _right_child(self, node):
        return self._tree.children_right[node]

    def get_distribution(self, node):
        return self._tree.value[node]

    def get_impurity(self, node):
        return self._tree.impurity[node]

    @property
    def max_depth(self):
        return self._tree.max_depth

    @property
    def num_nodes(self):
        return self._tree.node_count

    @property
    def root(self):
        return 0

    @property
    def domain(self):
        return self._domain

    def rules(self, node):
        """
        See Also
        --------
        Ported directly from the classification tree graph widget located in
        orange3/Orange/widgets/classify/owclassificationtreegraph.py::rulew
        """
        if node != self.root:
            parent_attr = self.domain.attributes[self.splitting_attribute(
                self.parent(node))]
            parent_attr_cv = parent_attr.compute_value
            is_left_child = self._tree.children_left[self.parent(node)] == node
            pr = self.rules(self.parent(node))
            if isinstance(parent_attr_cv, Indicator) and \
                hasattr(parent_attr_cv.variable, "values"):
                values = parent_attr_cv.variable.values
                attr_name = parent_attr_cv.variable.name
                sign = ["=", "≠"][is_left_child * (len(values) != 2)]
                value = values[abs(parent_attr_cv.value -
                                   is_left_child * (len(values) == 2))]
            else:
                attr_name = parent_attr.name
                sign = [">", "≤"][is_left_child]
                value = "%.3f" % self._tree.threshold[self.parent(node)]
            pr.append((attr_name, sign, value))
            return pr
        else:
            return []

    def attribute(self, node):
        return self.domain.attributes[self.splitting_attribute(node)]

    def splitting_attribute(self, node):
        return self._tree.feature[node]

    def leaves(self, node):
        start, stop = self._subnode_range(node)
        if start == stop:
            # leaf
            return np.array([node], dtype=int)
        else:
            isleaf = self._tree.children_left[start: stop] == -1
            assert np.flatnonzero(isleaf).size > 0
            return start + np.flatnonzero(isleaf)

    def _subnode_range(self, node):
        """Get the range of indices where there are subnodes of the given node.
        Taken from the classificationtreegraph.py"""
        right = left = node
        if self._tree.children_left[left] == -1:
            assert self._tree.children_right[node] == -1
            return node, node
        else:
            left = self._tree.children_left[left]
            # run down to the right most node
            while self._tree.children_right[right] != -1:
                right = self._tree.children_right[right]

            return left, right + 1

    def get_samples_in_node(self, X):
        # TODO figure out and document.

        def assign(node_id, indices):
            if self._tree.children_left[node_id] == -1:
                return [indices]
            else:
                feature_idx = self._tree.feature[node_id]
                thresh = self._tree.threshold[node_id]

                column = X[indices, feature_idx]
                leftmask = column <= thresh
                leftind = assign(self._tree.children_left[node_id],
                                 indices[leftmask])
                rightind = assign(self._tree.children_right[node_id],
                                  indices[~leftmask])
                return list.__iadd__(leftind, rightind)

        N, _ = X.shape

        items = np.arange(N, dtype=int)
        leaf_indices = assign(0, items)
        return leaf_indices
