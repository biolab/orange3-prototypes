import unittest
from unittest.mock import MagicMock, Mock

from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.widget import OWWidget

from ..multiinput import MultiInputMixin, InputTypes


# First test class
class Widget(MultiInputMixin):
    my_property = 1
    attribute = 'none'

    def set_data(self, data):
        pass

    def dummy_method(self):
        pass


@Widget.data_handler(target_type=InputTypes.DISCRETE)
class Discrete:
    attribute = 'discrete'

    def simple_method(self, *args):
        pass


@Widget.data_handler(target_type=InputTypes.CONTINUOUS)
class Continuous:
    attribute = 'continuous'

    def simple_method(self, *args):
        pass


# Other test class to make sure things don't collide
class OtherWidget(MultiInputMixin):
    def set_data(self, data):
        pass


@OtherWidget.data_handler(target_type=InputTypes.DISCRETE)
class OtherDiscrete:
    attribute = 'discrete'

    def simple_method(self, *args):
        pass


@OtherWidget.data_handler(target_type=InputTypes.CONTINUOUS)
class OtherContinuous:
    attribute = 'continuous'

    def simple_method(self, *args):
        pass


# Test class with class hierarchy
class BaseWidget:
    LEARNER = None


class Learner(BaseWidget, MultiInputMixin):
    LEARNER = 'general'

    def set_data(self, data):
        pass


@Learner.data_handler(target_type=InputTypes.DISCRETE)
class DiscreteLearner:
    LEARNER = 'discrete'


@Learner.data_handler(target_type=InputTypes.CONTINUOUS)
class ContinuousLearner:
    LEARNER = 'continuous'


# Test class
class TestMultiInput(unittest.TestCase):
    def setUp(self):
        self.iris = Table('iris')
        self.housing = Table('housing')

    def test_each_subclass_of_MultiInput_has_their_own_handlers(self):
        """Each subclass of `MultiInput` should have their own lookup table of
        handlers for each input type."""
        self.assertFalse(Widget.handlers is OtherWidget.handlers)
        # check that Widget has appropriate handlers
        self.assertEqual(Widget.handlers,
                         {InputTypes.DISCRETE: Discrete,
                          InputTypes.CONTINUOUS: Continuous})
        # check that OtherWidget has appropriate handlers
        self.assertEqual(OtherWidget.handlers,
                         {InputTypes.DISCRETE: OtherDiscrete,
                          InputTypes.CONTINUOUS: OtherContinuous})

    def test_calling_trigger_function_triggers_data_change(self):
        """Calling the registered trigger function should update the handler
        selection parameters."""
        data = [1, 2, 3]

        obj = Widget()
        obj.handle_new_data = Mock()

        obj.set_data(data)

        obj.handle_new_data.assert_called_once_with(data)

    def test_calling_non_trigger_function_does_not_update_data(self):
        """Calling a function that is not a registered trigger function should
        not cause the class to update its handler selection parameters."""
        obj = Widget()
        obj.handle_new_data = Mock()

        obj.dummy_method()

        obj.handle_new_data.assert_not_called()

    def test_handle_new_data_with_discrete_target(self):
        obj = Widget()

        obj.handle_new_data(self.iris)

        self.assertEqual(obj.target_type, InputTypes.DISCRETE)

    def test_handle_new_data_with_continuous_target(self):
        obj = Widget()

        obj.handle_new_data(self.housing)

        self.assertEqual(obj.target_type, InputTypes.CONTINUOUS)

    def test_calling_continuous_handler_method(self):
        """Calling a method with a defined continuous handler should call the
        method on that handler class."""
        Discrete.simple_method.__get__ = Mock()
        Continuous.simple_method.__get__ = Mock()

        obj = Widget()
        obj.set_data(Table('housing'))

        obj.simple_method()

        Discrete.simple_method.__get__.assert_not_called()
        Continuous.simple_method.__get__.assert_called_once_with(obj)

    def test_calling_discrete_handler_method(self):
        """Calling a method with a defined discrete handler should call the
        method on that handler class."""
        Discrete.simple_method.__get__ = Mock()
        Continuous.simple_method.__get__ = Mock()

        obj = Widget()
        obj.set_data(self.iris)

        obj.simple_method()

        Discrete.simple_method.__get__.assert_called_once_with(obj)
        Continuous.simple_method.__get__.assert_not_called()

    def test_accessing_handler_property(self):
        """Accessing an attribute on the handler should fetch the attribute
        from the appropriate handler."""
        obj = Widget()

        obj.set_data(self.housing)
        self.assertEqual(obj.attribute, Continuous.attribute)

        obj.set_data(self.iris)
        self.assertEqual(obj.attribute, Discrete.attribute)

    def test_accessing_non_handler_property_takes_base_class_property(self):
        """Accessing an attribute that the base class contains but the handler
        does not should use the base class property."""
        obj = Widget()

        self.assertEqual(obj.my_property, 1)

        obj.target_type = InputTypes.CONTINUOUS
        self.assertEqual(obj.my_property, 1)

    def test_overriding_properties_in_class_hierarchy(self):
        obj = Learner()

        obj.set_data(self.iris)
        self.assertEqual(obj.LEARNER, 'discrete')

        obj.set_data(self.housing)
        self.assertEqual(obj.LEARNER, 'continuous')

    def test_calling_reserved_properties_calls_base_class_properties(self):
        """Some properties should always be taken from the base class and
        ignored in the handler class."""
        obj = Widget()
        self.assertEqual(obj.__class__, Widget)

    def test_delete_attr_from_class_when_not_present_in_base_class(self):
        """If the data is set to None, and the handler added a method that was
        not initially present on the base class, then in order to restore the
        base class, we need to delete the attribute from the instance."""
        obj = Widget()
        # Bind a method that the base class does not have
        obj.set_data(self.iris)

        obj.set_data(None)
        self.assertFalse(hasattr(obj, 'simple_method'))


# Test integration with Orange widgets
class OWMulti(OWWidget, MultiInputMixin):
    pass


class TestMultiInputWidgetIntegration(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWMulti)

    def test_is_instantiable(self):
        self.assertIsInstance(self.widget, OWMulti)
