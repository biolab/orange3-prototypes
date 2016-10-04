import unittest
from unittest.mock import MagicMock

from Orange.data import Table
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
    pass


@Learner.data_handler(target_type=InputTypes.DISCRETE)
class DiscreteLearner:
    LEARNER = 'discrete'


@Learner.data_handler(target_type=InputTypes.CONTINUOUS)
class ContinuousLearner:
    LEARNER = 'continuous'


# Test class
class TestMultiInput(unittest.TestCase):
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
        obj.handle_new_data = MagicMock()

        obj.set_data(data)

        obj.handle_new_data.assert_called_once_with(data)

    def test_calling_non_trigger_function_does_not_update_data(self):
        """Calling a function that is not a registered trigger function should
        not cause the class to update its handler selection parameters."""
        obj = Widget()
        obj.handle_new_data = MagicMock()

        obj.dummy_method()

        obj.handle_new_data.assert_not_called()

    def test_handle_new_data_with_discrete_target(self):
        obj = MultiInputMixin()
        data = Table('iris')

        obj.handle_new_data(data)

        self.assertEqual(obj.target_type, InputTypes.DISCRETE)

    def test_handle_new_data_with_continuous_target(self):
        obj = MultiInputMixin()
        data = Table('housing')

        obj.handle_new_data(data)

        self.assertEqual(obj.target_type, InputTypes.CONTINUOUS)

    def test_calling_continuous_handler_method(self):
        """Calling a method with a defined continuous handler should call the
        method on that handler class."""
        obj = Widget()
        obj.target_type = InputTypes.CONTINUOUS

        Discrete.simple_method.__get__ = MagicMock()
        Continuous.simple_method.__get__ = MagicMock()

        obj.simple_method()

        Discrete.simple_method.__get__.assert_not_called()
        Continuous.simple_method.__get__.assert_called_once_with(obj)

    def test_calling_discrete_handler_method(self):
        """Calling a method with a defined discrete handler should call the
        method on that handler class."""
        obj = Widget()
        obj.target_type = InputTypes.DISCRETE

        Discrete.simple_method.__get__ = MagicMock()
        Continuous.simple_method.__get__ = MagicMock()

        obj.simple_method()

        Discrete.simple_method.__get__.assert_called_once_with(obj)
        Continuous.simple_method.__get__.assert_not_called()

    def test_accessing_handler_property(self):
        """Accessing an attribute on the handler should fetch the attribute
        from the appropriate handler."""
        obj = Widget()

        obj.target_type = InputTypes.CONTINUOUS
        self.assertEqual(obj.attribute, Continuous.attribute)

        obj.target_type = InputTypes.DISCRETE
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

        obj.target_type = InputTypes.DISCRETE
        self.assertEqual(obj.LEARNER, 'discrete')

        obj.target_type = InputTypes.CONTINUOUS
        self.assertEqual(obj.LEARNER, 'continuous')


class TestMultiInputWidgetIntegration(unittest.TestCase):
    def test_calling_reserved_properties_calls_base_class_properties(self):
        """Some properties should always be taken from the base class and
        ignored in the handler class."""
        obj = Widget()
        self.assertEqual(obj.__class__, Widget)
