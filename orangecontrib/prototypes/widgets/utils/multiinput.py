from enum import Enum

from Orange.data import Table


class InputTypes(Enum):
    NONE, DISCRETE, CONTINUOUS = range(3)


class MultiInputMixinMeta(type):
    """Ensure that each subclass of the `MultiInput` has their own registry of
    data handlers."""

    def __new__(mcs, *args, **kwargs):
        return super().__new__(mcs, *args, **kwargs)

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls.handlers = {}
        cls.trigger = 'set_data'


class MultiInputMixin(metaclass=MultiInputMixinMeta):

    def __init__(self):
        super().__init__()
        self.target_type = InputTypes.NONE

    @classmethod
    def data_handler(cls, **kwargs):

        target_type = kwargs.get('target_type', InputTypes.NONE)

        def decorator(target):
            cls.register(target_type, target)
            return target

        return decorator

    @classmethod
    def register(cls, target_type, handler):
        cls.handlers[target_type] = handler

    def handle_new_data(self, data):
        if not isinstance(data, Table):
            raise Exception('Should only be used with Orange.data.Table')

        if data.domain.has_discrete_class:
            self.target_type = InputTypes.DISCRETE
        elif data.domain.has_continuous_class:
            self.target_type = InputTypes.CONTINUOUS

    def __getattribute__(self, item):
        ga = super().__getattribute__

        # Handle trigger functions
        trigger = ga('trigger')
        if item == trigger:
            def wrapped(data):
                ga('handle_new_data')(data)
                ga('set_data')(data)
            return wrapped

        # Handle any handler functions
        handlers = ga('handlers')
        target_type = ga('target_type')
        # If the target type has a registered handler, use the accessed
        # property of the handler instead, if it exists
        if target_type in handlers and hasattr(handlers[target_type], item):
            # Get the attribute from the handler class
            attr = getattr(handlers[target_type], item)
            if callable(attr):
                # Bind the current object to the function `self` parameter
                bound_method = attr.__get__(self)
                return bound_method
            else:
                return attr

        # If not a trigger function or handler function was not found
        return ga(item)
