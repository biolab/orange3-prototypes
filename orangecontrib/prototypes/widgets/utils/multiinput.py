from enum import Enum

from Orange.data import Table
from Orange.widgets.widget import WidgetMetaClass


class InputTypes(Enum):
    NONE, DISCRETE, CONTINUOUS = range(3)


class MultiInputMixinMeta(WidgetMetaClass):
    def __new__(mcs, *args, **kwargs):
        # WidgetMetaClass checks for a `name` attribute to assert whether or
        # not its dealing with a widget
        mcs.name = False
        cls = super().__new__(mcs, *args, **kwargs)
        # Add attributes that every class inheriting the mixin.
        # Every class should store their own handlers, and this saves having
        # to declare these attributes in every class inheriting the mixin
        cls.handlers = {}
        cls.trigger = 'set_data'
        return cls


class MultiInputMixin(metaclass=MultiInputMixinMeta):

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

    @classmethod
    def on_input_change(cls):
        pass

    def handle_new_data(self, data):
        if not isinstance(data, Table) and data is not None:
            raise Exception('Should only be used with Orange.data.Table')

        # Determine the target type of the new data
        if data is None:
            self.target_type = InputTypes.NONE
        elif data.domain.has_discrete_class:
            self.target_type = InputTypes.DISCRETE
        elif data.domain.has_continuous_class:
            self.target_type = InputTypes.CONTINUOUS

        self.bind_handler_attributes()

    def get_attrs_to_override(self):
        """Get a list of attributes that the handlers override.

        Since handlers can inherit from their base widget, we don't want to
        override properties that are inherited from the base class. We only
        want to override properties that need to be changed when changing the
        handler.

        """
        reserved = ['handlers', 'trigger']
        attrs = set()
        for htype in self.handlers:
            handler_attrs = (p for p in self.handlers[htype].__dict__.keys()
                             if not p.startswith('__') and p not in reserved)
            attrs = attrs.union(handler_attrs)
        return attrs

    def bind_handler_attributes(self):
        # If the target_type contains a handler, use that
        if self.target_type in self.handlers:
            handler = self.handlers[self.target_type]
        # Otherwise, we need to reset the methods to the original class
        else:
            handler = self.__class__

        # Replace the instance methods with the handler methods
        for attr_name in self.get_attrs_to_override():
            # If the handler the attr defined, replace the attribute
            if hasattr(handler, attr_name):
                attr = getattr(handler, attr_name)
                # If the attribute is a method, we need to bind it to the
                # instance first
                if callable(attr):
                    attr = attr.__get__(self)
                setattr(self, attr_name, attr)
            # If the handler does not have the attribute, delete it
            else:
                delattr(self, attr_name)

    def __getattribute__(self, item):
        ga = super().__getattribute__

        # Handle trigger functions
        trigger = ga('trigger')
        if item == trigger:
            def wrapped(data):
                ga('handle_new_data')(data)
                ga(trigger)(data)
            return wrapped

        # If not a trigger function or handler function was not found
        return ga(item)
