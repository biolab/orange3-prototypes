"""
==========
Prototypes
==========

Various widgets not polished for regular use.

"""

# Category description for the widget registry

NAME = "Prototypes"

ID = "orange.widgets.prototypes"

DESCRIPTION = "Various widgets not polished for regular use."

BACKGROUND = "#ACE3CE"

ICON = "icons/Category-Prototypes.svg"

PRIORITY = 7

try:
    # Add Orange3-Network to the list of offical add-ons
    from Orange.canvas.application.addons import OFFICIAL_ADDONS
    if "Orange3-Network" not in OFFICIAL_ADDONS:
        OFFICIAL_ADDONS.append("Orange3-Network")
except ImportError:
    # Nothing to patch
    pass
