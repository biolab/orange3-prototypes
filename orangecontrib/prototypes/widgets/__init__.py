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

# Location of widget help files.
WIDGET_HELP_PATH = (
# Used for development.
# You still need to build help pages using
# make htmlhelp
# inside doc folder
("{DEVELOP_ROOT}/doc/_build/htmlhelp/index.html", None),

# Online documentation url, used when the local documentation is available.
# Url should point to a page with a section Widgets. This section should
# includes links to documentation pages of each widget. Matching is
# performed by comparing link caption to widget name.
("http://orange3-prototypes.readthedocs.io/en/latest/", "")
)