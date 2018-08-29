"""
==========
Prototypes
==========

Various widgets not polished for regular use.

"""
import sysconfig

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

    # Documentation included in wheel
    # Correct DATA_FILES entry is needed in setup.py and documentation has to be
    # built before the wheel is created.
    ("{}/help/orange3-prototypes/index.html".format(sysconfig.get_path("data")),
     None),

    # Online documentation url, used when the local documentation is available.
    # Url should point to a page with a section Widgets. This section should
    # includes links to documentation pages of each widget. Matching is
    # performed by comparing link caption to widget name.
    ("https://orange3-prototypes.readthedocs.io/en/latest/", "")
)
