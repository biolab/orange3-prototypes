#!/usr/bin/env python
import sys
from os import walk, path

from setuptools import setup, find_packages

VERSION = '0.21.1'

ENTRY_POINTS = {
    'orange3.addon': (
        'prototypes = orangecontrib.prototypes',
    ),
    # Entry point used to specify packages containing widgets.
    'orange.widgets': (
        # Syntax: category name = path.to.package.containing.widgets
        # Widget category specification can be seen in
        #    orangecontrib/datafusion/widgets/__init__.py
        'Prototypes = orangecontrib.prototypes.widgets',
    ),
    # Register widget help
    "orange.canvas.help": (
    'html-index = orangecontrib.prototypes.widgets:WIDGET_HELP_PATH',
    ),
}

DATA_FILES = [
    # Data files that will be installed outside site-packages folder
]

LONG_DESCRIPTION = open(path.join(path.dirname(__file__), 'README.pypi')).read()


def include_documentation(local_dir, install_dir):
    global DATA_FILES
    # if 'bdist_wheel' in sys.argv and not path.exists(local_dir):
    #     print("Directory '{}' does not exist. "
    #           "Please build documentation before running bdist_wheel."
    #           .format(path.abspath(local_dir)))
    #     sys.exit(0)
    doc_files = []
    for dirpath, dirs, files in walk(local_dir):
        doc_files.append((dirpath.replace(local_dir, install_dir),
                          [path.join(dirpath, f) for f in files]))
    DATA_FILES.extend(doc_files)


if __name__ == '__main__':
    include_documentation('doc/_build/html', 'help/orange3-prototypes')
    setup(
        name="Orange3-Prototypes",
        description="Prototype Orange widgets — only for the brave.",
        version=VERSION,
        author='Bioinformatics Laboratory, FRI UL',
        author_email='contact@orange.biolab.si',
        url='https://github.com/biolab/orange3-prototypes',
        keywords=(
            'orange3 add-on',
        ),
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        packages=find_packages(),
        package_data={
            "orangecontrib.prototypes.widgets": ["icons/*.svg", "tests/*.tab"]
        },
        install_requires=[
            'Orange3>=3.34',
            'numpy>=1.19.5',
            'scipy>=1.9.0',
            'scikit-learn>=1.0.1',
            'pyqtgraph',
            'AnyQt>=0.1.0',
            'pandas>=1.3.0',
            'openai>=1',
            'tiktoken',
        ],
        extras_require={
            'test': ['coverage'],
            'doc': ['sphinx', 'recommonmark', 'sphinx_rtd_theme'],
        },
        entry_points=ENTRY_POINTS,
        namespace_packages=['orangecontrib'],
        include_package_data=True,
        zip_safe=False,
        data_files=DATA_FILES,
        test_suite="orangecontrib.prototypes.tests.suite",
    )
