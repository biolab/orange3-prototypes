#!/usr/bin/env python

from setuptools import setup, find_packages

VERSION = '0.4.7'

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
}

if __name__ == '__main__':
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
        packages=find_packages(),
        package_data={
            "orangecontrib.prototypes.widgets": ["icons/*.svg"],
        },
        install_requires=[
            'Orange3',
        ],
        entry_points=ENTRY_POINTS,
        namespace_packages=['orangecontrib'],
        include_package_data=True,
        zip_safe=False,
    )
