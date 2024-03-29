import os
import unittest


def suite(loader=None, pattern='test*.py'):
    if loader is None:
        loader = unittest.TestLoader()
    if pattern is None:
        pattern = 'test*.py'

    test_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(test_dir)

    ts = unittest.TestSuite()
    for folder in [test_dir]:
        ts.addTests(loader.discover(folder, pattern, top_level_dir=project_dir))

    return ts


def load_tests(loader, tests, pattern):
    return suite(loader, pattern)


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
