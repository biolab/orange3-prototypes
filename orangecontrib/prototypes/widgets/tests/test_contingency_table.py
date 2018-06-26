import numpy as np
from Orange.data import DiscreteVariable
from Orange.widgets.tests.base import GuiTest
from Orange.widgets.widget import OWWidget
from AnyQt.QtCore import Qt
import unittest

from orangecontrib.prototypes.widgets.contingency_table import ContingencyTable


class TestContingencyTable(GuiTest):
    def setUp(self):
        self.table = ContingencyTable(OWWidget())
        self.var1 = DiscreteVariable("char", ("a", "b"), ordered=True)
        self.var2 = DiscreteVariable("num", ("0", "1"), ordered=True)

    def _get(self, i, j):
        return self.table._item(i, j).data(Qt.DisplayRole)

    def _get_headers(self):
        return ([self._get(i, 1) for i in range(2, self.table.tablemodel.rowCount() - 1)]
                + [self._get(1, j) for j in range(2, self.table.tablemodel.columnCount() - 1)]
                + [self._get(2, 0), self._get(0, 2)])

    def test_set_variables(self):
        self.table.set_variables(self.var1, self.var2)
        self.assertEqual(self.var1.values + self.var2.values + [self.var1.name, self.var2.name], self._get_headers())

    def _get_data(self):
        return [self._get(i, j)
                for i in range(2, self.table.tablemodel.rowCount() - 1)
                for j in range(2, self.table.tablemodel.columnCount() - 1)]

    def test_update_table(self):
        self.table.set_variables(self.var1, self.var2)
        array = np.array([[1, 2], [3, 4]])
        self.table.update_table(array)
        self.assertEqual([str(x) for x in array.flatten()], self._get_data())


if __name__ == "__main__":
    unittest.main()
