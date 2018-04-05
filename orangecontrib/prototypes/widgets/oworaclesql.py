import datetime

from AnyQt.QtWidgets import QSizePolicy, QPlainTextEdit, QLineEdit

from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Msg, Output
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, TimeVariable
from PyQt5.QtCore import QTimer

try:
    import cx_Oracle
except ImportError:
    cx_Oracle = None

        
class OWOracleSQL(OWWidget):
    name = "Oracle SQL"
    description = "Select data from oracle databases"
    icon = "icons/OracleSQL.svg"

    class Error(OWWidget.Error):
        no_backends = Msg("Please install cx_Oracle package. It is either missing or not working properly")

    class Outputs:
        data = Output("Data", Table)

    autocommit = settings.Setting(False, schema_only=True)
    savedQuery = settings.Setting(None, schema_only=True)
    savedUsername = settings.Setting(None, schema_only=True)
    savedPwd = settings.Setting(None, schema_only=True)
    savedDB = settings.Setting(None, schema_only=True)

    def __init__(self):
        super().__init__()

        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.domain = None
        self.data = None
        self.query = ''
        if self.savedQuery is not None:
            self.query = self.savedQuery
        self.username = ''
        if self.savedUsername is not None:
            self.username = self.savedUsername
        self.password = ''
        if self.savedPwd is not None:
            self.password = self.savedPwd
        self.database = ''
        if self.savedDB is not None:
            self.database = self.savedDB

        #Control Area layout
        self.connectBox = gui.widgetBox(self.controlArea, "Database connection")
        self.connectBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        #Database
        self.userLabel = gui.label(self.connectBox, self, 'User name')
        self.connectUser = QLineEdit(self.username, self)
        self.connectBox.layout().addWidget(self.connectUser)
        self.passwordLabel = gui.label(self.connectBox, self, 'Password')
        self.connectPassword = QLineEdit(self.password, self)
        self.connectPassword.setEchoMode(QLineEdit.Password)
        self.connectBox.layout().addWidget(self.connectPassword)
        self.dbLabel = gui.label(self.connectBox, self, 'Database')
        self.connectDB = QLineEdit(self.database, self)
        self.connectBox.layout().addWidget(self.connectDB)
        self.runSQL = gui.auto_commit(self.connectBox, self, 'autocommit',
                                      label='Run SQL', commit=self.commit)
        # query
        self.sqlBox = gui.widgetBox(self.mainArea, "SQL")
        self.queryTextEdit = QPlainTextEdit(self.query, self)
        self.sqlBox.layout().addWidget(self.queryTextEdit)

        QTimer.singleShot(0, self.commit)

    def handleNewSignals(self):
        self._invalidate()
        
    def countUniques(self,lst):
        return len(set([x for x in lst if x is not None]))

    def setOfUniques(self,lst):
        return sorted(set([x for x in lst if x is not None]))

    def dateToStr(self,lst):
        return [str(x) if x is not None else x for x in lst]

    def commit(self):
        if cx_Oracle is None:
            data = []
            columns = []
            self.Error.no_backends()
            username = None
            password = None
            database = None
            query = None            
        else:   
            username = self.connectUser.text()
            password = self.connectPassword.text()
            database = self.connectDB.text()
            con = cx_Oracle.connect(username+"/"+password+"@"+database)
            query = self.queryTextEdit.toPlainText()
            
            cur = con.cursor()
            cur.execute(query)
            data = cur.fetchall()
            columns = [i[0] for i in cur.description]      
        
        data_tr = list(zip(*data))

        def create_variable(column_name, column_data):
            continuous_types = (int, float, type(None))
            datetime_types = (datetime.datetime, type(None))
            column_name = str(column_name)

            if all(isinstance(x, continuous_types) for x in column_data):
                return ContinuousVariable(column_name)

            if all(isinstance(x, datetime_types) for x in column_data):
                return TimeVariable(column_name)

            if self.countUniques(column_data) < 101:
                return DiscreteVariable(str(column_name), self.setOfUniques(column_data))

            return DiscreteVariable(str(column_name), self.setOfUniques(column_data))

        featurelist = [create_variable(column_name, column_data)
                       for column_name, column_data in zip(columns, data_tr)]

        data_tr = [
            self.dateToStr(data) if isinstance(var, TimeVariable) else data
            for (var, data) in zip(featurelist, data_tr)]
        data = list(zip(*data_tr))

        orangedomain = Domain(featurelist)
        orangetable = Table(orangedomain,data)
        
        self.Outputs.data.send(orangetable)
        self.savedQuery = query
        self.savedUsername = username
        self.savedPwd = password
        self.savedDB = database

    def _invalidate(self):
        self.commit()
 

def main():
    from AnyQt.QtWidgets import QApplication
    app = QApplication([])
    w = OWOracleSQL()
    w.show()

    app.exec()


if __name__ == '__main__':
    main()
