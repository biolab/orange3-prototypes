import twitter
import networkx as nx
from Orange.widgets import widget, gui
from Orange.widgets.credentials import CredentialManager
import orangecontrib.network as network
from orangecontrib.text import twitter as txt_twitter
from PyQt4 import QtGui, QtCore

class OWTwitterNetwork(widget.OWWidget):
    class APICredentialsDialog(widget.OWWidget):
        name = "Twitter API Credentials"
        want_main_area = False
        resizing_enabled = False

        cm_key = CredentialManager('Twitter API Key')
        cm_secret = CredentialManager('Twitter API Secret')

        key_input = ''
        secret_input = ''

        class Error(widget.OWWidget.Error):
            invalid_credentials = widget.Msg('These credentials are invalid.')

        def __init__(self, parent):
            super().__init__()
            self.parent = parent
            self.credentials = None

            form = QtGui.QFormLayout()
            form.setMargin(5)
            self.key_edit = gui.lineEdit(self, self, 'key_input', controlWidth=400)
            form.addRow('Key:', self.key_edit)
            self.secret_edit = gui.lineEdit(self, self, 'secret_input', controlWidth=400)
            form.addRow('Secret:', self.secret_edit)
            self.controlArea.layout().addLayout(form)

            self.submit_button = gui.button(self.controlArea, self, "OK", self.accept)

            self.load_credentials()

        def load_credentials(self):
            self.key_edit.setText(self.cm_key.key)
            self.secret_edit.setText(self.cm_secret.key)

        def save_credentials(self):
            self.cm_key.key = self.key_input
            self.cm_secret.key = self.secret_input

        def check_credentials(self):
            c = txt_twitter.Credentials(self.key_input, self.secret_input)
            if self.credentials != c:
                if c.valid:
                    self.save_credentials()
                else:
                    c = None
                self.credentials = c

        def accept(self, silent=False):
            if not silent: self.Error.invalid_credentials.clear()
            self.check_credentials()
            if self.credentials and self.credentials.valid:
                super().accept()
            elif not silent:
                self.Error.invalid_credentials()

    name = "Twitter User Graph"
    description = "Create a graph of Twitter users."
    icon = "icons/Twitter.svg"
    priority = 10
    outputs = [("Followers", network.Graph),
               ("Following", network.Graph),
               ("All", network.Graph)]

    want_main_area = False


    def __init__(self):
        super().__init__()

        self.n_all = 0
        self.n_followers = 0
        self.n_following = 0
        self.on_rate_limit = None
        self.api_dlg = self.APICredentialsDialog(self)

        # GUI
        # Set API key button.
        key_dialog_button = gui.button(self.controlArea, self, 'Twitter API Key',
                                       callback=self.open_key_dialog,
                                       tooltip="Set the API key for this widget.")
        key_dialog_button.setFocusPolicy(QtCore.Qt.NoFocus)
        box = gui.widgetBox(self.controlArea, "Info")
        box.layout().addWidget(QtGui.QLabel("Users:"))
        self.users = QtGui.QTextEdit()
        box.layout().addWidget(self.users)
        gui.label(box, self, 'Following: %(n_following)d\n'
                             'Followers: %(n_followers)d\n'
                             'All: %(n_all)d')
        self.button = gui.button(box, self, "Create graph", self.fetch_users)

    def open_key_dialog(self):
        self.api_dlg.exec_()

    def fetch_users(self):
        CONSUMER_KEY = CredentialManager('Twitter API Key').key
        CONSUMER_SECRET = CredentialManager('Twitter API Secret').key
        OAUTH_TOKEN = ''
        OAUTH_TOKEN_SECRET = ''
        auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
        t = twitter.Twitter(auth=auth)

        followers_graph = nx.Graph()
        following_graph = nx.Graph()
        all_users = nx.Graph()

        users = self.users.toPlainText().split("\n")

        for user in users:
            result = t.users.show(screen_name=user)
            id = result["id"]
            followers_graph.add_node(id)
            following_graph.add_node(id)
            all_users.add_node(id)
            cursor = -1
            while cursor != 0:
                response = t.followers.ids(screen_name=user, cursor=cursor)
                for f_id in response['ids']:
                    followers_graph.add_edge(id, f_id)
                    all_users.add_edge(id, f_id)
                cursor = response['next_cursor']
            cursor = -1
            while cursor != 0:
                response = t.friends.ids(screen_name=user, cursor=cursor)
                for f_id in response['ids']:
                    following_graph.add_edge(id, f_id)
                    all_users.add_edge(id, f_id)
                cursor = response['next_cursor']

        all_users = network.readwrite._wrap(all_users)
        followers = network.readwrite._wrap(followers_graph)
        following = network.readwrite._wrap(following_graph)
        self.send("Followers", followers)
        self.send("Following", following)
        self.send("All", all_users)

        self.n_all = len(all_users)
        self.n_followers = len(followers)
        self.n_following = len(following)

if __name__=="__main__":
    from PyQt4.QtGui import QApplication
    a = QApplication([])
    ow = OWTwitterNetwork()
    ow.show()
    a.exec_()
