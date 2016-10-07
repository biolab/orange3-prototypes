import twitter
import networkx as nx
import orangecontrib.network as network

from Orange.widgets import widget, gui
from PyQt4 import QtGui

class OWTwitterNetwork(widget.OWWidget):
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

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.consumer_key = QtGui.QLineEdit()
        box.layout().addWidget(QtGui.QLabel("Key:"))
        box.layout().addWidget(self.consumer_key)
        self.consumer_secret = QtGui.QLineEdit()
        box.layout().addWidget(QtGui.QLabel("Secret:"))
        box.layout().addWidget(self.consumer_secret)
        box.layout().addWidget(QtGui.QLabel("Users:"))
        self.users = QtGui.QTextEdit()
        box.layout().addWidget(self.users)
        gui.label(box, self, 'Following: %(n_following)d\n'
                             'Followers: %(n_followers)d\n'
                             'All: %(n_all)d')
        self.button = gui.button(box, self, "Create graph", self.fetch_users)

    def fetch_users(self):
        CONSUMER_KEY = self.consumer_key.text()
        CONSUMER_SECRET = self.consumer_secret.text()
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
