import twitter
import networkx as nx
import orangecontrib.network as network

from Orange.widgets import widget, gui
from PyQt4 import QtGui

class OWTwitterNetwork(widget.OWWidget):
    name = "Twitter User Network"
    description = ""
    icon = "icons/Twitter.svg"
    priority = 10
    outputs = [("Followers", network.Graph), ("Following", network.Graph), ("All", network.Graph)]

    want_main_area = False

    def __init__(self):
        super().__init__()

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
        self.infoa = gui.widgetLabel(box, '')
        self.infob = gui.widgetLabel(box, '')
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
        all = nx.Graph()

        users = self.users.toPlainText().split("\n")

        for user in users:
            result = t.users.show(screen_name=user)
            id = result["id"]
            followers_graph.add_node(id)
            followers = []
            following_graph.add_node(id)
            following = []
            all.add_node(id)
            cursor = -1
            while cursor != 0:
                response = t.followers.ids(screen_name=user, cursor=cursor)
                followers.extend(response['ids'])
                cursor = response['next_cursor']
            for f_id in followers:
                followers_graph.add_edge(id, f_id)
                all.add_edge(id, f_id)
            cursor = -1
            while cursor != 0:
                response = t.friends.ids(screen_name=user, cursor=cursor)
                following.extend(response['ids'])
                cursor = response['next_cursor']
            for f_id in following:
                following_graph.add_edge(id, f_id)
                all.add_edge(id, f_id)

        self.infob.setText("Number of nodes: %i" % nx.number_of_nodes(all))
        all = network.readwrite._wrap(all)
        followers = network.readwrite._wrap(followers_graph)
        following = network.readwrite._wrap(following_graph)
        self.send("Followers", followers)
        self.send("Following", following)
        self.send("All", all)

        self.infoa.setText("Number of followers: %i" % len(followers))
        self.infob.setText("Number of following: %i" % len(following))

if __name__=="__main__":
    from PyQt4.QtGui import QApplication
    a = QApplication([])
    ow = OWTwitterNetwork()
    ow.show()
    a.exec_()
