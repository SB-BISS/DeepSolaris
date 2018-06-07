import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
import os

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Image Labeling'

        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(300,300,300,400)
        self.iter = 0
        self.pics = [file for file in os.listdir('roofs/')]

        # Create widget
        self.img = QLabel(self)
        self.img.setPixmap(QtGui.QPixmap("roofs/"+self.pics[self.iter]).scaled(200, 200))
        self.img.resize(200, 200)
        self.img.move(50,50)

        self.ent = QPushButton(self)
        self.ent.move(50, 250)
        self.ent.resize(100, 100)
        self.ent.setText("Positive")
        self.ent.setShortcut("Ctrl+Space")
        self.ent.clicked.connect(self.PosNext)

        self.can = QPushButton(self)
        self.can.move(150, 250)
        self.can.resize(100, 100)
        self.can.setText("Negative")
        self.can.setShortcut("Ctrl+v")
        self.can.clicked.connect(self.NegNext)

        self.show()

    def PosNext(self):
        if not (os.path.isdir("pos/")):
            os.mkdir("pos/")
        try:
            os.rename("roofs/"+self.pics[self.iter], "pos/"+self.pics[self.iter] )
            self.iter +=1
            self.img.setPixmap(QtGui.QPixmap("roofs/" + self.pics[self.iter]).scaled(200, 200))
            self.img.resize(200, 200)
            self.img.move(50, 50)
        except:
            pass

    def NegNext(self):
        if not (os.path.isdir("neg/")):
            os.mkdir("neg/")
        try:
            os.rename("roofs/"+self.pics[self.iter], "neg/"+self.pics[self.iter] )
            self.iter +=1
            self.img.setPixmap(QtGui.QPixmap("roofs/" + self.pics[self.iter]).scaled(200, 200))
            self.img.resize(200, 200)
            self.img.move(50, 50)
        except:
            pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())