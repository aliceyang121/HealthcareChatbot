import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox, QInputDialog,
        QLabel, QPlainTextEdit, QRadioButton, QVBoxLayout, QWidget)

class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        grid = QGridLayout()
        grid.addWidget(self.createExampleGroup(), 0, 0)
        self.setLayout(grid)

        self.setWindowTitle("PyQt5 Group Box")
        self.resize(400, 300)

    def createExampleGroup(self):
        groupBox = QGroupBox("Best Food")
        txt = QPlainTextEdit("Coucou toi")
        text = QInputDialog()
        t = QLabel('Hello there')
        vbox = QVBoxLayout()
        vbox.addWidget(t)
        vbox.addWidget(txt)
        vbox.addWidget(text)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox

if __name__ == '__main__':
    app = QApplication(sys.argv)
    clock = Window()
    clock.show()
    sys.exit(app.exec_())