import sys
from PyQt5.QtWidgets import (QApplication, QGridLayout, QGroupBox, QLineEdit, QLabel,  QFrame,
                             QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QFileDialog, QScrollArea, QFormLayout)
from PyQt5 import QtGui
from PyQt5.QtGui import QIcon
from emotion_recognition import detect_emotion


# Function that return a QLabel with the user message color AND tracks user emotion
def user_input(text, emotion):
    # Define the message
    user_input = QLabel("User: " + text + "\n\tSentiment: " + emotion)
    # Set the color for the message
    user_input.setStyleSheet("QLabel { background-color : lightblue}")
    # Return the QLabel
    return user_input


# Function that return a QLabel with the chatbot message color
def chatbot_input(text):
    # Define the message
    chatbot_input = QLabel("Chatbot: " + text)
    # Set the color for the message
    chatbot_input.setStyleSheet("QLabel { background-color : #C0C0C0}")
    # Return the QLabel
    return chatbot_input


def message_history():
    group_box = QGroupBox("Messages")
    vertical_box = QVBoxLayout()
    # Add some predefined messages
    for i in range(3):
        vertical_box.addWidget(user_input("Hello there, what's your name?", "joy"))
        vertical_box.addWidget(chatbot_input("i ' m felix , and i ' ve a brother . do you have any siblings ?"))
        vertical_box.addWidget(user_input("Yes I have 1 sister and 1 brother. How old is your brother?", "joy"))
        vertical_box.addWidget(chatbot_input("he ' s 10 years old . what do you like to do for fun ? i study psychology ."))

    # Add the messages to the groupbox
    group_box.setLayout(vertical_box)
    # Return the groupbox and the verticalbox in order to update it
    return group_box, vertical_box


# Open the window in order to select the files
def getfile(self):
    image = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Image/Videos (*.png *.jpg *.gif *.mp4 *.wav)")


# When the user send a message
def add_new_message(message, box):
    # Add the message to the box only if there's a message
    if len(message.text()) > 0:
        emotion = detect_emotion(message.text())
        box.addWidget(user_input(message.text(), emotion))
        message.setText("")


def messages(messages_box):
    group_box = QGroupBox("New message")
    horizontal_box = QHBoxLayout()
    # Add the input line to the horizontal box
    new_message_input = QLineEdit()
    # If we press the ENTER key, we send the message
    new_message_input.returnPressed.connect(lambda: add_new_message(new_message_input, messages_box))

    # Create the send button
    # TODO: If there's no text, display the photo button, otherwise the send button (not both)
    send_button = QPushButton()
    # Change the icon
    send_button.setIcon(QIcon("Images/send.jpg"))
    # Send the message if the user press the send button
    send_button.clicked[bool].connect(lambda: add_new_message(new_message_input, messages_box))
    # Add the input line and the button
    horizontal_box.addWidget(new_message_input)
    horizontal_box.addWidget(send_button)

    # Add a button in order to input photos and videos
    import_file = QPushButton()
    import_file.setIcon(QIcon("Images/photo.png"))
    # import_file.clicked.connect(getfile(import_file))
    horizontal_box.addWidget(import_file)

    group_box.setLayout(horizontal_box)
    return group_box


# Add a separation between the new message and the history
def new_message_on_bottom():
    # Initialise the frame
    frame = QFrame()
    vertical_box = QVBoxLayout()
    # Fill the box with blank so the new message is on the bottom
    vertical_box.addStretch(1)
    # Add the box to the frame
    frame.setLayout(vertical_box)
    return frame


class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        self.add_scrollbar()

        grid = QGridLayout(self.scrollAreaWidgetContents)

        # Add the message history
        messages_history, vertical_box = message_history()
        grid.addWidget(messages_history)

        # Separation between the new messages input and the history
        grid.addWidget(new_message_on_bottom())

        # Add the input line for new messages
        grid.addWidget(messages(vertical_box))
        self.setLayout(grid)

        self.setWindowTitle("Healthcare Chatbot")
        self.resize(720, 720)


    def add_scrollbar(self):
        # Adding scrollbar
        self.layout = QHBoxLayout(self)
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        # Add the QGridLayout to the QWidget that is added to the QScrollArea
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.layout.addWidget(self.scrollArea)


# TODO: add persona
# TODO: add video/images
if __name__ == '__main__':
    app = QApplication(sys.argv)
    clock = Window()
    clock.show()
    sys.exit(app.exec_())
