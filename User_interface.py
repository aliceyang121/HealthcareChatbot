import sys
from PyQt5.QtWidgets import (QApplication, QGridLayout, QGroupBox, QLineEdit, QLabel, QFrame,
                             QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QFileDialog, QScrollArea, QFormLayout,
                             QMainWindow, QMessageBox, QAction)
from PyQt5.QtGui import QIcon, QPixmap
from emotion_recognition import detect_emotion
from chatbot import create_agent_and_persona, next_answer, analyse_store_answer

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
    widget = QGroupBox("Message")
    # Add the scrollbar
    scroll = QScrollArea()
    scroll.setWidget(widget)
    scroll.setMinimumHeight(600)
    scroll.setWidgetResizable(True)
    vertical_box = QVBoxLayout()

    # To know if the chatbot said the sentence or the user
    count = 0
    history = open("data/history.txt")
    for line in history:
        if count % 2 == 0:
            list_word = line.split(" ")
            # The emotion is the last word of the line
            vertical_box.addWidget(user_input("".join(list_word[:-1]), list_word[-1]))
        else:
            vertical_box.addWidget(chatbot_input(line))
        count = count + 1
    history.close()

    # Add the messages to the box
    widget.setLayout(vertical_box)
    # Return the scrollbar and the verticalbox in order to update it
    return scroll, vertical_box


# Open the window in order to select the files
def getfile(self, box):
    (image, _) = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Image/Videos (*.png *.jpg *.gif *.mp4 *.wav)")
    pixmap = QPixmap(image)
    pixmap = pixmap.scaledToWidth(600)
    image_input = QLabel(self)
    image_input.setPixmap(pixmap)
    box.addWidget(image_input)


# When the user send a message
def add_new_message(message, box, blender_bot):
    # Add the message to the box only if there's a message
    if len(message.text()) > 0:
        emotion = detect_emotion(message.text())
        box.addWidget(user_input(message.text(), emotion))
        bot_input = chatbot_input(next_answer(blender_bot, message.text()))
        box.addWidget(bot_input)
        # Add the new elements to the history file.
        # TODO: Improve this function so we're not opening the file every time
        history = open("data/history.txt", "a")
        history.write(message.text() + " " + emotion + "\n" + bot_input.text() + "\n")
        analyse_store_answer(message.text(), '')
        message.setText("")



def messages(messages_box, blender_bot):
    group_box = QGroupBox("New message")
    horizontal_box = QHBoxLayout()
    # Add the input line to the horizontal box
    new_message_input = QLineEdit()
    # If we press the ENTER key, we send the message
    new_message_input.returnPressed.connect(lambda: add_new_message(new_message_input, messages_box, blender_bot))

    # Create the send button
    # TODO: If there's no text, display the photo button, otherwise the send button (not both)
    send_button = QPushButton()
    # Change the icon
    send_button.setIcon(QIcon("Images/send.jpg"))
    # Send the message if the user press the send button
    send_button.clicked[bool].connect(lambda: add_new_message(new_message_input, messages_box, blender_bot))
    # Add the input line and the button
    horizontal_box.addWidget(new_message_input)
    horizontal_box.addWidget(send_button)

    # Add a button in order to input photos and videos
    import_file = QPushButton()
    import_file.setIcon(QIcon("Images/photo.png"))
    import_file.clicked.connect(lambda: getfile(import_file, messages_box))
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


def change_persona():
    print('Hello there')


class UserInterface(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.blender_bot = create_agent_and_persona('Your persona: My name is Jean-Eudes')
        self.layout = QHBoxLayout(self)
        self.title = "Healthcare Chatbot"
        self.setWindowTitle("Healthcare Chatbot")
        self.resize(720, 720)
        self.add_scrollbar_widgets()
        self.set_menu()

    def reset_chatbot(self):
        alert = QMessageBox.about(self, "Warning", "Are you sure you want to reset the "
                                                   "chatbot?\n All data will be loss")

        #alert.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        # alert.buttonClicked.connect(msgButtonClick)
        # returnValue = alert.exec()
        # if returnValue == QMessageBox.Ok:
        #     print('OK clicked')
        # TODO: Clear history
        # TODO: Print the Qmessage with the ok and cancel buttons
        self.blender_bot.reset()



    def set_menu(self):
        persona = QAction("Change persona", self)
        persona.triggered.connect(lambda: change_persona())

        reset = QAction("Reset Chatbot", self)

        reset.triggered.connect(lambda: self.reset_chatbot())
        menu = self.menuBar()
        menu.addAction(persona)
        menu.addAction(reset)

    def add_scrollbar_widgets(self):
        # Initialise grid and add the QGridLayout to the QWidget that is added to the QScrollArea
        grid = QGridLayout(self)
        # Add the message history
        messages_history, vertical_box = message_history()
        grid.addWidget(messages_history)
        # Separation between the new messages input and the history
        grid.addWidget(new_message_on_bottom())

        # Add the input line for new messages
        grid.addWidget(messages(vertical_box, self.blender_bot))
        self.central_widget.setLayout(grid)


# TODO: add persona
if __name__ == '__main__':
    app = QApplication(sys.argv)
    user_interface = UserInterface()
    user_interface.show()
    sys.exit(app.exec_())
