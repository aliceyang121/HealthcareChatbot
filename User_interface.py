import sys
from PyQt5.QtWidgets import (QApplication, QGridLayout, QGroupBox, QLineEdit, QLabel, QFrame,
                             QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QFileDialog, QScrollArea, QFormLayout,
                             QMainWindow, QMessageBox, QAction)
from PyQt5.QtGui import QIcon, QPixmap
from emotion_recognition import detect_emotion
from chatbot import create_agent_and_persona, next_answer, analyse_store_answer

def show_emotion(text, label):
    emotion = detect_emotion(text)
    label.setText(emotion)

# Function that return a QLabel with the user message color AND tracks user emotion
def user_input(text):
    # Define the message
    user_input = QLabel("User: " + text)
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
    message_history_box = QVBoxLayout()

    # To know if the chatbot said the sentence or the user
    count = 0
    history = open("data/history.txt")
    for line in history:
        if count % 2 == 0:
            list_word = line.split(" ")
            # The emotion is the last word of the line
            message_history_box.addWidget(user_input(" ".join(list_word[:])))
        else:
            message_history_box.addWidget(chatbot_input(line))
        count = count + 1
    history.close()

    # Add the messages to the box
    widget.setLayout(message_history_box)
    # Return the scrollbar and the verticalbox in order to update it
    return scroll, message_history_box


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
        # emotion = detect_emotion(message.text())
        box.addWidget(user_input(message.text()))
        bot_input = chatbot_input(next_answer(blender_bot, message.text()))
        box.addWidget(bot_input)
        # Add the new elements to the history file.
        # TODO: Improve this function so we're not opening the file every time
        history = open("data/history.txt", "a")
        history.write(message.text() + "\n" + bot_input.text() + "\n")
        analyse_store_answer(message.text(), '')
        message.setText("")

def messages(message_history_box, blender_bot):
    group_box = QGroupBox("New message")
    new_messages_box = QHBoxLayout()
    # Add the input line to the horizontal box
    new_message_input = QLineEdit()
    # If we press the ENTER key, we send the message
    new_message_input.returnPressed.connect(lambda: add_new_message(new_message_input, message_history_box, blender_bot))

    # Create the send button
    # TODO: If there's no text, display the photo button, otherwise the send button (not both)
    send_button = QPushButton()
    # Change the icon
    send_button.setIcon(QIcon("Images/send.jpg"))
    # Send the message if the user press the send button
    send_button.clicked[bool].connect(lambda: add_new_message(new_message_input, message_history_box, blender_bot))
    # Add the input line and the button
    new_messages_box.addWidget(new_message_input)
    new_messages_box.addWidget(send_button)

    # Create show emotion button
    sentiment_group_box = QGroupBox("Sentiment")
    sentiment_box = QHBoxLayout()
    emotion_button = QPushButton()
    emotion_button.setIcon(QIcon("Images/emoji.png"))
    emotion_display = QLabel()
    sentiment_box.addWidget(emotion_display)
    emotion_button.clicked.connect(lambda: show_emotion(new_message_input.text(), emotion_display))
    new_messages_box.addWidget(emotion_button)

    # Add a button in order to input photos and videos
    import_file = QPushButton()
    import_file.setIcon(QIcon("Images/photo.png"))
    import_file.clicked.connect(lambda: getfile(import_file, message_history_box))
    new_messages_box.addWidget(import_file)


    group_box.setLayout(new_messages_box)
    sentiment_group_box.setLayout(sentiment_box)
    return group_box, sentiment_group_box


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
        scroll_area, message_history_box = message_history()
        grid.addWidget(scroll_area)

        # Separation between the new messages input and the history
        # grid.addWidget(new_message_on_bottom())

        new_messages_box, emotion_box = messages(message_history_box, self.blender_bot)

        # Add the input line for new messages
        grid.addWidget(new_messages_box)
        # Add the sentiment display
        grid.addWidget(emotion_box)

        self.central_widget.setLayout(grid)


# TODO: add persona
if __name__ == '__main__':
    app = QApplication(sys.argv)
    user_interface = UserInterface()
    user_interface.show()
    sys.exit(app.exec_())
