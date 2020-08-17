import sys
from PyQt5.QtWidgets import (QApplication, QGridLayout, QGroupBox, QLineEdit, QLabel, QFrame,
                             QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QFileDialog, QScrollArea, QFormLayout,
                             QMainWindow, QMessageBox, QAction, QInputDialog, QDialog, QSpacerItem, QSizePolicy)
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtGui import *
from emotion_recognition import detect_emotion
from chatbot import create_agent_and_persona, next_answer, analyse_store_answer
import subprocess
import random
import webbrowser

# Creates QLabel for texts
class Bubble(QLabel):
    def __init__(self,text,user=True):
        super(Bubble,self).__init__(text)
        self.setContentsMargins(5,5,5,5)
        self.user = user
        # Sets color of the text
        if (user):
            self.setStyleSheet("color: white;")
        else:
            self.setStyleSheet("color: black;")

    def paintEvent(self, e):
        p = QPainter(self)
        path = QPainterPath()
        p.setRenderHint(QPainter.Antialiasing,True)
        path.addRoundedRect(0,0,self.width()-1,self.height()-1,5,5);
        # Sets color for the text bubble
        if (self.user):
            p.setPen(QColor(0, 106, 255));
            p.fillPath(path, QColor(0, 106, 255));
        else:
            p.setPen(QColor(211,211,211));
            p.fillPath(path, QColor(211,211,211));
        p.drawPath(path);
        super(Bubble,self).paintEvent(e)

# Creates Widget to hold Bubble Qlabel
class BubbleWidget(QWidget):
    def __init__(self,text,left=True, user=True):
        super(BubbleWidget,self).__init__()
        hbox = QHBoxLayout()
        label = Bubble(text, user)

        # Creates text bubble on right side
        if not left:
            hbox.addSpacerItem(QSpacerItem(1,1,QSizePolicy.Expanding,QSizePolicy.Preferred))
        hbox.addWidget(label)

        # Creates text bubble on left side
        if left:
            hbox.addSpacerItem(QSpacerItem(1,1,QSizePolicy.Expanding,QSizePolicy.Preferred))

        hbox.setContentsMargins(0,0,0,0)

        self.setLayout(hbox)
        self.setContentsMargins(0,0,0,0)

def show_emotion_and_music(text, label):
    emotion = determine_overall_emotion()
    label.setText(emotion)

    # Create the message box
    alert = QMessageBox()
    # Add text, warning icon and title
    alert.setText("Your emotion is {}. Would you like some {} music?".format(emotion, emotion))
    alert.setWindowTitle("Music Suggestion")
    alert.setIcon(QMessageBox.Information)
    # Add the buttons to the message box
    alert.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    retval = alert.exec()
    # If the user push ok, we reset
    if retval == 1024:
        # determine type of music
        if emotion == "joy" :
            string = random_line('music/joy_music.txt').split(";")
        elif emotion == "fear" :
            string = random_line('music/fear_music.txt').split(";")
        elif emotion == "sadness" :
            string = random_line('music/sadness_music.txt').split(";")
        else :
            string = random_line('music/anger_music.txt').split(";")

        music_link = string[0]
        music_name = string[1]
        label.setText(emotion + "\nSong Recommendation: " + music_name)
        webbrowser.open_new(music_link)


def random_line(fname):
    lines = open(fname).read().splitlines()
    return random.choice(lines)

def determine_overall_emotion():
    input = ""
    history = open("data/history.txt")
    lines = history.readlines()
    ctr = 0
    for line in reversed(lines):
        if ctr == 6:    # only look at the last 3 text exchanges
            break
        elif (ctr % 2 == 0):
            input += " " + line
    emotion = detect_emotion(input)
    return emotion


# Function that return a QLabel with the user message color AND tracks user emotion
# def user_input(text):
#     # Define the message
#     user_input = QLabel(text)
#     # Set the color for the message
#     user_input.setStyleSheet("QLabel { background-color : lightblue}")
#     # Return the QLabel
#     return user_input
#
#
# # Function that return a QLabel with the chatbot message color
# def chatbot_input(text):
#     # Define the message
#     chatbot_input = QLabel(text)
#     # Set the color for the message
#     chatbot_input.setStyleSheet("QLabel { background-color : #C0C0C0}")
#     # Return the QLabel
#     return chatbot_input


def message_history():
    widget = QGroupBox("Message")
    # Add the scrollbar
    scroll = QScrollArea()
    scroll.setWidget(widget)
    scroll.setMinimumHeight(600)
    scroll.setWidgetResizable(True)
    # Box where we add the message present in the history file
    message_history_box = QVBoxLayout()
    # doc = QTextDocument()
    # To know if the chatbot said the sentence or the user
    count = 0
    try:
        history = open("data/history.txt", 'r')
    except FileNotFoundError:
        history = open("data/history.txt", "w+")
    for line in history:
        # Case where it's a user message
        if count % 2 == 0:
            # The emotion is the last word of the line
            # doc.setHtml(user_input(line).text())
            # user_text = wrap_text(doc.toPlainText())
            user_text = wrap_text(line)
            message_history_box.addWidget(BubbleWidget(user_text,left=False))
        # Chatbot message
        else:
            # doc.setHtml(chatbot_input(line).text())
            bot_text = wrap_text(line)
            message_history_box.addWidget(BubbleWidget(bot_text,left=True,user=False))
        count = count + 1
    history.close()

    # Add the messages to the box
    widget.setLayout(message_history_box)
    # Return the scrollbar and the verticalbox in order to update it
    return scroll, message_history_box


# Open the window in order to select the files
def getfile(self, box):
    # Open a popup so the user can select a file
    (image, _) = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Image/Videos (*.png *.jpg *.gif *.mp4 *.wav)")
    # Create the place for an image and add the selected file
    pixmap = QPixmap(image)
    pixmap = pixmap.scaledToWidth(600)
    image_input = QLabel(self)
    image_input.setPixmap(pixmap)
    box.addWidget(image_input)

def wrap_text(string, n=14):
    '''returns a string where \\n is inserted between every n words'''
    words = string.split()
    final = ''
    for i in range(0, len(words), n):
        final += ' '.join(words[i:i+n]) + '\n'
    final = final.rstrip()
    return final

# When the user send a message
def add_new_message(message, box, blender_bot):
    # Add the message to the box only if there's a message
    if len(message.text()) > 0:
        # doc = QTextDocument()
        # doc.setHtml(message(user))
        user_text = wrap_text(message.text())
        # Add the user input to the ui
        box.addWidget(BubbleWidget(user_text,left=False))
        # Compute the bot input
        # doc.setHtml(bot_input.text())
        bot_text = wrap_text(next_answer(blender_bot, message.text()))
        # Add the bot input to the ui
        box.addWidget(BubbleWidget(bot_text, left=True, user=False))
        # Add the new elements to the history file.
        # TODO: Improve this function so we're not opening the file every time
        history = open("data/history.txt", "a")
        user_text = user_text.replace('\n', ' ')
        bot_text = bot_text.replace('\n', ' ')
        history.write(message.text() + "\n" + bot_text + "\n")
        # Store the answer if it's a relevant information about the user
        analyse_store_answer(message.text(), '')
        # Empty the message area
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
    emotion_button.clicked.connect(lambda: show_emotion_and_music(new_message_input.text(), emotion_display))
    new_messages_box.addWidget(emotion_button)

    # Add a button in order to input photos and videos
    import_file = QPushButton()
    import_file.setIcon(QIcon("Images/photo.png"))
    # Get the file and add it to the message history
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


class UserInterface(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        # Initialise the blender bot
        self.blender_bot = create_agent_and_persona()
        self.title = "Healthcare Chatbot"
        self.setWindowTitle("Healthcare Chatbot")
        # Set the size of the window
        self.resize(720, 720)
        # Add the scrollbar and the widgets
        self.add_scrollbar_widgets()
        # Add the menu
        self.set_menu()

    # Change the persona
    def change_persona(self):
        # TODO: Set the persona and store it
        # popup = QDialog()
        # vertical_box = QVBoxLayout()
        # popup.setMinimumSize(500, 500)
        # popup.setWindowTitle("Change persona")
        # persona1 = QLineEdit('Sentence for the persona')
        # persona2 = QLineEdit('Sentence for the persona')
        # persona3 = QLineEdit('Sentence for the persona')
        # persona4 = QLineEdit('Sentence for the persona')
        # persona5 = QLineEdit('Sentence for the persona')
        # button_ok = QPushButton("ok")
        # vertical_box.addWidget(persona1)
        # vertical_box.addWidget(persona2)
        # vertical_box.addWidget(persona3)
        # vertical_box.addWidget(persona4)
        # vertical_box.addWidget(persona5)
        # vertical_box.addWidget(button_ok)
        # popup.setLayout(vertical_box)
        # popup.exec()

        alert = QMessageBox()
        # Add text, warning icon and title
        alert.setText("Are you sure you want to change the chatbot's persona?\n"
                      "This action may take some time")
        alert.setWindowTitle("Warning")
        alert.setIcon(QMessageBox.Warning)
        # Add the buttons to the message box
        alert.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        retval = alert.exec()
        if retval == 1024:
            self.blender_bot.reset()
            print("Persona reset")


        # Reset the chatbot and empty the history
    def reset_chatbot(self):
        # Create the message box
        alert = QMessageBox()
        # Add text, warning icon and title
        alert.setText("Are you sure you want to reset the chatbot?\n"
                      "All data will be lost\n "
                      "This action may take some time")
        alert.setWindowTitle("Warning")
        alert.setIcon(QMessageBox.Warning)
        # Add the buttons to the message box
        alert.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        retval = alert.exec()
        # If the user push ok, we reset
        if retval == 1024:
            f = open("data/history.txt", 'w').close()
            f = open("data/user_facts.txt", 'w').close()
            self.blender_bot.reset()
            self.close()
            subprocess.call("python" + " User_interface.py", shell=True)

    # Add the menu with the change persona and reset chatbot buttons
    def set_menu(self):
        # Create the change persona option
        persona = QAction("Change Persona", self)
        persona.triggered.connect(lambda: self.change_persona())
        # Create the reset chatbot option
        reset = QAction("Reset Chatbot", self)
        reset.triggered.connect(lambda: self.reset_chatbot())
        # Create the menu and add the persona
        menu = self.menuBar()
        menu.setNativeMenuBar(False)
        menu.addAction(persona)
        menu.addAction(reset)

    def add_scrollbar_widgets(self):
        # Initialise grid and add the QGridLayout to the QWidget that is added to the QScrollArea
        grid = QGridLayout(self)
        # Add the message history
        scroll_area, message_history_box = message_history()
        grid.addWidget(scroll_area)

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
