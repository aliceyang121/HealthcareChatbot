import sys
from PyQt5.QtWidgets import (QApplication, QGridLayout, QGroupBox, QLineEdit, QLabel, QFrame,
                             QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QFileDialog, QScrollArea,
                             QMainWindow, QMessageBox, QAction, QSpacerItem, QSizePolicy, QDialog)
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QPainterPath, QColor
from sentence_transformers import SentenceTransformer

from emotion_recognition import detect_emotion
from chatbot import create_agent_and_persona, next_answer, analyse_store_answer, greetings
import subprocess
from random import choice
import webbrowser
import csv
from speech_recognition import Recognizer, Microphone, UnknownValueError
# from video_recognition import video_emotion_recognition


# Creates QLabel for texts
class Bubble(QLabel):
    def __init__(self, text, user=True):
        super(Bubble, self).__init__(text)
        self.setContentsMargins(5, 5, 5, 5)
        self.user = user
        # Sets color of the text
        if user:
            self.setStyleSheet("color: white;")
        else:
            self.setStyleSheet("color: black;")

    def paintEvent(self, e):
        p = QPainter(self)
        path = QPainterPath()
        p.setRenderHint(QPainter.Antialiasing, True)
        path.addRoundedRect(0, 0, self.width() - 1, self.height() - 1, 5, 5);
        # Sets color for the text bubble
        if self.user:
            p.setPen(QColor(0, 106, 255));
            p.fillPath(path, QColor(0, 106, 255));
        else:
            p.setPen(QColor(211, 211, 211));
            p.fillPath(path, QColor(211, 211, 211));
        p.drawPath(path);
        super(Bubble, self).paintEvent(e)


# Creates Widget to hold Bubble Qlabel
class BubbleWidget(QWidget):
    def __init__(self, text, left=True, user=True):
        super(BubbleWidget, self).__init__()
        hbox = QHBoxLayout()
        label = Bubble(text, user)

        # Creates text bubble on right side
        if not left:
            hbox.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Preferred))
        hbox.addWidget(label)

        # Creates text bubble on left side
        if left:
            hbox.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Preferred))

        hbox.setContentsMargins(0, 0, 0, 0)

        self.setLayout(hbox)
        self.setContentsMargins(0, 0, 0, 0)


def show_emotion_and_music(text, label):
    emotion, probability = determine_overall_emotion()
    probability = str(probability)
    label.setText("Emotion: " + emotion + "\nProbability: " + probability)

    # Create the message box
    alert = QMessageBox()
    # Add text, warning icon and title
    alert.setText("Your emotion is {}. Would you like some music?".format(emotion))
    alert.setWindowTitle("Music Suggestion")
    alert.setIcon(QMessageBox.Information)
    # Add the buttons to the message box
    alert.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    retval = alert.exec()
    # If the user push ok, we reset
    if retval == 1024:
        # determine type of music
        if emotion in ["joy", "surprise", "neutral"]:
            string = random_line('music/joy_music.txt').split(";")
        elif emotion == "fear":
            string = random_line('music/fear_music.txt').split(";")
        elif emotion == "sadness":
            string = random_line('music/sadness_music.txt').split(";")
        elif emotion in ["anger", "disgust"]:
            string = random_line('music/anger_music.txt').split(";")
        elif emotion == "NA":
            pass

        music_link = string[0]
        music_name = string[1]
        label.setText(emotion + "\nSong Recommendation: " + music_name)
        webbrowser.open_new(music_link)


def determine_overall_emotion():
    history = open("data/history.csv", 'r')
    history_reader = csv.reader(history, delimiter=';')
    emotion = "NA"
    probability = -1
    for line in reversed(list(history_reader)):
        if line[0] == 'U':    # only look at the last text exchange by user
            emotion, probability = detect_emotion(line[1])
            # emotions.append(emotion)
            # probabilities.append(probability)
            break

    if probability > 90:
        return emotion, probability

    else:
        # return video_emotion_recognition()
        return "NA", probability


def random_line(fname):
    file = open(fname)
    result = choice(file.read().splitlines())
    file.close()
    return result


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
    # returns a string where \\n is inserted between every n words
    words = string.split()
    final = ''
    for i in range(0, len(words), n):
        final += ' '.join(words[i:i + n]) + '\n'
    final = final.rstrip()
    return final


# When the user send a message
def add_new_message(message, box, blender_bot):
    # Add the message to the box only if there's a message
    if len(message.text()) > 0:
        user_text = wrap_text(message.text())
        # Add the user input to the ui
        box.addWidget(BubbleWidget(user_text, left=False))
        # Compute the bot input
        bot_text = wrap_text(next_answer(blender_bot, message.text()))
        blender_bot.last_message = bot_text
        # Add the bot input to the ui
        box.addWidget(BubbleWidget(bot_text, left=True, user=False))
        # Add the new elements to the history file.
        bot_text = bot_text.replace('\n', ' ')
        history = open("data/history.csv", 'a')
        writer = csv.writer(history, delimiter=';')
        writer.writerow(['U', message.text()])
        writer.writerow(['C', bot_text])
        history.close()
        message.setText("")


# Extract audio from the microphone and convert it to text
def audio_to_text(message_input):
    # initialise the recognizer
    r = Recognizer()
    # Use the sysdefault microphone
    for i, microphone_name in enumerate(Microphone.list_microphone_names()):
        if microphone_name == "sysdefault":
            micro = Microphone(device_index=i)
    with micro as source:
        # Extract the audio and convert it to text
        audio = r.listen(source)
    # recognize speech using Google Speech Recognition and add it to the text input area
    try:
        message_input.setText(r.recognize_google(audio))
    except UnknownValueError:
        message_input.setText('The audio was not understood')


# Add the message input and the buttons
def messages(message_history_box, blender_bot):
    group_box = QGroupBox("New message")
    new_messages_box = QHBoxLayout()
    # Add the input line to the horizontal box
    new_message_input = QLineEdit()
    # If we press the ENTER key, we send the message
    new_message_input.returnPressed.connect(
        lambda: add_new_message(new_message_input, message_history_box, blender_bot))
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

    audio_button = QPushButton()
    audio_button.setIcon(QIcon('Images/audio.png'))
    audio_button.clicked.connect(lambda: audio_to_text(new_message_input))
    new_messages_box.addWidget(audio_button)

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


def persona():
    try:
        file = open("data/persona.txt")
        result = file.read().splitlines()
        file.close()
    except FileNotFoundError:
        result = []
    return result


def persona_qline_edit(lines, number_persona, index):
    if number_persona >= index:
        return QLineEdit(lines[index - 1])
    else:
        return QLineEdit()


def add_your_persona(personas):
    result = []
    for line in personas:
        if line != '':
            result.append('Your persona! ' + line)
    return result


def set_personas(popup, persona1, persona2, persona3, persona4, persona5):
    file = open("data/persona.txt", 'w')
    file.write(persona1 + "\n" + persona2 + "\n" + persona3 + "\n" + persona4 + "\n" + persona5 + "\n")
    file.close()
    popup.close()


class UserInterface(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        # Initialise the blender bot
        personas = persona()
        self.blender_bot = create_agent_and_persona(add_your_persona(personas))
        self.blender_bot.last_message = greetings()
        self.blender_bot.embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
        self.blender_bot.memory = self.add_memory()
        self.blender_bot.persona = personas
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
        popup = QDialog()
        vertical_box = QVBoxLayout()
        popup.setMinimumSize(300, 300)
        popup.setWindowTitle("Change persona")
        number_persona = len(self.blender_bot.persona)
        persona1 = persona_qline_edit(self.blender_bot.persona, number_persona, 1)
        persona2 = persona_qline_edit(self.blender_bot.persona, number_persona, 2)
        persona3 = persona_qline_edit(self.blender_bot.persona, number_persona, 3)
        persona4 = persona_qline_edit(self.blender_bot.persona, number_persona, 4)
        persona5 = persona_qline_edit(self.blender_bot.persona, number_persona, 5)
        button_set = QPushButton("Set persona", popup)
        button_set.clicked.connect(lambda: set_personas(popup, persona1.text(), persona2.text(), persona3.text(),
                                                        persona4.text(), persona5.text()))
        vertical_box.addWidget(QLabel("Set the persona of the chatbot.\n "
                                      "If you change it, the new persona will be create after restarting the "
                                      "application. "
                                      "\n\nExample of persona: \n My name is John \n"))
        vertical_box.addWidget(persona1)
        vertical_box.addWidget(persona2)
        vertical_box.addWidget(persona3)
        vertical_box.addWidget(persona4)
        vertical_box.addWidget(persona5)
        vertical_box.addWidget(button_set)
        popup.setLayout(vertical_box)
        popup.exec()

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
            open("data/history.csv", 'w').close()
            open("data/user_facts.csv", 'w').close()
            self.blender_bot.reset()
            self.close()
            subprocess.call("python" + " User_interface.py", shell=True)

    def save_info(self):
        # Obtain last user input and chatbot question
        history = open("data/history.csv", 'r')
        history_reader = csv.reader(history, delimiter=';')
        question = False
        for line in reversed(list(history_reader)):
            if line[0] == 'U':
                user_info = line[1]
                question = True
            elif question:
                question_info = line[1]
                question = False
                break

        alert = QDialog()
        alert.setMinimumSize(500, 200)
        # Add text, icon and title
        vertical_box = QVBoxLayout()
        question_field = QLineEdit(question_info)
        answer_field = QLineEdit(user_info)

        vertical_box.addWidget(QLabel('Question: '))
        vertical_box.addWidget(question_field)
        vertical_box.addWidget(QLabel('Question: '))
        vertical_box.addWidget(answer_field)
        alert.setWindowTitle("Save Personal Info")
        # alert.setIcon(QMessageBox.Information)
        # Add the buttons to the message box
        # alert.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        button_set = QPushButton("Set persona", alert)
        vertical_box.addWidget(button_set)
        button_set.clicked.connect(lambda: analyse_store_answer(answer_field.text(), question_field.text(), alert))
        alert.setLayout(vertical_box)
        alert.exec()


    # Add the menu with the change persona and reset chatbot buttons
    def set_menu(self):
        # Create the change persona option
        persona = QAction("Change Persona", self)
        persona.triggered.connect(lambda: self.change_persona())
        # Create the reset chatbot option
        reset = QAction("Reset Chatbot", self)
        reset.triggered.connect(lambda: self.reset_chatbot())
        # Create save personal information option
        save = QAction("Save Information", self)
        save.triggered.connect(lambda: self.save_info())
        # Create the menu and add the persona
        menu = self.menuBar()
        menu.setNativeMenuBar(False)
        menu.addAction(persona)
        menu.addAction(reset)
        menu.addAction(save)

    def add_scrollbar_widgets(self):
        # Initialise grid and add the QGridLayout to the QWidget that is added to the QScrollArea
        grid = QGridLayout(self)
        # Add the message history
        scroll_area, message_history_box = self.message_history()
        grid.addWidget(scroll_area)

        new_messages_box, emotion_box = messages(message_history_box, self.blender_bot)

        # Add the input line for new messages
        grid.addWidget(new_messages_box)
        # Add the sentiment display
        grid.addWidget(emotion_box)
        self.central_widget.setLayout(grid)

    # Add all the question and answers to the robot memory
    def add_memory(self):
        question = []
        answer = []
        # Open the file and fill the question and answer
        try:
            user_facts = open("data/user_facts.csv", 'r')
            reader = csv.reader(user_facts, delimiter=';')
            for row in reader:
                question.append(row[0])
                answer.append(row[1])
        # Case where the file doesn't exists, we create it
        except FileNotFoundError:
            user_facts = open("data/user_facts.csv", 'w')
        # Close the file
        user_facts.close()
        # Convert the facts to tensors
        if len(question) == 0:
            questions_embedding = None
        else:
            questions_embedding = self.blender_bot.embedder.encode(question, convert_to_tensor=True)
        return questions_embedding, answer

    def message_history(self):
        widget = QGroupBox("Message")
        # Add the scrollbar
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setMinimumHeight(600)
        scroll.setWidgetResizable(True)
        # Box where we add the message present in the history file
        message_history_box = QVBoxLayout()
        # To know if the chatbot said the sentence or the user
        history = open("data/history.csv", 'r')
        history_reader = csv.reader(history, delimiter=';')
        for row in history_reader:
            # Case where it's a user message
            if (row[0] == 'U'):
                # The emotion is the last word of the line
                user_text = wrap_text(row[1])
                message_history_box.addWidget(BubbleWidget(user_text, left=False))
            # Chatbot message
            else:
                # doc.setHtml(chatbot_input(line).text())
                bot_text = wrap_text(row[1])
                message_history_box.addWidget(BubbleWidget(bot_text, left=True, user=False))
        history.close()
        # Add the greetings
        # print(self.blender_bot.last_message)
        self.blender_bot.observe({'text': '', "episode_done": False})
        self.blender_bot.self_observe({'text': self.blender_bot.last_message, "episode_done": False})
        message_history_box.addWidget(BubbleWidget(self.blender_bot.last_message, left=True, user=False))
        history = open("data/history.csv", 'a')
        writer = csv.writer(history, delimiter=';')
        writer.writerow(['C', self.blender_bot.last_message])
        history.close()
        # Add the messages to the box
        widget.setLayout(message_history_box)
        # Return the scrollbar and the verticalbox in order to update it
        return scroll, message_history_box


# TODO: add persona
if __name__ == '__main__':
    app = QApplication(sys.argv)
    user_interface = UserInterface()
    user_interface.show()
    sys.exit(app.exec_())
