from parlai.core.agents import create_agent_from_model_file


# Add the user input and the question if necessary
def analyse_store_answer(user_input, bot_input):
    if user_input[0] == 'I' and len(user_input) > 5:
        if user_input[1:4] == "'m " or user_input[1] == ' ':
            if user_input[1:4] == "'m ":
                user_input = user_input[3:]
                user_input = "You are" + user_input + "\n"
            else:
                user_input = user_input[1:]
                user_input = "You" + user_input + "\n"

            print("We will store: " + user_input)
            file_user_facts = open("data/user_facts.txt", "a")
            file_user_facts.write(user_input)
            file_user_facts.close()
            # TODO improve the condition to select a message to store


# Print all the messages in the conversation
def print_convs(convs):
    for i in all_convs:
        print(i)


# Answer to the user
def next_answer(blender_agent, user_input, boolean_finish=False):
    all_convs.append(f"You: {user_input}")
    blender_agent.observe({'text': user_input, "episode_done": boolean_finish})
    response = blender_agent.act()
    all_convs.append("BlenderBot: {}".format(response['text']))
    print("BlenderBot: {}".format(response['text']))
    return response


# Ask the user for an input and check if the input is valid
def ask_user_input():
    user_input = input('You:')
    while user_input == '':
        user_input = input('The message was empty, enter your message:')
    return user_input


def create_agent_and_persona(persona):
    blender_agent = create_agent_from_model_file("zoo:blender/blender_90M/model")
    blender_agent.observe({'text': persona})
    return blender_agent


if __name__ == '__main__':
    all_convs = []
    user_input = ask_user_input()
    blender_agent = create_agent_and_persona()
    while user_input != '[EXIT]':
        bot_input = next_answer(blender_agent, user_input)
        user_input = ask_user_input()
        analyse_store_answer(user_input, bot_input)
# TODO: link with the ui
# TODO: Add the setup files (after python setup.py)