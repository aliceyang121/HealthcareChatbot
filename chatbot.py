from parlai.core.agents import create_agent_from_model_file
from sentence_transformers import SentenceTransformer, util
import csv


def extract_question(input):
    partition = input.partition('?')
    input = partition[0] + partition[1]
    input = input.partition('.')
    while '.' in input[1]:
        input = input[2]
        input = input.partition('.')
    input = input[0]
    input = input.replace('your ', 'my ')
    input = input.replace('are you ', 'am I ')
    input = input.replace('you ', 'I ')
    input = input.replace('my ', 'your ')
    return input


# Add the user input and the question if necessary
def analyse_store_answer(user_input, bot_input):
    bot_input = extract_question(bot_input)
    if '?' in bot_input and user_input[0] == 'I' and len(user_input) > 5:
        if user_input[1:4] == "'m " or user_input[1] == ' ':
            if user_input[1:4] == "'m ":
                user_input = user_input[3:]
                user_input = "You are" + user_input
            else:
                user_input = user_input[1:]
                user_input = "You" + user_input
            file_user_facts = open("data/user_facts.csv", 'a')
            writer = csv.writer(file_user_facts, delimiter=';')
            writer.writerow([bot_input.replace('\n', " "), user_input])


def max_index(list_value):
    index = 0
    maximum = 0
    i = 0
    for element in list_value:
        if element>maximum:
            maximum = element
            index = i
        i = i+1
    return maximum, index


# Answer to the user
def next_answer(blender_agent, user_input, boolean_finish=False):
    blender_agent.observe({'text': user_input, "episode_done": boolean_finish})
    questions, answer = blender_agent.memory
    query_embedding = blender_agent.embedder.encode(user_input, convert_to_tensor=True)
    try:
        facts_embedding = blender_agent.embedder.encode(questions, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, facts_embedding)[0]
        top_result, index = max_index(cos_scores)
    except RuntimeError:
        top_result = 0
    if top_result > 0.9:
        response = blender_agent.act(answer[index], from_db=True)
    else:
        response = blender_agent.act()
    return response['text']


# Ask the user for an input and check if the input is valid
def ask_user_input():
    user_input = input('You:')
    while user_input == '':
        user_input = input('The message was empty, enter your message:')
    return user_input


def create_agent_and_persona(persona=''):
    blender_agent = create_agent_from_model_file("zoo:blender/blender_90M/model")
    blender_agent.observe({'text': persona})
    return blender_agent


if __name__ == '__main__':
    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarit
    embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
    user_input = ['Hello', "Bonjour"]
    query_embedding = embedder.encode(user_input, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, user_input_em)[0]

    # We use np.argpartition, to only partially sort the top_k results
    top_result = max(-cos_scores)

    print("\n\n======================\n\n")
    print("Query:", user_input)
    print(user_input.strip(), "(Score: %.4f)" % cos_scores)