try:
    from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
except ImportError:
    from transformers.modeling_bert import BertModel
import torch
import tensorflow as tf
from torch import nn
import pandas as pd


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)

def prediction_probabilities(model, output, class_names):
    model = model.eval()
    prediction_probs = []
    with torch.no_grad():
        probs = nn.functional.softmax(output, dim=1)
        prediction_probs.extend(probs)
        prediction_probs = torch.stack(prediction_probs).cpu()
    pred_df = pd.DataFrame({'class_names': class_names, 'values': prediction_probs[0]})
    max_probability = max(pred_df['values'])
    return max_probability

def detect_emotion(review_text):
    # set device to cpu
    device = torch.device("cpu")

    # load pretrained model
    PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'

    # load pretrained bert tokenizer
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    # create classifier
    model = SentimentClassifier(4)
    bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

    # load trained weights
    model.load_state_dict(torch.load('final_model.bin', map_location=device))
    model = model.to(device)

    MAX_LEN = 100  # token length
    class_names = ["anger", "fear", "joy", "sadness"]  # emotions

    # encode sentence
    encoded_review = tokenizer.encode_plus(
        review_text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    # prediction
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)

    probability = prediction_probabilities(model, output, class_names) * 100

    return class_names[prediction], probability


if __name__ == '__main__':
    review_text = input("Enter sentence: ")
    sentiment, probability = detect_emotion(review_text)
    print(f'Sentiment: {sentiment} \n Probability: {probability}')
