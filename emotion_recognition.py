import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn, optim
import torch.nn.functional as F

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

def emotion_recognition(review_text):
    print(1)
    device = torch.device("cpu")
    print(2)
    PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
    print(3)
    # load pretrained bert tokenizer
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    print(4)
    model = SentimentClassifier(4)
    print(5)
    bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    print(6)
    model.load_state_dict(torch.load('final_model.bin', map_location=device))
    print(7)
    model = model.to(device)
    MAX_LEN = 100

    # BERT emotion analysis
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

    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)

    class_names = ["anger", "fear", "joy", "sadness"]

    print(f'Review text: {review_text}')
    print(f'Sentiment  : {class_names[prediction]}')


if __name__ == '__main__':
    review_text = input("Enter sentence: ")
    emotion_recognition(review_text)
