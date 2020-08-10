import torch

def emotion_recognition(review_text):
    device = torch.device("cpu")
    model.load_state_dict(torch.load(PATH))

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

    output = model(input_ids, attentionmask)
    , prediction = torch.max(output, dim=1)

    print(f'Review text: {review_text}')
    print(f'Sentiment  : {class_names[prediction]}')


if name == 'main':
    review_text = input("Enter sentence: ")
    emotion_recognition(review_text)
