from transformers import BertTokenizer, AutoTokenizer, AutoModelForSequenceClassification
import torch
from config import cfg
import numpy as np

list_of_tweets = [
    'I really like NVIDIA lately considering elons efforts with the company chances are it will go up',
    'NVIDIA just shit init'
]


tokenizer = AutoTokenizer.from_pretrained(cfg.dataset_loader_args['sentiment_model'])
sentiment_model = AutoModelForSequenceClassification.from_pretrained(cfg.dataset_loader_args['sentiment_model'])

encoded_inputs = tokenizer(
            list_of_tweets,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=250  # Adjust max_length as needed
        )

with torch.no_grad():
    outputs = sentiment_model(**encoded_inputs)
    predictions = outputs.logits.argmax(dim=1)
sentiment_scores = predictions.numpy() + 1
sentiment_scores

sentiment_scores
print(sentiment_scores, np.mean(sentiment_scores))