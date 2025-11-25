import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

article = pd.read_csv("article_clean.csv")
sentence = pd.read_csv("sentence_clean.csv")

data = pd.concat([article, sentence])

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

X = tokenizer(
    list(data["text"]),
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="tf"
)

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-5),
    loss=model.compute_loss,
    metrics=["accuracy"]
)

model.fit(
    X["input_ids"],
    data["label"],
    epochs=2,
    batch_size=16
)

model.save_pretrained("ai_detector_bert_model")
