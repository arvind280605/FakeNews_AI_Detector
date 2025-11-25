import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

train = pd.read_csv("train_clean.csv")
valid = pd.read_csv("valid_clean.csv")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(data):
    return tokenizer(
        list(data["text"]),
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="tf"
    )

X_train = tokenize(train)
X_valid = tokenize(valid)

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-5),
    loss=model.compute_loss,
    metrics=["accuracy"]
)

model.fit(
    X_train["input_ids"],
    train["label"],
    validation_data=(X_valid["input_ids"], valid["label"]),
    epochs=2,
    batch_size=16
)

model.save_pretrained("fake_news_bert_model")
