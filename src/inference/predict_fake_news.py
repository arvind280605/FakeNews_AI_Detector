from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

model = TFBertForSequenceClassification.from_pretrained("fake_news_bert_model")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def predict(text):
    tokens = tokenizer(text, return_tensors="tf", padding=True, truncation=True)
    output = model(tokens["input_ids"])
    pred = tf.argmax(output.logits, axis=1).numpy()[0]
    return "FAKE" if pred == 1 else "REAL"

if __name__ == "__main__":
    print(predict("Obama was born in Kenya."))
