import sys
import os
import pandas as pd

# -----------------------------------------
# Fix import path so Python can find `src/`
# -----------------------------------------

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from src.preprocessing.clean_text import clean_text


# ------------------------------------------------------
#  FAKE NEWS DATASET (LIAR dataset: train/valid/test TSV)
# ------------------------------------------------------
def load_liar_dataset(train_path, valid_path, test_path):
    print("📌 Loading Fake News (LIAR) Dataset...")

    # Load TSV files — No header included
    train_df = pd.read_csv(train_path, sep="\t", header=None, encoding="utf-8")
    valid_df = pd.read_csv(valid_path, sep="\t", header=None, encoding="utf-8")
    test_df  = pd.read_csv(test_path,  sep="\t", header=None, encoding="utf-8")

    # Keep only label (col 1) and text (col 2)
    train_df = train_df[[1, 2]].rename(columns={1: "label", 2: "text"})
    valid_df = valid_df[[1, 2]].rename(columns={1: "label", 2: "text"})
    test_df  = test_df[[1, 2]].rename(columns={1: "label", 2: "text"})

    # Convert LIAR dataset labels to binary:
    # 1 = Fake, 0 = Real
    fake_labels = ["false", "pants-fire"]
    real_labels = ["true", "mostly-true", "half-true"]

    def convert_label(x):
        return 1 if x in fake_labels else 0

    train_df["label"] = train_df["label"].apply(convert_label)
    valid_df["label"] = valid_df["label"].apply(convert_label)
    test_df["label"]  = test_df["label"].apply(convert_label)

    # Clean text
    print("📌 Cleaning Fake News text...")
    train_df["text"] = train_df["text"].apply(clean_text)
    valid_df["text"] = valid_df["text"].apply(clean_text)
    test_df["text"]  = test_df["text"].apply(clean_text)

    return train_df, valid_df, test_df


# ---------------------------------------------------
#  AI vs HUMAN WRITTEN DATASET (ChatGPT detection)
# ---------------------------------------------------
# ---------- AI vs HUMAN DATASET ----------
def load_ai_data(article_path, sentence_path):
    article_df = pd.read_csv(article_path)
    sentence_df = pd.read_csv(sentence_path)

    # Rename actual columns to standard names
    article_df = article_df.rename(columns={"article": "text", "class": "label"})
    sentence_df = sentence_df.rename(columns={"sentence": "text", "class": "label"})

    # Drop unwanted index column if present
    if "Unnamed: 0" in article_df.columns:
        article_df = article_df.drop(columns=["Unnamed: 0"])

    if "Unnamed: 0" in sentence_df.columns:
        sentence_df = sentence_df.drop(columns=["Unnamed: 0"])

    # Clean text
    article_df["text"] = article_df["text"].apply(clean_text)
    sentence_df["text"] = sentence_df["text"].apply(clean_text)

    return article_df, sentence_df



# -----------------------------------------
# Test the loader (run this file directly)
# -----------------------------------------
if __name__ == "__main__":

    # Fake News Paths
    train_path = r"D:\FakeNews_AI_Detection\data\raw\train.tsv"
    valid_path = r"D:\FakeNews_AI_Detection\data\raw\valid.tsv"
    test_path  = r"D:\FakeNews_AI_Detection\data\raw\test.tsv"

    train_df, valid_df, test_df = load_liar_dataset(train_path, valid_path, test_path)

    print("\nTrain shape:", train_df.shape)
    print("Valid shape:", valid_df.shape)
    print("Test shape:", test_df.shape)

    # AI/Human Paths
    article_path  = r"D:\FakeNews_AI_Detection\data\raw\article_level_data.csv"
    sentence_path = r"D:\FakeNews_AI_Detection\data\raw\sentence_level_data.csv"

    article_df, sentence_df = load_ai_data(article_path, sentence_path)

    print("\nArticle dataset shape:", article_df.shape)
    print("Sentence dataset shape:", sentence_df.shape)
