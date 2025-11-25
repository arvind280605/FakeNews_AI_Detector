import os
import sys

# -------- FIX PYTHON PATH --------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.preprocessing.load_dataset import load_liar_dataset, load_ai_data

# -------- FILE PATHS --------
train_path = r"D:\FakeNews_AI_Detection\data\raw\train.tsv"
valid_path = r"D:\FakeNews_AI_Detection\data\raw\valid.tsv"
test_path  = r"D:\FakeNews_AI_Detection\data\raw\test.tsv"

article_path  = r"D:\FakeNews_AI_Detection\data\raw\article_level_data.csv"
sentence_path = r"D:\FakeNews_AI_Detection\data\raw\sentence_level_data.csv"

# -------- LOAD LIAR DATASET --------
print("\n=== Loading LIAR Dataset ===")
train_df, valid_df, test_df = load_liar_dataset(train_path, valid_path, test_path)

print("Train shape :", train_df.shape)
print("Valid shape :", valid_df.shape)
print("Test shape  :", test_df.shape)

print("\n--- TRAIN SAMPLE ---")
print(train_df.head())

print("\n--- VALID SAMPLE ---")
print(valid_df.head())

print("\n--- TEST SAMPLE ---")
print(test_df.head())

# -------- LOAD ARTICLE & SENTENCE --------
print("\n=== Loading AI Fake News Dataset ===")
article_df, sentence_df = load_ai_data(article_path, sentence_path)

print("Article shape :", article_df.shape)
print("Sentence shape:", sentence_df.shape)

print("\n--- ARTICLE SAMPLE ---")
print(article_df.head())

print("\n--- SENTENCE SAMPLE ---")
print(sentence_df.head())
