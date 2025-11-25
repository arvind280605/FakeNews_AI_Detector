# files.py
import os
import sys
import pandas as pd

# Add preprocessing folder to Python path so we can import load_dataset
sys.path.append(r"D:\FakeNews_AI_Detection\src\preprocessing")

from load_dataset import load_liar_dataset, load_ai_data

# ----------------- Paths -----------------
train_path = r"D:\FakeNews_AI_Detection\data\raw\train.tsv"
valid_path = r"D:\FakeNews_AI_Detection\data\raw\valid.tsv"
test_path  = r"D:\FakeNews_AI_Detection\data\raw\test.tsv"

article_path  = r"D:\FakeNews_AI_Detection\data\raw\article_level_data.csv"
sentence_path = r"D:\FakeNews_AI_Detection\data\raw\sentence_level_data.csv"

processed_dir = r"D:\FakeNews_AI_Detection\data\processed"
os.makedirs(processed_dir, exist_ok=True)

# ----------------- Load & Clean -----------------
train_df, valid_df, test_df = load_liar_dataset(train_path, valid_path, test_path)
article_df, sentence_df = load_ai_data(article_path, sentence_path)

# ----------------- Combine all datasets -----------------
combined_df = pd.concat([train_df, valid_df, test_df, article_df, sentence_df], ignore_index=True)

# ----------------- Save -----------------
output_file = os.path.join(processed_dir, "processed_data.csv")
combined_df.to_csv(output_file, index=False)

print(f"✅ All processed data saved as: {output_file}")
print(f"Shape of combined data: {combined_df.shape}")
