from transformers import AutoTokenizer

model_path = r"D:\FakeNews_AI_Detection\model\ai_detector_model"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.save_pretrained(model_path)

print("Tokenizer added successfully!")
