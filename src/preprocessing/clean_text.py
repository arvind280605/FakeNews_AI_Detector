import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)               # remove links
    text = re.sub(r"@\w+", " ", text)                  # remove mentions
    text = re.sub(r"#\w+", " ", text)                  # remove hashtags
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)        # remove special chars
    text = re.sub(r"\s+", " ", text).strip()
    return text
