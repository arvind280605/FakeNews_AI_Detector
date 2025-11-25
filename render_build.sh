#!/bin/bash

# ---------------------- CREATE MODEL DIRECTORIES -------------------------
mkdir -p model/ai_detector_model
mkdir -p model/final_model

# ---------------------- INSTALL GDOWN -------------------------
echo "Checking gdown..."
if ! command -v gdown &> /dev/null
then
    echo "gdown not found. Installing..."
    pip install --quiet gdown
fi

# ---------------------- DOWNLOAD AI DETECTOR MODEL ----------------------
echo "Downloading AI detector model..."
# Google Drive folder ID for AI detector model
AI_MODEL_FOLDER_ID="1wkflcPBJ81FSTzeYAPN0JWHfw5ZX-nQH"

# Download all files in the folder
gdown --folder https://drive.google.com/drive/folders/$AI_MODEL_FOLDER_ID -O model/ai_detector_model

# ---------------------- DOWNLOAD FAKE MODEL -----------------------------
echo "Downloading Fake model..."
# Google Drive folder ID for Fake model
FAKE_MODEL_FOLDER_ID="1sIFq6Iuon-qp2W_Gen7RPhzzl-HauVrE"

# Download all files in the folder
gdown --folder https://drive.google.com/drive/folders/$FAKE_MODEL_FOLDER_ID -O model/final_model

echo "All models downloaded successfully!"
