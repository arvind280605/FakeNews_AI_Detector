#!/bin/bash

# ---------------------- CLEAN OLD MODELS -------------------------
rm -rf model/ai_detector_model
rm -rf model/final_model

# ---------------------- CREATE MODEL DIRECTORIES -----------------
mkdir -p model/ai_detector_model
mkdir -p model/final_model

# ---------------------- DOWNLOAD AI DETECTOR MODEL ----------------
echo "Downloading AI detector model..."
AI_MODEL_FOLDER_ID="1wkflcPBJ81FSTzeYAPN0JWHfw5ZX-nQH"
pip install --quiet gdown
gdown --folder https://drive.google.com/drive/folders/$AI_MODEL_FOLDER_ID -O model/ai_detector_model

# ---------------------- DOWNLOAD FAKE MODEL ---------------------
echo "Downloading Fake model..."
FAKE_MODEL_FOLDER_ID="1sIFq6Iuon-qp2W_Gen7RPhzzl-HauVrE"
gdown --folder https://drive.google.com/drive/folders/$FAKE_MODEL_FOLDER_ID -O model/final_model

echo "All models downloaded successfully!"
