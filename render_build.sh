#!/bin/bash

# Ensure LFS files are pulled
git lfs install
git lfs pull

# Install Python dependencies
pip install -r requirements.txt
