 Fake News & AI Text Detection System

A Streamlit-based intelligent analysis system that detects fake news, AI-generated text, and plagiarism-like patterns in assignments, articles, or audio transcriptions using advanced NLP and deep learning models.

The system accepts text through manual input, voice input, web scraping, or file uploads (TXT, PDF, CSV, WAV, MP3) and generates a detailed report with probabilities, charts, and confidence metrics.

Features:

Fake News Detection

Classifies text as REAL or FAKE using a deep learning classifier.

AI Text Detection

Distinguishes between HUMAN-written and AI-generated content.

Shows probability scores and confidence levels.

Multi-Input Support

Typed text

Voice input (WAV / MP3)

File uploads (TXT, PDF, CSV, audio files)

Web scraping (extracts text from URLs, blogs, news articles, Instagram posts, etc.)

Chunked Processing

Automatically splits large text into chunks for efficient processing.

Prevents memory issues and ensures fast analysis.

Visualization

Bar charts for fake news confidence

Pie charts for AI detection results

Progress meters during processing

PDF Report Generation

One-click export of a full PDF report containing:

Extracted text

Fake news result

AI-text probability

Charts and confidence scores

Smart Alerts

Warns users if the input text is too short.

Alerts when large files may take longer to process.

Technologies Used:

Python 3.10+

Streamlit – Interactive web UI

PyTorch – Deep learning model backend

Transformers (HuggingFace) – Pre-trained NLP models

Librosa & SpeechRecognition – Audio processing

Matplotlib – Charts and visualizations

PyPDF2 – PDF extraction and processing

Pandas – CSV and data handling

How It Works:
1. Input Text

Users can type, speak, upload, or scrape text from a URL.

2. Preprocessing

Text is cleaned

Large files are split into chunks for efficient model processing

3. Prediction

Fake News Model: Predicts REAL or FAKE

AI Detector: Computes HUMAN vs AI probabilities

4. Visualization

Confidence bar charts

AI detection pie chart

Progress indicators

5. Export Report

Generates a downloadable PDF summary of the complete analysis.
