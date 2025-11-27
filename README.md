🧠 Fake News & AI Text Detection System

A Streamlit-based application that detects fake news, AI-generated text, and plagiarism in assignments or articles using advanced NLP and deep learning models.

The system analyzes text from manual input, voice input, or uploaded files (TXT, PDF, CSV, WAV, MP3) and generates a detailed report with probabilities, charts, and confidence metrics.

📌 Features

📰 Fake News Detection

Classifies text as REAL or FAKE.

🤖 AI Text Detection

Identifies HUMAN-written vs AI-generated text.

📝 Multi-input Support

Typed text

Web Scraping

Extract text from URLs (articles, blogs, Instagram posts, etc.)

Voice input

File upload (TXT, PDF, CSV, WAV, MP3)

📚 Chunked Processing

Handles large text files efficiently.

📊 Visualization

Bar chart for fake news confidence

Pie chart for AI detection

Progress meter

📄 PDF Report Generation

Download full analysis with text and probabilities.

⚠️ Smart Alerts

Warns if input text is too short

Notifies when large files may take longer

🛠 Technologies Used

🐍 Python 3.10+

🌐 Streamlit – Interactive web interface

🔥 PyTorch – Deep learning backend

🤗 Transformers – Pre-trained NLP models

🎤 Librosa & SpeechRecognition – Audio processing

📈 Matplotlib – Charts & plots

📄 PyPDF2 – PDF handling

🧮 Pandas – CSV / data processing

📝 How It Works

🧾 Input Text
Type, speak, upload, or scrape text.

🧹 Preprocessing
Text is cleaned and chunked if large.

🎯 Prediction

Fake news model → REAL or FAKE

AI detector → HUMAN or AI probabilities

📊 Visualization
Charts + progress bars for confidence.

📥 Export Report
Download a PDF summarizing the full analysis.
