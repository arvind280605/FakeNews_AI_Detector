🧠 Fake News & AI Text Detection System

A Streamlit-based application that detects fake news, AI-generated text, and plagiarism in assignments or articles using advanced NLP and deep learning models.

The system analyzes text from manual input, voice input, or uploaded files (TXT, PDF, CSV, WAV, MP3) and generates a detailed report with probabilities, charts, and confidence metrics.

📌 Features

Fake News Detection – Classifies text as REAL or FAKE.

AI Text Detection – Identifies HUMAN-written vs AI-generated text.

Multi-input Support:

Typed text

Web Scraping

Extract text from URLs such as articles, blogs, Instagram posts, etc.
(Using BeautifulSoup / Requests, depending on your implementation.)

Voice input

File upload (TXT, PDF, CSV, WAV, MP3)

Chunked Processing – Handles large text files efficiently.

Visualization – Bar chart for fake news confidence, pie chart for AI detection, and progress meter.

PDF Report Generation – Download full analysis with text and probabilities.

Smart Alerts:

Warns if input text is very short (predictions may be unreliable).

Informs users when large files may take longer to process.

🛠 Technologies Used

Python 3.10+

Streamlit – Interactive web interface

PyTorch – Deep learning backend

Transformers (Hugging Face) – Pre-trained NLP models

Librosa & SpeechRecognition – Audio processing

Matplotlib – Charts for visualization

PyPDF2 & Pandas – PDF and CSV file handling

📝 How It Works

Input Text: Type, speak, or upload a file.

Preprocessing: Text is cleaned and split into chunks if large.

Prediction:

Fake news model outputs REAL or FAKE probabilities.

AI detection model outputs HUMAN or AI-generated probabilities.




Visualization: Shows charts and progress bars for confidence.

Export Report: Download a PDF summarizing results and analyzed text.
