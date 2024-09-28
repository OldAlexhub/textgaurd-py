# TextGuard AI Spam Detection API

TextGuard AI is a Flask-based API designed to detect spam messages using machine learning. The API uses a **RandomForestClassifier** to classify user-submitted text as either "Good" (ham) or "Spam". It is powered by a trained model that utilizes **TF-IDF** vectorization for text feature extraction and a dataset stored in MongoDB.

## Features

- **Spam Detection**: Submits text for analysis, returning whether the message is spam or legitimate.
- **Pre-trained Model**: The API uses a Random Forest model trained on a dataset of text messages.
- **MongoDB Integration**: Text messages and classifications are stored and retrieved from MongoDB.
- **RESTful API**: Simple API interface for interacting with the model.

## Technologies Used

- **Flask**: A Python micro-framework for creating the API.
- **scikit-learn**: Machine learning library for training the Random Forest classifier.
- **MongoDB**: NoSQL database used for storing the dataset.
- **TF-IDF**: Technique used to convert text into numerical features.
- **dotenv**: Manages environment variables securely.
- **CORS**: Handles cross-origin resource sharing for the API.

## Setup and Installation

### Prerequisites

- Python 3.x
- MongoDB instance (either local or cloud-based)
- Required Python libraries (listed below)

### Clone the Repository

```bash
git clone https://github.com/OldAlexhub/textgaurd-py.git
cd textguard-ai-api
```
