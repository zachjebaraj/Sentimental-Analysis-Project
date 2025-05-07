# Sentimental-Analysis-Project using Streamlit
Sentiment Analysis for Customer Feedback using Streamlit
# 1. Project Overview
This project implements a Sentiment Analysis system to classify customer feedback as Positive, Negative, or Neutral. The application uses Natural Language Processing (NLP) and Machine Learning to automate sentiment classification and visualize trends in real-time through a Streamlit dashboard.
# 2. Features
• User input interface for analyzing customer feedback

• Sentiment classification using Logistic Regression

• Real-time sentiment trend visualization

• Feedback history tracking within the session

# 3. Technologies Used
• Python 3

• Streamlit

• Scikit-learn

• Pandas

• TfidfVectorizer

• Regex for text preprocessing

# 4. Project Structure

• app.py               - Main Streamlit application

• model.py             - Machine Learning model training and prediction

• utils.py             - Text cleaning functions

• sample_data.csv      - Example dataset for training

• requirements.txt     - List of dependencies
# 5. How to Run the App

  # 1. Create a virtual environment:

     python -m venv .venv

  # 2. Activate the environment:

     • On macOS/Linux: source .venv/bin/activate

     • On Windows: .venv\Scripts\activate

  # 3. Install dependencies:

     pip install -r requirements.txt

  # 4. Run the application:

     streamlit run app.py

# 6. Sample Output
The app will show the predicted sentiment for the input text and display a chart of sentiment distribution from all processed inputs.
