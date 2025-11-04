# CRSD-Sentiment-Analysis

A machine learning project built using **Customer Review Sentiment Dataset (CRSD)** that classifies customer reviews into **Positive**, **Neutral**, or **Negative** sentiments. This project combines **NLP preprocessing, vectorization, model training** and a **FastAPI interactive web interface**.

---

## Dataset Overview
The dataset used in this project is a synthetic dataset generated using different AI language models for sentiment analysis tasks.
**Source:** https://github.com/Infinitode/CRSD/tree/main

**Columns:**

review: the review text

model: AI language model used for generating the review text 

sentiment: sentiment of the review

**Records:** 8200

---

## Project Structure

CRSD-Sentiment-Analysis/

│─ templates/index.html        # the UI

├─ README.md                  

├─ data.csv                    # Customer Review Sentiment Data in csv 

├─ main.ipynb                  # Jupyter Notebook(data/text preprocessing, vectorization, machine learning)

├─ requirements.txt

│─ sentiment_pipeline.joblib   # exported model for inference

├─ server.py                   # FastAPI web server
       
---

## Methodology

- **Data Preprocessing**
Converted columns to the appropriate data types.
Dropped null values.

- **Exploratory Data Analysis**
The three columns were explored for finding any hidden patterns in the data.

- **Text Preprocessing**
Tokenization, stopword removal, lemmatization and emoji removal using spaCy's large English model.
Implemented TF-IDF vectorization technique.

- **Machine Learning**
The TF-IDF vectorization was paired traditional ML classifiers:
i. Logistic Regression
ii. Multinomial Naive Bayes
iii. Decision Tree

The best performing model, Logistic Regression, was exported as a unified pipeline for inference.

---

## Model Evaluation
|    Model    | Accuracy | F1-Score |
|-------------|----------|----------|
|Logistic Regression| 0.92 | 0.92 |
|Multinomial Naive Bayes| 0.89 | 0.89 |
|Decision Tree| 0.85 | 0.85 |

---

## Web Application
The trained model is deployed with **FastAPI** and served with a lightweight HTML landing page.

---

## Run the app locally
- Install all dependencies using the command on terminal: pip install -r requirements.txt

- Run FastAPI server: uvicorn server:app --reload --port 9000
  
- Access the app: http://localhost:9000/

---

## Example Classification

- Positive sentiment classification

<img width="729" height="414" alt="Image" src="https://github.com/user-attachments/assets/78950ecc-ffe1-4f5d-ac60-12b958d6175e" />


- Negative sentiment classification

<img width="718" height="405" alt="Image" src="https://github.com/user-attachments/assets/66d97943-65ff-4ca6-903a-16007fdfaa16" />


- Neutral sentiment classification

<img width="720" height="408" alt="Image" src="https://github.com/user-attachments/assets/6769c5d5-4574-4035-ab4a-8d524ae43532" />
