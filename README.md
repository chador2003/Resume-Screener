# 🤖 Resume Screener – AI-Powered Resume Classification System

Resume Screener is a smart, automated solution that leverages Natural Language Processing (NLP) and Machine Learning (ML) to streamline the resume screening process. Designed to help HR professionals and recruiters categorize and evaluate resumes efficiently, this system enables fast and fair candidate filtering based on job relevance.

> 🧠 Built with deep learning, smart preprocessing, and intelligent classification.

---

## 📌 Table of Contents

- [🚀 Features](#-features)
- [🧪 Models Used](#-models-used)
- [📊 Performance](#-performance)
- [🛠️ Tech Stack](#-tech-stack)
- [🧬 Methodology](#-methodology)
- [📁 Dataset](#-dataset)
- [⚙️ How to Run](#-how-to-run)
- [📚 References](#-references)
- [👥 Contributors](#-contributors)

---

## 🚀 Features

✅ Automatically classifies resumes into job categories  
✅ Supports input from structured text and raw PDF files  
✅ Applies NLP for cleaning, lemmatization, and stopword removal  
✅ Uses TF-IDF and label encoding for feature extraction  
✅ Balances class distribution using data augmentation  
✅ Supports PDF resume parsing using `pdfplumber`  
✅ Evaluates model performance using accuracy, F1-score, and more

---

## 🧪 Models Used

- **Multinomial Logistic Regression** – Best performance with 90% accuracy  
- **Naive Bayes** – Fast and effective for high-dimensional data  
- **Support Vector Machine (SVM)** – Good margin separation  
- **Random Forest** – Robust and versatile  
- **Gated Recurrent Unit (GRU)** – Effective for sequence-based classification

---

## 📊 Performance

| Model                       | Accuracy | F1 Score |
|----------------------------|----------|----------|
| Multinomial Logistic Reg.  | 90%      | 0.90     |
| GRU (Deep Learning)        | ~88%     | 0.89     |
| SVM                        | ~87%     | 0.87     |
| Random Forest              | ~85%     | 0.86     |
| Naive Bayes                | ~84%     | 0.84     |

📌 Evaluation done using classification report (precision, recall, F1, support)

---

## 🛠️ Tech Stack

| Layer         | Technology                                  |
|---------------|---------------------------------------------|
| **Language**  | Python 3.x                                  |
| **Libraries** | `scikit-learn`, `nltk`, `pdfplumber`, `pandas`, `matplotlib`, `seaborn` |
| **Models**    | Logistic Regression, Naive Bayes, SVM, Random Forest, GRU (Keras) |
| **Input**     | CSV files, PDF resumes                      |

---

## 🧬 Methodology

1. **Data Collection**  
   - Kaggle resume datasets  
   - Real-world PDF resumes

2. **Preprocessing**  
   - Lowercasing, stopword removal, lemmatization  
   - Tokenization, punctuation & number cleaning  
   - TF-IDF vectorization

3. **Data Augmentation**  
   - Synonym replacement to balance class distribution

4. **Feature Engineering**  
   - Label Encoding for categories  
   - TF-IDF vector representation

5. **Model Training & Evaluation**  
   - Train/test split (80/20)  
   - Hyperparameter tuning for best performance  
   - Evaluation metrics: Accuracy, F1-score, Precision, Recall

---

## 📁 Dataset

- ✅ [Kaggle Resume Dataset #1](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)  
- ✅ [Kaggle Resume Dataset #2](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)  
- ✅ Extracted PDFs using `pdfplumber` for real-world input

---
