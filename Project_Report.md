# PROJECT REPORT
## Fake and Real News Detection System
**Course:** BCA – Data Science  
**Project Type:** Minor Project

### 1. Abstract
The Fake and Real News Detection System is a machine-learning application that analyses text entered by a user and predicts whether it resembles examples labelled REAL or FAKE in a training dataset. It uses TF-IDF to convert text into numerical features and Logistic Regression for classification. A Streamlit web interface makes the system easy to use.

### 2. Introduction
Online platforms allow information to spread quickly. Misleading news can confuse readers and influence decisions. Machine learning can assist by identifying linguistic patterns in labelled examples. This project demonstrates a basic text-classification workflow; it does not independently verify facts.

### 3. Problem Statement
To develop a simple application that accepts news text and classifies it into one of two categories, REAL or FAKE, using supervised machine learning.

### 4. Objectives
- Understand text preprocessing and feature extraction.
- Build a supervised text-classification model.
- Train and evaluate Logistic Regression.
- Create a simple user interface using Streamlit.
- Demonstrate predictions on user-entered text.

### 5. Existing and Proposed System
**Existing approach:** Readers may rely only on manual checking, which can take time and may be inconsistent.  
**Proposed approach:** A trained classifier provides an instant preliminary text-pattern prediction. Human verification remains necessary.

### 6. Tools and Technologies
Python, Pandas, scikit-learn, TF-IDF, Logistic Regression, Joblib and Streamlit.

### 7. System Requirements
**Hardware:** Basic laptop/desktop, 4 GB RAM recommended, internet connection for initial package installation.  
**Software:** Windows/Linux/macOS, Python 3.10+, VS Code or another Python editor, required Python packages.

### 8. Methodology
1. Collect labelled news text.
2. Clean missing values and standardise labels.
3. Split data into training and testing subsets.
4. Apply TF-IDF vectorisation with unigrams and bigrams.
5. Train Logistic Regression.
6. Evaluate using accuracy, precision, recall, F1-score and confusion matrix.
7. Save the trained pipeline.
8. Load it in Streamlit and predict user input.

### 9. Algorithm
**Input:** News text  
**Output:** REAL or FAKE prediction

- Read the labelled CSV dataset.
- Separate text and labels.
- Split into training and testing data.
- Convert text to TF-IDF vectors.
- Train Logistic Regression.
- Predict labels for test data and calculate evaluation metrics.
- Save the complete pipeline.
- For new text, load the pipeline and return the predicted class.

### 10. Modules
**Dataset module:** Stores text and labels.  
**Training module:** Trains and evaluates the classifier.  
**Prediction module:** Loads the saved model and predicts a class.  
**Interface module:** Accepts user text and displays output.

### 11. Evaluation
Run `python train_model.py`. The terminal prints test accuracy, classification report and confusion matrix. Record the actual values produced by your chosen dataset here; do not copy or invent accuracy figures. The included small sample dataset is only for demonstration.

### 12. Advantages
- Simple and quick to use.
- Demonstrates an end-to-end ML workflow.
- Low computational requirements.
- Can be retrained with a larger dataset.

### 13. Limitations
- Results depend on dataset size, quality and balance.
- A model may learn writing style rather than factual truth.
- New topics and evolving misinformation can reduce performance.
- It cannot replace professional fact-checking.

### 14. Future Scope
- Use a larger, diverse and well-verified dataset.
- Add multilingual support.
- Compare Naive Bayes, SVM and transformer-based models.
- Add source credibility and claim verification features.
- Deploy the application online after security and performance testing.

### 15. Conclusion
The project demonstrates how TF-IDF and Logistic Regression can be used for news-text classification and how a trained model can be integrated into a Streamlit application. Its output should be treated as an initial indication only, not a final judgement of truth.

### 16. References
- scikit-learn documentation: https://scikit-learn.org/
- Streamlit documentation: https://docs.streamlit.io/
- Pandas documentation: https://pandas.pydata.org/docs/
