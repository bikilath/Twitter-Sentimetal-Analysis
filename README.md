# Twitter Sentiment Analysis: NNM vs SVM vs LSTM

## Overview
This project performs sentiment analysis on Twitter data using three different machine learning models:  
- **Neural Network Model (NNM)**  
- **Support Vector Machine (SVM)**  
- **Long Short-Term Memory (LSTM) Neural Network**

The goal is to compare the performance of these models in classifying tweets based on their sentiment (positive, negative, neutral).

---

## Project Structure
- `model_nnm/` — Contains the Neural Network model code and related files.  
- `model_svm/` — Contains the SVM model code and related files.  
- `model_lstm/` — Contains the LSTM model code and related files.  
- `data/` — Dataset files used for training and testing the models.  
- `README.md` — This file.

---

## Models and Methods

### Neural Network Model (NNM)
- Fully connected feedforward neural network.
- Input features prepared using text vectorization techniques.
- Used ReLU activation and softmax output for multi-class classification.
- Trained with Adam optimizer and categorical cross-entropy loss.

### Support Vector Machine (SVM)
- Traditional machine learning classifier.
- Used TF-IDF features extracted from tweets.
- Tuned hyperparameters for optimal classification.

### Long Short-Term Memory (LSTM)
- Recurrent neural network architecture for sequential data.
- Utilized word embeddings and tokenized padded sequences.
- Employed dropout and Adam optimizer during training.
- Evaluated using accuracy, precision, recall, and F1-score.

---

## Summary of Findings
Among the three models, the **LSTM model performed the best**, demonstrating superior ability to understand the contextual and sequential nature of the tweet data.  
The Neural Network Model (NNM) showed decent performance, while the SVM, though simpler, was less effective in capturing the nuances of the text.

---

## How to Use
1. Load the dataset from the `data/` directory.  
2. Run the scripts inside each model folder (`model_nnm/`, `model_svm/`, and `model_lstm/`) to train and evaluate the respective models.  
3. Compare the printed or saved metrics to analyze model performance.

---

## Technologies Used
- Python  
- TensorFlow / Keras (for NNM and LSTM)  
- scikit-learn (for SVM)  
- Pandas, NumPy  
- Google Colab (for development environment)  

---

## Contact
For questions or collaboration, please contact me at biktef21@gmail.com.

---

Thank you for checking out the project!
