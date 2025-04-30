# ML Email Classifier 

[![Made with ❤️ by mths0](https://img.shields.io/badge/Made%20By-mths0-blue)](https://github.com/mths0)
[![Made with ❤️ by Saudll](https://img.shields.io/badge/Made%20By-Saudll-blue)](https://github.com/Saudll)



This project is a **Machine Learning-based Email Classifier** that detects whether an email is **Spam** or **Ham (Not Spam)** using a **Multinomial Naïve Bayes** model trained on labled email dataset.


---

## Project Goal

The main goal is to **build an accurate spam detection system** using:
- Data balancing techniques
- TF-IDF feature extraction
- Hyperparameter tuning (`alpha`, n-grams)
- Model evaluation using Precision, Recall, F1-Score
- A final trained model ready for deployment

---

## 📂 Project Structure

<pre>
├── LICENSE                
├── README.md              
├── data
│   ├── raw                
│   └── external           
│
├── models                 <- Saved trained models
│
├── notebooks              
│
├── references             
│
├── reports
│   └── figures            
│
├── requirements.txt       <- List of Python libraries used
│
└── src                    
    ├── spam_email_classifier.py
    ├── data_analys.ipynb          
    ├── GUI.py
    │   
    │   
    │   
    └── services
        
</pre>

---

 Main Features
Preprocessing: Tokenization, stopword removal, stemming

Balancing Data: Upsampling spam samples for better learning

Feature Extraction: TF-IDF with n-grams (1,2)

Model: Multinomial Naïve Bayes with tuned alpha

Evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

Live Testing: Enter email text and get instant prediction (spam or ham)

## Results

During the development, different versions of the model were trained and evaluated to improve performance.
Here’s a summary of the experiments:

| Model Version                    | Accuracy | Ham Precision | Ham Recall | Spam Precision | Spam Recall | Notes |
|-----------------------------------|----------|---------------|------------|----------------|-------------|-------|
| **Default** (No tuning)           | 96%      | 0.95          | 1.00       | 1.00           | 0.66        | Model biased towards ham |
| **TF-IDF tuned** (ngram 1–2,
|            min_df=5, max_df=0.9)  | 97%      | 0.96          | 0.98       | 0.98           | 0.96        | Bigram features helped |
| **Naïve Bayes (α = 0.5)**         | 98%      | 0.97          | 0.98       | 0.98           | 0.97        | Smoother model |
| **Naïve Bayes (α = 0.3)**         | 99%      | 0.99          | 0.99       | 1.00           | 0.99        | Best balance achieved |
| **Naïve Bayes (α = 0.1)**         | 99%      | 0.99          | 0.99       | 0.99           | 0.99        | Sharpest decision boundary |

 **Final Best Model**:  
- **Vectorizer Settings**: `TfidfVectorizer(max_df=0.9, min_df=5, ngram_range=(1,2))`
- **Classifier**: `MultinomialNB(alpha=0.3)`
- **Test Accuracy**: **99%**
- **Spam Recall**: **99%**  
- **Spam Precision**: **100%**


---

### Key Observations:

- Without balancing the dataset, the model missed a lot of spam (low recall 66%).
- Adding n-grams helped the model detect spam **phrases** (like "click here" or "free money").
- Fine-tuning the `alpha` parameter increased confidence without overfitting.
- The final model detects **both spam and ham** emails good enogh.

---









