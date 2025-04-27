# ML Email Classifier 🚀

[![Made with ❤️ by mths0](https://img.shields.io/badge/Made%20By-mths0-blue)](https://github.com/mths0)
[![Made with ❤️ by Saudll](https://img.shields.io/badge/Made%20By-Saudll-blue)](https://github.com/Saudll)



This project is a **Machine Learning-based Email Classifier** that detects whether an email is **Spam** or **Ham (Not Spam)** using a **Multinomial Naïve Bayes** model trained on a text dataset.

It is structured based on a simplified **Cookiecutter Data Science** format and ready for deployment.

---

## 🧠 Project Goal

The main goal is to **build an accurate spam detection system** using:
- Data balancing techniques
- TF-IDF feature extraction
- Hyperparameter tuning (`alpha`, n-grams)
- Model evaluation using Precision, Recall, F1-Score
- A final trained model ready for deployment

---

## 📂 Project Structure

<pre>
├── LICENSE                <- Open-source license
├── README.md              <- Project overview
├── data
│   ├── raw                <- Raw dataset (not pushed if large)
│   └── external           <- External data if added
│
├── models                 <- Saved trained models (e.g., spam_classifier.pkl)
│
├── notebooks              <- Exploratory data analysis and experiments
│
├── references             <- Supporting material (data dictionaries, resources)
│
├── reports
│   └── figures            <- Generated visualizations (charts, graphs)
│
├── requirements.txt       <- List of Python libraries used
│
└── src                    <- Source code
    ├── __init__.py
    ├── dataset.py         <- Data loading and preprocessing
    ├── modeling
    │   ├── __init__.py
    │   ├── train.py       <- Training the Naïve Bayes model
    │   └── predict.py     <- Inference and prediction scripts
    └── services
        └── __init__.py

yaml
Copy
Edit

---

## 🛠️ How to Run

1. **Clone the repository**

```bash
git clone https://github.com/mths0/ML-Email-Classifier.git
cd ML-Email-Classifier
Install required libraries

bash
Copy
Edit
pip install -r requirements.txt
Train the model (if you want)

bash
Copy
Edit
python src/modeling/train.py
Or load the pre-trained model and predict

bash
Copy
Edit
python src/modeling/predict.py
📈 Main Features
Preprocessing: Tokenization, stopword removal, stemming

Balancing Data: Upsampling spam samples for better learning

Feature Extraction: TF-IDF with n-grams (1,2)

Model: Multinomial Naïve Bayes with tuned alpha

Evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

Live Testing: Enter email text and get instant prediction (spam or ham)

📊 Results

Model Setting	Accuracy	Spam Recall	Spam Precision
Default	96%	66%	100%
Tuned TF-IDF + α=0.3	99%	99%	99%
✅ Achieved 99% accuracy after hyperparameter tuning and balancing!

⚡ Tech Stack
Python 3.10+

scikit-learn

NLTK

pandas

matplotlib

seaborn



🙌 Credits
Created with ❤️ by mohannad and Saud

yaml
Copy
Edit



It would make your GitHub look **next level professional**! 🔥🌟  
Just say "yes"! 🚀
