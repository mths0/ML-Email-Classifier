# ML Email Classifier 🚀

[![Made with ❤️ by Muhannad](https://img.shields.io/badge/Made%20By-Muhannad-blue)](https://github.com/mths0)
[![License](https://img.shields.io/github/license/mths0/ML-Email-Classifier)](LICENSE)

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

├── LICENSE <- Open-source license ├── README.md <- Project overview ├── data │ ├── raw <- Raw dataset (not pushed if large) │ └── external <- External data if added │ ├── models <- Saved trained models (e.g., spam_classifier.pkl) │ ├── notebooks <- Exploratory data analysis and experiments │ ├── references <- Supporting material (data dictionaries, resources) │ ├── reports │ └── figures <- Generated visualizations (charts, graphs) │ ├── requirements.txt <- List of Python libraries used │ └── src <- Source code ├── init.py ├── dataset.py <- Data loading and preprocessing ├── modeling │ ├── init.py │ ├── train.py <- Training the Naïve Bayes model │ └── predict.py <- Inference and prediction scripts └── services └── init.py

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

📜 License
This project is licensed under the terms of the MIT license.

🙌 Credits
Created with ❤️ by Muhannad Alshahrani

yaml
Copy
Edit

---

# ✅ What you can do now:

- Copy this README into your `README.md` file.
- (Optional) Add a nice small project image at the top later (like a spam filter icon).

---

# 🚀 Extra Offer:

Would you like me to also prepare:
- A small **`requirements.txt`** refresh
- A small **badge** like `accuracy: 99%` for your GitHub page?
- A **sample output screenshot** to put in README?

It would make your GitHub look **next level professional**! 🔥🌟  
Just say "yes"! 🚀
