import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import string
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import os 
from sklearn.utils import resample
from sklearn.metrics import classification_report
import joblib


#for testing.
def predict_email(email, model, vector, print_state=True):
    # Convert email into numerical vector using the trained TF-IDF vectorizer
    email_vector = vector.transform([email])
    
    # Convert sparse matrix to dense array
    email_vector_dense = email_vector.toarray()
    
    prediction = model.predict(email_vector_dense)
    
    # Print or return based on print_state the prediction
    if print_state:
        if prediction[0] == "spam":
            print("The email is predicted as spam.")
        else:
            print("The email is predicted as ham.")
    else:
        if prediction[0] == "spam":
            return "spam"
        else:
            return "ham"
        

if __name__ == "__main__":
        

    #Reading the dataset from csv file.
    Dataset=pd.read_csv("spam.csv",encoding='latin1')
    Dataset.rename(columns={"v1":"label", "v2":"message"}, inplace=True)
    Dataset = Dataset[["label", "message"]]

    # ham,spam in percentage
    ham=Dataset[Dataset['label']== "ham"]
    spam=Dataset[Dataset['label']=="spam"]
    print('Spam Prercentage =',(len(spam)/len(Dataset['label']))*100,'%')
    print('ham Prercentage =',(len(ham)/len(Dataset['label']))*100,'%')

    # Upsample spam
    spam_upsampled = resample(
        spam,
        replace=True,                 # Sample with replacement
        n_samples=len(ham),            # Match the number of ham samples
        random_state=42
    )

    # Combine balanced data
    balanced_dataset = pd.concat([ham, spam_upsampled])

    # Shuffle the data
    balanced_dataset = balanced_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    Dataset = balanced_dataset


    #Preprocessing data
    test=Dataset.head(5)
    BeforeDS=Dataset.head(5)
    Dataset['message']=Dataset['message'].apply(word_tokenize)
    Dataset['message']=Dataset['message'].apply(lambda x: [re.sub(r'[^a-zA-Z0-9\s]', '', word) for word in x])
    stop_words = set(stopwords.words('english'))
    Dataset['message'] = Dataset['message'].apply(lambda x: [word for word in x if word not in stop_words and word not in string.punctuation])
    ps = PorterStemmer()
    Dataset['message'] = Dataset['message'].apply(lambda x: [ps.stem(word) for word in x]) 
    Dataset['message'] = Dataset['message'].apply(lambda x: ' '.join(x))


    #using: 
    # TfidfVectorizer(max_features=3000) 
    # train_test_split(x,y,test_size=0.2,random_state=42)
    #confusion Matrix : [[965   0] [ 23 127]]
    #Split the Dataset in to traning and testing.
    #max_df	 -->Ignore very common words (like "the", "you")
    # min_df -->Ignore very rare words
    # ngram_range=(1,2) --.	Include word pairs ("free money") not just single words
    Vectorizer= TfidfVectorizer(max_df=0.9, min_df=5,ngram_range=(1,2))
    x=Vectorizer.fit_transform(Dataset['message'])
    #5572 emails and 8713 unique words (features).
    #prints--> (5572, 8713)
    print(x.shape)
    y=Dataset['label']  
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25)
    joblib.dump(Vectorizer, "C:/Users\Mohannad/git repository/ML-Email-Classifier/models/Vectorizer.pkl")

    #Buliding the model

    # lower alpha, the model becomes sharper.
    NB_classifier=MultinomialNB(alpha=0.1)
    NB_classifier.fit(X_train,y_train)
    print(NB_classifier.score(X_train,y_train))
    print(NB_classifier.score(X_test,y_test))
    yc=NB_classifier.predict(X_test)
    print("confusion Matrix :",confusion_matrix(y_test,yc))
    print(classification_report(y_test, yc))

    joblib.dump(NB_classifier, "C:/Users\Mohannad/git repository/ML-Email-Classifier/models/NB_classifier_model.pkl")

    while(True) :
    # Get user input for email
        user_email = input("Enter the email text: ")

        # Predict whether the input email is spam or ham
        predict_email(user_email, NB_classifier,Vectorizer)


#Enter the email text: You've won a free gift card worth $100! Click to claim it.
# The email is predicted as spam.
# Enter the email text: Limited time offer: Buy one get one free! Don’t miss out on this amazing deal.
# The email is predicted as ham.
# Enter the email text: If you have any further questions, feel free to reach out. I’m here to help.
# The email is predicted as spam.
#

#default balanced
#confusion Matrix : [[1217    0]
#  [  59  117]]
#               precision    recall  f1-score   support

#          ham       0.95      1.00      0.98      1217
#         spam       1.00      0.66      0.80       176

#     accuracy                           0.96      1393
#    macro avg       0.98      0.83      0.89      1393
# weighted avg       0.96      0.96      0.95      1393


#Vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, ngram_range=(1,2))

#               precision    recall  f1-score   support

#          ham       0.96      0.98      0.97      1216
#         spam       0.98      0.96      0.97      1197

#     accuracy                           0.97      2413
#    macro avg       0.97      0.97      0.97      2413
# weighted avg       0.97      0.97      0.97      2413

##NB_classifier = MultinomialNB(alpha=0.5)
# confusion Matrix : [[1211   20]
#  [  34 1148]]
#               precision    recall  f1-score   support

#          ham       0.97      0.98      0.98      1231
#         spam       0.98      0.97      0.98      1182

#     accuracy                           0.98      2413
#    macro avg       0.98      0.98      0.98      2413
# weighted avg       0.98      0.98      0.98      2413

##NB_classifier = MultinomialNB(alpha=0.3)
# confusion Matrix : [[1186    6]
#  [   9 1212]]
#               precision    recall  f1-score   support

#          ham       0.99      0.99      0.99      1192
#         spam       1.00      0.99      0.99      1221

#     accuracy                           0.99      2413
#    macro avg       0.99      0.99      0.99      2413
# weighted avg       0.99      0.99      0.99      2413

#NB_classifier = MultinomialNB(alpha=0.1)
# confusion Matrix : [[1184   15]
#  [  13 1201]]
#               precision    recall  f1-score   support

#          ham       0.99      0.99      0.99      1199
#         spam       0.99      0.99      0.99      1214

#     accuracy                           0.99      2413
#    macro avg       0.99      0.99      0.99      2413
# weighted avg       0.99      0.99      0.99      2413