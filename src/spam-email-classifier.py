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

#Reading the dataset from csv file.
Dataset=pd.read_csv('messages.csv')

#ham,spam in percentage
ham=Dataset[Dataset['label']==0]
spam=Dataset[Dataset['label']==1]
print('Spam Prercentage =',(len(spam)/len(Dataset['label']))*100,'%')
print('ham Prercentage =',(len(ham)/len(Dataset['label']))*100,'%')


#Preprocessing data
Dataset.drop(columns='subject',inplace=True)
test=Dataset.head(5)
BeforeDS=Dataset.head(5)
Dataset['message']=Dataset['message'].apply(word_tokenize)
Dataset['message']=Dataset['message'].apply(lambda x: [re.sub(r'[^a-zA-Z0-9\s]', '', word) for word in x])
stop_words = set(stopwords.words('english'))
Dataset['message'] = Dataset['message'].apply(lambda x: [word for word in x if word not in stop_words and word not in string.punctuation])
ps = PorterStemmer()
Dataset['message'] = Dataset['message'].apply(lambda x: [ps.stem(word) for word in x]) 
Dataset['message'] = Dataset['message'].apply(lambda x: ' '.join(x))



#print(BeforeDS)
#print(Dataset.head(5))

#Split the Dataset in to traning and testing.
Vectorizer= TfidfVectorizer()
x=Vectorizer.fit_transform(Dataset['message'])
print(x.shape)
y=Dataset['label']  
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25)

#Buliding the model
NB_classifier=MultinomialNB()
NB_classifier.fit(X_train,y_train)
print(NB_classifier.score(X_train,y_train))
print(NB_classifier.score(X_test,y_test))
yc=NB_classifier.predict(X_test)
print("confusion Matrix :",confusion_matrix(y_test,yc))


#for testing.
def predict_email(email):
    # Convert email into numerical vector using the trained TF-IDF vectorizer
    email_vector = Vectorizer.transform([email])
    
    # Convert sparse matrix to dense array
    email_vector_dense = email_vector.toarray()
    
    # Use the trained SVM model to make predictions
    prediction = NB_classifier.predict(email_vector_dense)
    
    # Print the prediction
    if prediction[0] == 1:
        print("The email is predicted as spam.")
    else:
        print("The email is predicted as ham.")


while(True) :
  # Get user input for email
  user_email = input("Enter the email text: ")

  # Predict whether the input email is spam or ham
  predict_email(user_email)