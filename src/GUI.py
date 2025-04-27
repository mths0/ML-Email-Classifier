import joblib
import os
from spam_email_classifier import predict_email
import ttkbootstrap as ttk
# from ttkbootstrap.constants import *

#load model
NB_classifier_model = joblib.load("C:/Users/Mohannad/git repository/ML-Email-Classifier/models/NB_classifier_model.pkl")
Vectorizer = joblib.load("C:/Users/Mohannad/git repository/ML-Email-Classifier/models/Vectorizer.pkl")



def print_result():
    #1.0 --> start at line 1, char 0
    #end-1c --> go until end minus 1 char
    user_input = text_box.get("1.0", "end-1c")
    result = predict_email(user_input,
                  NB_classifier_model,
                  Vectorizer,
                  print_state=False)
    email_string.set(result)
    
window = ttk.Window()
window.title("spam email detector")
window.geometry("500x650")

style = ttk.Style("superhero")

label = ttk.Label(window, text="Enter your Email", font=("Helvetica", 16))
label.pack(side="top", pady=10)

text_box = ttk.Text(window, font=("Helvetica", 12))
text_box.pack(pady=10)

email_string = ttk.StringVar()
result = ttk.Label(window, text="spam or ham" , font=("Helvetica", 16), textvariable= email_string)
result.pack(pady= 10)

b1 = ttk.Button(window, text="Check", bootstyle= "danger", style="TButton", width=20, command= print_result)
b1.pack(side="bottom", padx=5, pady=10)



window.mainloop()



