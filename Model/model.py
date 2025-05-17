import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import neattext.functions as nfx #to clean and proccess text
from sklearn.model_selection import train_test_split #to split data into train and test set
#for training:
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import joblib #for saving the model

#LOADING THE DATASET
data_frame = pd.read_csv("Data/emotion_dataset.csv")
print(data_frame.columns)
print(data_frame.head())
print(data_frame['Emotion'].value_counts())

sns.countplot(x='Emotion',data=data_frame)
plt.show()


#PRE-PROCESSING THE DATA
data_frame['Clean_Text'] = data_frame['Text'].apply(nfx.remove_userhandles) #remove the user handles
data_frame['Clean_Text'] = data_frame['Clean_Text'].apply(nfx.remove_stopwords)
print(data_frame)

#SPLITTING THE DATA INTO INPUT VARIABLES AND TARGET VARIABLE
x = data_frame['Clean_Text']
y = data_frame['Emotion']

#SPLITTING DATA INTO TRAIN AND TEST SET
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

pipe_lr = Pipeline(steps=[('cv', CountVectorizer()), ('lr', LogisticRegression(max_iter=1000))])
pipe_lr.fit(x_train, y_train)
pipe_lr.score(x_test, y_test)

pipe_svm = Pipeline(steps=[('cv', CountVectorizer()), ('svc', SVC(kernel='rbf', C=10))])
pipe_svm.fit(x_train, y_train)
pipe_svm.score(x_test, y_test)

pipe_rf = Pipeline(steps=[('cv', CountVectorizer()), ('rf', RandomForestClassifier(n_estimators=10))])
pipe_rf.fit(x_train, y_train)
pipe_rf.score(x_test, y_test)

print("Training Logistic Regression...")
pipe_lr.fit(x_train, y_train)
print("Logistic Regression done.")

print("Training SVM...")
pipe_svm.fit(x_train, y_train)
print("SVM done.")

print("Training Random Forest...")
pipe_rf.fit(x_train, y_train)
print("Random Forest done.")

print("Saving model...")
joblib.dump(pipe_lr, 'Data/text_emotion.pkl')
print("Model saved.")