import os 
from sklearn import cross_validation
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score
# from bs4 import BeautifulSoup
import re

def review_to_wordlist( review, remove_stopwords=False ):
    # review_text = BeautifulSoup(review).get_text()
    review_text = review
    review_text = re.sub("[^а-яА-Яa-zA-Z]"," ", review_text)
    words = review_text.lower().split()
    return(words)


def Vector(train, remove_stopwords = False):
    print ("Cleaning and parsing tweets...\n") 
    traindata = [] 
    for i in range( 0, len(train["review"])):
        traindata.append(" ".join(review_to_wordlist(train["review"][i], remove_stopwords)))
    return traindata  

train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 't.csv'), header=0, delimiter="\t", quoting=3)  #открывается обучающий датасет

train = train[:5000]
y = train["sentiment"]
traindata = Vector(train, )    
'''
print ("count tf-idf... ")
X = TfIdf(traindata)

print ("test SVC model")
model = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', 
                  fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
model.fit(X, y)
#result = model.predict(X_test)#_proba(X_test)
print ("20 Fold CV Score. TF-IDF: ", np.mean(cross_validation.cross_val_score(model, X, y, cv=20, scoring='roc_auc')))

'''
print ("count bag of words")
count = CountVectorizer()
X = count.fit_transform(traindata)

print ("test SVC model")
model = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', 
                  fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
model.fit(X, y)
print ("20 Fold CV Score. Bag of words: ", np.mean(cross_validation.cross_val_score(model, X, y, cv=20, scoring='roc_auc')))
print ("20 Fold CV Score. Bag of words: ", np.mean(cross_val_score(model, X, y, cv=20, scoring='roc_auc')))
# '''
