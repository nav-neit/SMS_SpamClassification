import pandas as pd

dataset = pd.read_csv('smsspamcollection/SMSSpamCollection', delimiter='\t', names = ['message_label','message'])

# cleaning and preprocessing

messages = dataset['message'].values

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re

wordnet = WordNetLemmatizer()
ps = PorterStemmer()
corpus = []
for i in range(len(messages)):
    message = re.sub('[^a-zA-Z]',' ',messages[i])
    message = message.lower()
    words = message.split()
    words = [wordnet.lemmatize(word) for word in words if not word in set(stopwords.words('english'))]
    message = ' '.join(words)
    corpus.append(message)
    
# creating the bag of words model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2500)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(dataset['message_label'],drop_first = True).values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)    

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
acc = accuracy_score(y_test,y_pred)
print('Accuracy of model = {:.2f} %'.format(acc*100))
