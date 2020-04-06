import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import naive_bayes

data = pd.read_csv("imdb2k.csv")

X = data['text']
y = data['label']

vectorizer = CountVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X).toarray()

classifier = naive_bayes.MultinomialNB()
classifier.fit(X,y)

prediction = classifier.predict(vectorizer.transform([input('enter input: ')]).toarray())
print('positive' if prediction[0] == 1 else 'negative')