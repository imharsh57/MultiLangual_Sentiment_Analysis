import pandas as pd


dataset = pd.read_csv('final_train2.csv')
dataset = dataset.dropna()
dataset.reset_index(inplace = True) 
dataset = dataset.drop(['index'], axis = 1) 

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
#from nltk.stem.wordnet import WordNetLemmatizer 


corpus = []
count=0
#now do the same for every row in dataset. run to loop for all rows

for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['comment_text'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    
    #lem = WordNetLemmatizer() #Another way of finding root word
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    #review = [lem.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    print("****************************************",count)
    count = count+1
    
"""     Adding corpus to csv 
backup = corpus
#corpus_dataset = pd.DataFrame(corpus)
#corpus_dataset['corpus'] = corpus_dataset
#corpus_dataset = corpus_dataset.drop([0], axis = 1) 
corpus_dataset.to_csv('corpus_dataset.csv')
"""    
    
# Creating the Bag of Words model
# Also known as the vector space model
# Text to Features (Feature Engineering on text data)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
features = cv.fit_transform(corpus).toarray()
#features = corpus

labels = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.20, random_state = 0)


"""
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(features_train, labels_train)

# Predicting the Test set results
labels_pred = classifier.predict(features_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_nb = confusion_matrix(labels_test, labels_pred)
"""


#applying knn on this text dataset
# Fitting Knn to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 25, p = 2)
classifier.fit(features_train, labels_train)

# Predicting the Test set results
labels_pred = classifier.predict(features_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(labels_test, labels_pred)

compare = pd.DataFrame({'Actual': labels_test, 'Predicted': labels_pred})  
print (compare )

labels_test = labels_test.astype(int) #convert into integer value
labels_pred = labels_pred.astype(int) #convert into integer value
from sklearn.metrics import accuracy_score
accuracy_score(labels_test,labels_pred) 


