
import pandas as pd
import numpy as np

df = pd.read_csv("sample.csv")
df = df.dropna()
df.reset_index(inplace = True) 
df = df.drop(['index'], axis = 1) 

#prepare the data to train the model
features = df.iloc[:,:-1].values 
labels = df.iloc[:, 1].values 



'''******************* linear regression***************************'''
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
features[:, 0] = labelencoder.fit_transform(features[:, 0])


from sklearn.model_selection import train_test_split  
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=0)  

from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(features_train, labels_train)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20, random_state=0)  
classifier.fit(features_train, labels_train) 

from sklearn.svm import SVC # kernels: linear, poly and rbf
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(features_train, labels_train)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, p = 2) #When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2
classifier.fit(features_train, labels_train)


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(features_train, labels_train)

from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.datasets import make_classification
#X, y = make_classification(n_features=4, random_state=0)
classifier = ExtraTreesClassifier(n_estimators=100, random_state=0)
classifier.fit(features_train, labels_train)

from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier(random_state=0)
classifier.fit(features_train, labels_train)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0).fit(features_train, labels_train)

from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(features_train, labels_train)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(features_train, labels_train)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(features_train, labels_train)


''' **************************************************************************'''
#import keras
#from keras.models import Sequential
#from keras.layers import Dense
#
#classifier = Sequential()
#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 1))
#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#classifier.fit(features_train, labels_train, batch_size = 10, epochs = 10)



labels_pred = classifier.predict(features_test) 

compare = pd.DataFrame({'Actual': labels_test, 'Predicted': labels_pred})  
print (compare )

labels_test = labels_test.astype(int) #convert into integer value
labels_pred = labels_pred.astype(int) #convert into integer value
from sklearn.metrics import accuracy_score
accuracy_score(labels_test,labels_pred) 