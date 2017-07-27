#########################################
# Importing and formatting the datasets #
#########################################

# Load the data
import pandas as pd
train_variants_df = pd.read_csv("training_variants")
test_variants_df = pd.read_csv("test_variants")
train_text_df = pd.read_csv("training_text", sep = "\|\|", engine = "python", skiprows = 1, names = ["ID", "Text"])
test_text_df = pd.read_csv("test_text", sep = "\|\|", engine = "python", skiprows = 1, names = ["ID", "Text"])

# Merge the info from the 2 data files
train_dataset = pd.merge(train_variants_df, train_text_df, on='ID')
test_dataset = pd.merge(test_variants_df, test_text_df, on='ID')

# Remove original variables to save memory
del train_variants_df
del train_text_df
del test_variants_df
del test_text_df

# Reformat by merging columns (adding gene and variant to text)
train_dataset['Text'] = train_dataset[['Gene', 'Variation', 'Text']].apply(lambda x: ' '.join(x), axis=1)
train_dataset = train_dataset.drop('ID', axis=1)
train_dataset = train_dataset.drop('Gene', axis=1)
train_dataset = train_dataset.drop('Variation', axis=1)
test_dataset['Text'] = test_dataset[['Gene', 'Variation', 'Text']].apply(lambda x: ' '.join(x), axis=1)
test_dataset = test_dataset.drop('ID', axis=1)
test_dataset = test_dataset.drop('Gene', axis=1)
test_dataset = test_dataset.drop('Variation', axis=1)



###############################
# Natural language processing #
###############################

# Cleaning the texts
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
train_corpus = []
for i in range(len(train_dataset)):
    print(i)
    info = re.sub('[^a-zA-Z0-9]', ' ', train_dataset['Text'][i])
    info = info.lower()
    info = info.split()
    ps = PorterStemmer()
    info = [ps.stem(word) for word in info if not word in set(stopwords.words('english'))]
    info = ' '.join(info)
    train_corpus.append(info)
test_corpus = []
for i in range(len(test_dataset)):
    print(i)
    info = re.sub('[^a-zA-Z0-9]', ' ', test_dataset['Text'][i])
    info = info.lower()
    info = info.split()
    ps = PorterStemmer()
    info = [ps.stem(word) for word in info if not word in set(stopwords.words('english'))]
    info = ' '.join(info)
    test_corpus.append(info)

# Create class labels for training set
y = train_dataset.iloc[:, 0].values

# Remove large dataframes to save memory
del test_dataset
del train_dataset
del i
del info

# Creating the term frequency - inverse document frequency model with tf-idf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(train_corpus).toarray()
X_test = cv.transform(test_corpus).toarray()

# Remove corpi to save memory
del test_corpus
del train_corpus



########################################
# Build and tune classification models #
########################################

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


## XGBoost ##

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred_prob = classifier.predict_proba(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Scoring model accuracy
print(classifier.score(x_test, y_test))

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 6)
print(accuracies.mean())
print(accuracies.std())







## Artificial Neural Network Model

# offsetting classes to be zero based
y = y - 1
y_train = y_train - 1
y_test = y_test - 1

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
X = sc.fit_transform(X)
X_test = sc.transform(X_test)

# Making the artificial neural network

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(activation="relu", input_dim=1000, units=300))
classifier.add(Dropout(rate = 0.05))

# Adding the second hidden layer
classifier.add(Dense(activation="sigmoid", units=300))
classifier.add(Dropout(rate = 0.05))

# Adding the output layer
classifier.add(Dense(activation="softmax", units=9))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train, batch_size = 300, epochs = 500)
classifier.fit(X, y, batch_size = 300, epochs = 100)

# Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = classifier.predict(X_test)

# Evaluating the model
# turn probabilities into class predictions
import numpy as np
classes = []
[classes.append(np.argmax(y_pred[item])) for item in range(len(x_test))]
classes = np.array(classes)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, classes)

# calculate accuracy
accuracy = np.trace(cm) / np.sum(cm)


