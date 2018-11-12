##import libraries
import sys
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Read file into pandas
# 3 files are yelp_labelled.txt, amazon_cells_labelled.txt, imdb_labelled
txt_file = 'amazon_cells_labelled.txt';
data = pd.read_csv(txt_file, sep="\t", names=['reviews', 'sentiment'])
X = data.reviews
y = data.sentiment

##get the test set
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size= 0.2)

##Training on unnormalized data set
def unNormalized_NBTrain():
    # Vectorise the dataset
    vect = CountVectorizer()
    #Using training data to transform text into counts of features for each message
    X_train_features = vect.fit_transform(X_train)
    ## Train the naive Bayes classifier
    NB = MultinomialNB()
    classifier=NB.fit(X_train_features, y_train)
    return classifier, vect

def unNormalized_NBAccuracy(filename, writeFile="NB-Unnormalized.txt"):
##    Test on a new document
    classifier, vect  = unNormalized_NBTrain()
    test_data = []
    with open(filename, 'r') as rf:
        for line in rf:
            test_data.append(line.strip('\r\n'))
##    Using training data to transform text into counts of features for each message        
    X_test_features = vect.transform(test_data)
##    sentence prediction
    predicted = classifier.predict(X_test_features)
    # writing results to a file
    with open(writeFile, 'w') as wf:
        for prediction in predicted:
            wf.write(str(prediction) + '\n')

    # calculate accuracy
    classifier, vect  = unNormalized_NBTrain()
    X_test_features = vect.transform(X_test)
    predicted = classifier.predict(X_test_features)
    accuracy = np.mean(predicted == y_test)
    print("Accuracy: Normalized NB: " + str(round(accuracy, 3)))


##Training on normalized data set
def normalized_NBTrain():
    #Vectorise the dataset
    #normalize using stopwords
    vect = CountVectorizer(stop_words = 'english',lowercase = True, ngram_range = (1,1), max_df = .80, min_df = 4)
    #Using training data to transform text into counts of features for each message
    X_train_features = vect.fit_transform(X_train)
    
    # Fit the estimator and transform the vector to tf-idf
    tf_transformer  = TfidfTransformer()
    X_train_tf  = tf_transformer.fit_transform(X_train_features)
    
    # Train the naive Bayes classifier
    classifier = MultinomialNB().fit(X_train_tf ,y_train)

    # return trained model
    return classifier, vect, tf_transformer


def normalized_NBAccuracy(filename, writeFile="NB-Normalized.txt"):
    ##    Test on a file
    classifier, vect, tf_transformer  = normalized_NBTrain()
    test_data = []
    with open(filename, 'r') as rf:
        for line in rf:
            test_data.append(line.strip('\r\n'))
##    transform the vector
    X_test_features = vect.transform(test_data) 
    X_test_tf = tf_transformer.transform(X_test_features)
    predicted = classifier.predict(X_test_tf)

    # write to file
    with open(writeFile, 'w') as wf:
        for prediction in predicted:
            wf.write(str(prediction) + '\n')
            
    # calculate accuracy
    classifier, vect, tfidf_transformer  = normalized_NBTrain()
    X_test_features = vect.transform(X_test)
    predicted = classifier.predict(X_test_features)
    accuracy = np.mean(predicted == y_test)
    print("Accuracy: Un-Normalized NB: " + str(round(accuracy, 3)))

# Un-normalized Logistic Regression 
##Training an unnormalized logistic regression model
def unNormalized_LRTrain():
    ##initialize CountVectorizer
    ##transform training dataset to word features
    vect = CountVectorizer()
    X_train_features = vect.fit_transform(X_train)
    # train model with training dataset
    LR = LogisticRegression(solver='lbfgs').fit(X_train_features, y_train)
    return LR, vect

##Calculate LR (unnormalized data) accuracy 
def unNormalized_LRAccuracy(filename, writeFile="LR-Unnormalized.txt"):
    LR, vect  = unNormalized_LRTrain()
    test_data = []
    with open(filename, 'r') as rf:
        for line in rf:
            test_data.append(line.strip('\r\n'))
    # transform test_data to features
    X_test_features = vect.transform(test_data) 
    predicted = LR.predict(X_test_features)
    
    # write prediction to file
    with open(writeFile, 'w') as wf:
        for prediction in predicted:
            wf.write(str(prediction) + '\n')

    LR, vect  = unNormalized_LRTrain()
    # transform test_data to features
    X_test_features = vect.transform(X_test)
    predicted = LR.predict(X_test_features)
    accuracy = np.mean(predicted == y_test)
    print("Accuracy: Normalized LR: " + str(round(accuracy, 3)))



##  Normalized Logistic Regression  
def normalized_LRTrain():
    #convert text into features
    vect = CountVectorizer(stop_words='english',lowercase = True, ngram_range = (1,1), max_df = .80, min_df = 4)
    # transform training dataset to word features 
    X_train_features =vect.fit_transform(X_train)
    # solve frequency discrepancies among long and short sentences
    tf_transformer = TfidfTransformer()
    X_train_tf = tf_transformer.fit_transform(X_train_features)
    
    LR = LogisticRegression(solver='lbfgs').fit(X_train_tf, y_train)
    return LR, vect, tf_transformer

def normalized_LRAccuracy(filename, writeFile="LR-Normalized.txt"):
    LR, vect, tf_transformer  = normalized_LRTrain()
    # load data for testing
    test_data = []
    with open(filename, 'r') as rf:
        for line in rf:
            test_data.append(line.strip('\r\n'))

    X_test_features = vect.transform(test_data) 
    X_test_tf = tf_transformer.transform(X_test_features)

    # predict test sentences
    predicted = LR.predict(X_test_tf)
    # write to file
    with open(writeFile, 'w') as wf:
        for prediction in predicted:
            wf.write(str(prediction) + '\n')

##    calculate accuracy of the normalized LR model
    LR, vect, tf_transformer = normalized_LRTrain()
    X_test_features = vect.transform(X_test)
    predicted = LR.predict(X_test_features)
    accuracy = np.mean(predicted == y_test)
    # print accuracy
    print("Accuracy: Un-Normalized LR: " + str(round(accuracy, 3)))

if __name__ == '__main__':
    
    unNormalized_NBAccuracy('amazon_cells_labelled.txt')
    
    normalized_NBAccuracy('amazon_cells_labelled.txt')
    
    unNormalized_LRAccuracy('amazon_cells_labelled.txt')

    normalized_LRAccuracy('amazon_cells_labelled.txt')

    



