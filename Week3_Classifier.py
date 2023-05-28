def clean(X, y):
    # get external support
    import pandas as pd
    import numpy as np

   # """
    #TODO:
    #Part 0, Step 2: 
     #   - Use the pandas {isna} and {dropna} functions to remove from the dataset any corrupted samples
    #"""
    #TODO:
    #Part 0, Step 2: 
     #   - Use the pandas {isna} and {dropna} functions to remove from the dataset any corrupted samples
    b = X.isna()
    b = 1-b
    b_sum = np.sum(b, axis=1)
    b_sum_boolean = b_sum > 879
    #print("b sum =", b_sum)
    #print("b_sum_boolean =", b_sum_boolean)
    y = np.array([y])
    print("y initial shape =",np.shape(y))
    y = np.reshape(y, (len(b_sum), 1))
    y = y[b_sum_boolean]
    print("y final shape =", np.shape(y))
    print("X initial shape=", np.shape(X))
    X = X.dropna(axis="rows")
    print("X final shape=", np.shape(X))
    # return the cleaned data

    return [X, y]


def train_test_validation_split(X, y, test_size, cv_size):
    # get external support
    
    # return split data
    import sklearn
    from sklearn.model_selection import train_test_split

    # Split the data into a training set and a temporary set
    X_train, X_test_cv, y_train, y_test_cv = train_test_split(X, y, test_size=test_size+cv_size, random_state=0, shuffle = "True")

    # Split the temporary set into a training set and a validation set
    X_test, X_cv, y_test, y_cv = train_test_split(X_test_cv, y_test_cv, test_size=cv_size/(cv_size+test_size), random_state=0, shuffle = "True")

    # return split data
    return [X_train, y_train, X_test, y_test, X_cv, y_cv]


def scale(X_train, X_test, X_cv):
    # get external support
    import sklearn
    from sklearn import preprocessing

   # """
    #TODO:
    #Part 0, Step 4: 
      #  - Use the {preprocessing.StandardScaler} of sklearn to normalize the data
       # - Scale the train, test and cross-validation sets accordingly
    #"""
    scaler = sklearn.preprocessing.StandardScaler()

    # Fit the scaler on the training data
    scaler.fit(X_train)

    # Transform the training, testing, and cross-validation data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_cv_scaled = scaler.transform(X_cv)
    # return the normalized data and the scaler
    return [X_train, X_test, X_cv, scaler]


def clean_split_scale(X, y):
    # clean data (remove NaN data points)
    [X, y] = clean(X, y)

    # split data into 90% train, 10% test, 10% cross validation
    [X_train, y_train, X_test, y_test, X_cv, y_cv] = train_test_validation_split(
        X, y, test_size=0.1, cv_size=0.1)

    # convert data and labels to numpy arrays, ravel labels
    X_train = X_train
    X_test = X_test
    X_cv = X_cv
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    y_cv = y_cv.ravel()

    # scale the data
    [X_train, X_test, X_cv, scaler] = scale(X_train, X_test, X_cv)

    # return cleaned, scaled and split data and the scaler
    return [X_train, y_train, X_test, y_test, X_cv, y_cv, scaler]

# get external support
from sklearn import svm

def train_binary_svm_classifier(X_train, y_train, C, gamma):
    """
    Train a binary Support Vector Machine classifier with sk-learn
    """

    """
    TODO:
    Part 1: 
        - Use the sklearn {svm.SVC} class to implement a binary classifier 
    """
    from sklearn import svm

    
    # Create a Support Vector Machine classifier object
    clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)

    # Train the classifier on the training data
    clf.fit(X_train, y_train)

    # Return the trained classifier object
    return clf


# get external support
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier


def train_multi_class_svm_classifier(X_train, y_train, C, gamma):
    """
    Train a multi-class Support Vector Machine classifier with sk-learn
    """

    """
    TODO:
    Part 2: 
        - Use the sklearn {OneVsRestClassifier} class to implement a multi-class classifier 
    """
    

    # Create a Support Vector Machine classifier object
    clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)

    # Wrap the classifier object with the OneVsRestClassifier class
    multi_clf = OneVsRestClassifier(clf.fit(X_train, y_train))

    # Train the multi-class classifier on the training data
    multi_clf.fit(X_train, y_train)

    # Return the trained multi-class classifier objec
    return clf