from sklearn import datasets
from sklearn.model_selection import train_test_split

def get_data():
    return datasets.load_iris()

def holdout(iris_ds):
    X = iris_ds.data
    y = iris_ds.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    return X_train, X_test, y_train, y_test
