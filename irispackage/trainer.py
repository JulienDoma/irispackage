import joblib
from irispackage.data import get_data, holdout
from irispackage.pipeline import IrisFeatures

class Trainer():
    
    def __init__(self):
        pass
        
    def train(self):
        
        X_train, X_test, y_train, y_test = holdout(get_data())
        
        iris_pipeline = IrisFeatures()
        pipeline = iris_pipeline.get_pipeline()
        
        pipeline.fit(X_train, y_train)
        
        joblib.dump(pipeline, 'model.joblib')