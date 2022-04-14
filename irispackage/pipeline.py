from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class IrisFeatures():
    
    def __init__(self):
        pass
    
    def get_pipeline(self):
        return Pipeline(steps=[('scaler', StandardScaler()),
                               ('regressor', LinearRegression())])
        
# class TaxiFeatures():
#     # feat engineering
    
#     def __init__(self, use_time=True, use_distance=True):
#         self.use_time = use_time
#         self.use_distance = use_distance
    
#     def get_pipeline(self):
#         pipe_time = make_pipeline(TimeFeatures(time_column='pickup_datetime'), StandardScaler())
#         pipe_distance = make_pipeline(DistanceTransformer(), RobustScaler())

#         # column transformer
#         dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
#         time_cols = ['pickup_datetime']

#         use_parameter=[]
#         if self.use_time == True:
#             use_parameter.append(('time', pipe_time, time_cols))
#         if self.use_distance == True:
#             use_parameter.append(('distance', pipe_distance, dist_cols))
            
#         feat_eng_bloc = ColumnTransformer(use_parameter) # remainder='passthrough'

#         # workflow
#         pipe_cols = Pipeline(steps=[('feat_eng_bloc', feat_eng_bloc),
#                                     ('regressor', RandomForestRegressor())])

#         pipe_cols.fit(X_train, y_train)