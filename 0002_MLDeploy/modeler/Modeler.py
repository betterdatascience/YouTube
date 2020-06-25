import os 
import joblib 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier


class Modeler:
    def __init__(self):
        self.df = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')
        try: self.model = joblib.load('models/iris.model')
        except: self.model = None 

    def fit(self):
        X = self.df.drop('species', axis=1)
        y = self.df['species']
        self.model = DecisionTreeClassifier().fit(X, y)
        joblib.dump(self.model, 'models/iris.model')

    def predict(self, measurement):
        if not os.path.exists('models/iris.model'):
            raise Exception('Model not trained yet. Call .fit() before making predictions')
        if len(measurement) != 4:
            raise Exception(f'Expected sepal_length, sepal_width, petal_length, petal_width, but got {measurement}')
        prediction = self.model.predict([measurement])
        return prediction[0]