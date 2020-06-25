import os 
import joblib
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from modeler.Modeler import Modeler

app = Flask(__name__)
api = Api(app)


class Predict(Resource):
    @staticmethod
    def post():
        data = request.get_json()
        sepal_length = data['sepal_length']
        sepal_width = data['sepal_width']
        petal_length = data['petal_length']
        petal_width = data['petal_width']

        m = Modeler()
        if not os.path.isfile('models/iris.model'):
            m.fit()
        prediction = m.predict([sepal_length, sepal_width, petal_length, petal_width])
        return jsonify({
            'Input': {
                'SepalLength': sepal_length,
                'SepalWidth': sepal_width,
                'PetalLength': petal_length,
                'PetalWidth': petal_width
            },
            'Class': prediction
        })

api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(debug=True)