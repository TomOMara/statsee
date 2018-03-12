from flask import Flask
from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin


app = Flask(__name__)
api = Api(app)
CORS(app)


class HelloWorld(Resource):
    """
        start with python api.py
    """

    def get(self):
        return {'hello': 'world'}

    def post(self):
        return {'status': 200,
                'data': '<p>API RESPONSE</p>'
                }


api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(debug=True)