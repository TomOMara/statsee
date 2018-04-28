from flask import Flask
from flask_cors import CORS
from flask_restful import Api

from DemoResourceAPI import DemoResourceAPI
from GraphParserAPI import GraphParserAPI
from GraphVerifierAPI import GraphVerifierAPI
from ExperimentResourceAPI import ExperimentResourceAPI

app = Flask(__name__)
api = Api(app)
CORS(app)

# Cross origin request allower
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Routes
api.add_resource(GraphParserAPI, '/')
api.add_resource(DemoResourceAPI, '/demo_webpage')
api.add_resource(ExperimentResourceAPI, '/experiment')
api.add_resource(GraphVerifierAPI, '/verify_graph')

if __name__ == '__main__':
    app.run(threaded=True, debug=True)
