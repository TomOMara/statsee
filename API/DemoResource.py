from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin
from flask import Response, render_template

class DemoResource(Resource):

    def get(self):
        """
        Example Get request
        :return:
        """
        return Response(render_template('demo_page.html'), mimetype='text/html')

