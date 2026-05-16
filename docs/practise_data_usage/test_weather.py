from flask import Flask
# from flask.re
import requests


app = Flask(__name__)

@app.route("/")
def hello_world():
    data = requests.get('https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m').json()

    return data.get('current')#.get('temperature_2m')

    # return "<p>Hello, World!</p>"

