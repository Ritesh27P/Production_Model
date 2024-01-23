import pickle
from symptomsConverter import SymptomsConverter
from flask import Flask, render_template, request, url_for

app = Flask(__name__)

# Models
sym_model = pickle.load(open('Symptom_Model1.sav', 'rb'))
heart_model = pickle.load(open('Health_Model1.sav', 'rb'))

@app.route('/')
def home():
    return "Hello World!"

@app.route('/disease', methods=['POST'])
def disease():
    inputs = request.json['inputs']
    inputs = SymptomsConverter().converts(inputs)
    pred = sym_model.predict([inputs])
    return {
        'disease': pred[0]
    }

@app.route('/heart', methods=["POST"])
def heart():
    inputs = request.json['inputs']
    # inputs = [float(i) for i in inputs]
    pred = heart_model.predict([inputs])
    return {
        'value': pred[0]
    }

if __name__ == '__main__':
    app.run(debug=True)