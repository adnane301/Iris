import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Iris.mdl', 'rb'))
ohe = pickle.load(open('PowerTransformer.encoder','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    
    compatibleFeatureSet = np.array([[sepal_length,sepal_width,petal_length,petal_width ]])
    class_number=model.predict(compatibleFeatureSet)[0]
      

    return render_template('index.html', class_text='Class is {}'.format(round(class_number[0][0])))


if __name__ == "__main__":
    app.run(debug=True)