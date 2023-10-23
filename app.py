from flask import Flask,request,jsonify, render_template
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression , ElasticNet , Lasso , Ridge

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
trans = pickle.load(open('transformer.pkl','rb'))


@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict',methods=['POST'])
def predict():
    age = request.form.get('Age')
    neck = request.form.get('Neck')
    knee = request.form.get('Knee')
    ankle = request.form.get('Ankle')
    biceps = request.form.get('Biceps')
    forearm = request.form.get('Forearm')
    wrist = request.form.get('Wrist')
    bmi = request.form.get('Bmi')
    acratio = request.form.get('ACratio')
    htratio = request.form.get('HTratio')


    input_query = np.array([[age,neck,knee,ankle,biceps,forearm,wrist,bmi,acratio,htratio]])

    input_query = trans.transform(input_query)

    result = model.predict(input_query)[0]
    fat = ((4.95/result) - 4.5) * 100
    response = {
        'Body Fat': str(fat),
        'Body Density': str(result)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)