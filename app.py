from flask import Flask, render_template, request,flash
import pickle
import pandas as pd
import numpy as np
import os



# Variables Globales
MODEL_PATH = "models/xgb_v1_deploy.pkl"


#Carga del modelo
model = pickle.load(open(MODEL_PATH, "rb"))

#Inicializamos aplicación flask
app = Flask(__name__)

#Configuraciones para entrada de datos al modelo
categoric_vars = ['Sex', 'AgeCategory', 'Race', 'Diabetic', 'GenHealth']
target_var = ['HeartDisease']
numeric_vars = ['MentalHealth', 'BMI', 'PhysicalHealth', 'SleepTime']
yes_no_vars =  ["Smoking","AlcoholDrinking","Stroke","DiffWalking","PhysicalActivity","Asthma","KidneyDisease","SkinCancer"]
features_col = numeric_vars+yes_no_vars+categoric_vars


@app.route('/',methods = ['GET'])
def home():
    return render_template('index.html')



@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        #Leemos Los campos de HTML cuando se ejecute la acción ----> action="{{url_for('predict')}}
        data1=float(request.form['a'])           # estado Mental MentalHealth
        data2=float(request.form['b'])           # BMI
        data3=float(request.form['c'])           # PhysicalHealth
        data4=float(request.form['d'])           # SleepTime
        data5=float(request.form['e'])           # Smoking
        data6=float(request.form['f'])           # AlcoholDrinking
        data7=float(request.form['g'])           # Stroke
        data8=float(request.form['h'])           # DiffWalking	
        data9=float(request.form['i'])           # PhysicalActivity
        data10=float(request.form['j'])          # Asthma 
        data11=float(request.form['k'])          # KidneyDisease
        data12=float(request.form['l'])          # SkinCancer
        data13=float(request.form['m'])          # Sexo
        data14=float(request.form['n'])          # Age
        data15=float(request.form['o'])          # Race
        data16=float(request.form['p'])          # Diabetic
        data17=float(request.form['q'])          # GenHEalth
        
        features=np.array([data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15,data16,data17])

        df = pd.DataFrame([features] , columns=features_col)
        print(features)
        pred = model.predict(df)
        print(pred)
        
        def statement():
            #Escribimos un mensaje para la predicción
            if pred == 0:
            
                return 'El modelo estima que con una probabilidad elevada no sufriras ninguna enfermedad cardíaca.'
            elif pred == 1:
            
                return 'Según el modelo existe una probabilidad elevada de que sufras una enfermedad de corazón a lo largo de tu vida. Consulta con tu médico'
    
        return render_template('index.html', statement= statement())
    else:
       return render_template('index.html')

#Lanzamos el principal de la app
if __name__=='__main__':
    app.run(debug=True)


