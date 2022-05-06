import pickle
import pandas as pd
import numpy as np



def test_model(model,features):
    categoric_vars = ['Sex', 'AgeCategory', 'Race', 'Diabetic', 'GenHealth']
    target_var = ['HeartDisease']
    numeric_vars = ['MentalHealth', 'BMI', 'PhysicalHealth', 'SleepTime']
    yes_no_vars =  ["Smoking","AlcoholDrinking","Stroke","DiffWalking","PhysicalActivity","Asthma","KidneyDisease","SkinCancer"]
    features_col = numeric_vars+yes_no_vars+categoric_vars
    df = pd.DataFrame([features] , columns=features_col)
    print(df)
    pred = model.predict(df)
    print(pred)
    return 


if __name__ == '__main__':

     # Variables Globales
    MODEL_PATH = "models/xgb_v1_deploy.pkl"

    #Carga del modelo
    model = pickle.load(open(MODEL_PATH, "rb"))
     
    #Nos inventamos los datos 
    features=np.array([20., 20., 20., 8. , 2,  2,  2,  1 , 2, 2 , 1 , 2 , 1 , 3 , 1,  2,  1])

    test_model(model,features)