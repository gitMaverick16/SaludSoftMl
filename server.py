import joblib
import numpy as np
import pandas as pd
import sklearn

from flask import Flask, render_template, request
from flask import jsonify

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import (
    cross_val_score, KFold
)


import matplotlib.pyplot as plt 


from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
 
from sklearn.linear_model import LogisticRegression
 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

#POSTMAN PARA PRUEBAS
@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
    
        # Cargamos los datos del dataframe de pandas
        dt_diabetes = pd.read_csv('data/diabetes.csv')
            
        # Guardamos nuestro dataset sin la columna de target
        dt_features = dt_diabetes.drop(['class'], axis=1)
        # Este será nuestro dataset, pero sin la columna
        dt_target = dt_diabetes['class']

        dt_features = StandardScaler().fit_transform(dt_features)

        x_train, x_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3)
        """ x_asd=[ [61,1,0,0,0,1,0,1,0,1,0,1,0,0,1,0],
                [60,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1],
                [58,1,0,0,0,1,0,0,0,1,0,1,0,1,1,0],
                [54,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [67,1,0,0,0,1,0,0,0,1,0,1,0,0,1,0],
                [66,1,0,0,0,1,1,0,1,1,0,1,1,1,1,0],
                [43,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                [62,0,1,1,1,1,0,0,1,0,0,0,1,0,0,1],
                [54,0,1,1,1,1,1,0,0,0,0,0,1,0,0,0],
                [39,0,1,1,1,0,1,0,0,1,0,1,1,0,0,0],
                [48,0,1,1,1,1,1,0,0,1,1,1,1,0,0,0],
                [58,0,1,1,1,1,1,0,1,0,0,0,1,1,0,1],
                [32,0,0,0,0,1,0,0,1,1,0,1,0,0,1,0],
                [42,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]] """
        
       
        user_age = float(request.form.get('age'))
        user_gender = float(request.form.get('gender'))
        user_polyuria = float(request.form.get('polyura'))
        user_polydipsia = float(request.form.get('polydipsia'))
        user_weight = float(request.form.get('weight'))
        user_weakness = float(request.form.get('weakness'))
        user_polyphagia = float(request.form.get('polyphagia'))
        user_genital = float(request.form.get('genital'))
        user_visual = float(request.form.get('visual'))
        user_itching = float(request.form.get('itching'))
        user_irritability = float(request.form.get('irritability'))
        user_delayed = float(request.form.get('delayed'))
        user_partial = float(request.form.get('partial'))
        user_muscle = float(request.form.get('muscle'))
        user_alopecia = float(request.form.get('alopecia'))
        user_obesity = float(request.form.get('obesity'))
        
        x_asd=[[user_age, user_gender, user_polyuria, user_polydipsia, user_weight, user_weakness,
        user_polyphagia, user_genital, user_visual, user_itching, user_irritability, user_delayed,
        user_partial, user_muscle, user_alopecia, user_obesity]]
        algoritmo = DecisionTreeClassifier()
        score = cross_val_score(algoritmo, dt_target, dt_features, scoring='neg_mean_squared_error')
        
        algoritmo.fit(x_train, y_train)
        resultado = algoritmo.predict(x_asd)

        
    #return jsonify({'prediccion' : str(resultado)})
    return render_template('index.html', resultado=resultado)

if __name__ == "__main__":
    #model = joblib.load('./models/best_model.pkl')
    app.run(port=8080)
    