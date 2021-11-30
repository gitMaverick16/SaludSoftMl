import pandas as pd
import sklearn
import numpy as np
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
    
if __name__ == "__main__":
    # Cargamos los datos del dataframe de pandas
    dt_diabetes = pd.read_csv('data/diabetes.csv')
        
    # Guardamos nuestro dataset sin la columna de target
    dt_features = dt_diabetes.drop(['class'], axis=1)
    # Este será nuestro dataset, pero sin la columna
    dt_target = dt_diabetes['class']

    dt_features = StandardScaler().fit_transform(dt_features)

    x_train, x_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3)
    x_asd=[ [61,1,0,0,0,1,0,1,0,1,0,1,0,0,1,0],
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
            [42,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    algoritmo = DecisionTreeClassifier()
    score = cross_val_score(algoritmo, dt_target, dt_features, scoring='neg_mean_squared_error')
    #print(np.abs(np.mean(score)))


    algoritmo.fit(x_train, y_train)
    print(algoritmo.predict(x_asd))

    """ print("imprimento x")
    print(x_test)
    print("imprimento y")
    print(y_test) """