#LA PREMIERE PARTIE :
#Importation du jeu de données California Housing :Q1
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
CH = fetch_california_housing()
#Affichage de la description de jeu de données California Housing :Q2
print("la description de jeu de données est",CH.DESCR)
#Affichage du contenu de la matrice de données(Data) :Q3
M = CH.data
print("la matrice des DATA est :\n", M)
#Affichage du contenu du vecteur Target :Q4
Y = CH.target
print("le vecteur Target est :", Y)
#Affichage des dimensions de la matrice de données :Q5
print("les dimensions de la matrice DATA sont :", M.shape)
#Affichage les dimensions du vecteur Target :Q6
print("les dimensions du vecteur Target sont :", Y.shape)
#Transformation du California Housing en DataFrame :Q7
DF = pd.DataFrame(CH.data, columns=CH.feature_names)
print("la transformation est :\n", DF)
#LA DEUXIEME PARTIE :
#Verification que le jeu de données ne contient pas des NaN :Q8
VN= DF.isnull().any().any()
print("le jeu de données contient des valeurs nulles (NaN):",VN)
#Division de jeu de données: 75% app,25% test :Q9
X_train, X_test, Y_train, Y_test = train_test_split( DF, Y, test_size=0.25, random_state=42)
print("la taille de l'ensemble d'apprentissage (M_train, Y_train) est :", X_train.shape, Y_train.shape)
print("la taille de l'ensemble test (x_test, Y_test) est :", X_test.shape, Y_test.shape)