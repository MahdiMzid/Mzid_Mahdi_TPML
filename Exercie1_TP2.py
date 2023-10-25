# 1 importer le jeu de données
from sklearn.datasets import load_iris
d_iris = load_iris()

# 2 Classe des fleurs
print("Classe des iris")
print(list(d_iris.target_names))

# 3 characteristiques des fleurs
print("Les charactéristiques des iris")
print(list(d_iris.feature_names))

# 4 Description du jeu de données
print("Description du jeu de données")
print(d_iris.DESCR)

# 5 Dimension du vecteur label
print("Dimension du vecteur label")
print(d_iris.target.shape)

# 6 Dimension de la matrice des features
print("Dimension de la matrice des features")
print(d_iris.data.shape)

# 7 Histogramme de la répartition des largeurs de pétales
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.hist(d_iris.data[:, 3], bins=30, color='red', alpha=0.7, edgecolor="black")
plt.xlabel("Largeur de pétales")
plt.ylabel("Nombre d'exemples")
plt.title("Histogramme  de largeur des pétales")
plt.show()

# 8 Distribution de probabilité des largeurs de pétales
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.kdeplot(d_iris.data[:, 3], color='green', alpha=0.7, shade=True)
plt.xlabel("Largeur de pétales")
plt.ylabel("Distribution de Probabilité ")
plt.title("Distribution de Probabilité  de largeur des pétales")
plt.show()

# 9
import numpy as np
species_countes = np.bincount(d_iris.target)
plt.bar(d_iris.target_names, species_countes)
plt.xlabel("Espèce")
plt.ylabel("Nombre d'exemples ")
plt.title("Répartition des exemples par espèces")
plt.show()
