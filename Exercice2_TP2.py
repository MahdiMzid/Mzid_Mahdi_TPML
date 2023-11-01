# 1.Importer le jeu de données
import seaborn as sns
titanic = sns.load_dataset("titanic")
print(titanic.head())
print('Default Import')

# 2.Sélectionner certains caractéristiques
selected_columns = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'deck']
titanic_selected = titanic[selected_columns]
print(titanic_selected.head())
print('Data game with specific characteristics')

# 3.Suppression des lignes contenant des champs NaN
titanic_cleaned = titanic_selected.dropna()
print(titanic_cleaned.head())
print('Data game with NaN entries')

# 4.Répartition du nombre d'exemples hommes et femmes en un graphe
import  matplotlib.pyplot as plt
nbr_passagers = titanic_cleaned['sex'].value_counts()
plt.figure()
plt.bar(nbr_passagers.index, nbr_passagers.values, color=['blue', 'red'])
plt.xlabel('sexe')
plt.ylabel('Nombre d''exemples')
plt.title('Répartition du nombre d''exemples hommes et femmes')
print('Showing Male and Female Distribution')
plt.show()


# 5.Répartition du nombre d'exemples hommes et femmes en fonction de pclass en un graphe
nbr_passagers_par_classe = titanic_cleaned.groupby(['pclass','sex']).size().unstack()
nbr_passagers_par_classe.plot(kind = 'bar', stacked =True, color=['blue', 'red'])
plt.xlabel('Classe')
plt.ylabel('Nombre d''exemples')
plt.title('Répartition du nombre d''exemples hommes et femmes par classe')
plt.xticks(rotation=0)
print('Showing Male and Female Distribution in fucntion of class')
plt.show()


# 6.Répartition du nombre d'exemples hommes et femmes en fonction de survived
nbr_passagers_par_survie = titanic_cleaned.groupby(['survived', 'sex']).size().unstack()
nbr_passagers_par_survie.plot(kind = 'bar', color=['blue', 'red'])
plt.xlabel('Survie')
plt.ylabel('Nombre d''exemples')
plt.title('Répartition du nombre d''exemples hommes et femmes par survie')
plt.xticks([0, 1], ['Non Survie', 'Survie'])
plt.xticks(rotation=0)
print('Showing Male and Female Distribution in fucntion of survived')
plt.show()


# 6.Relation entre pclass et age
cp_classe_age = sns.catplot(x='pclass', y='age', data = titanic_cleaned, kind = 'box')
cp_classe_age.set_axis_labels('Classe', 'Age')
cp_classe_age.fig.suptitle('Relation entre classe et age')
print('Showing Relationship between Class and Age')
plt.show()

# 7.Relation entre classe, age et sexe
cp_classe_age_sexe = sns.catplot(x='pclass', y='age', data=titanic_cleaned, kind='box', hue='sex')
cp_classe_age_sexe.set_axis_labels('Classe', 'Age')
cp_classe_age_sexe.fig.suptitle('Relation entre la Classe, Age et Sexe')
print('Showing Relationship between Class, Age and sex')
plt.show()