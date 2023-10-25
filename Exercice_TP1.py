import numpy

# Create List
T = list(range(1, 11))
print("Affichage de la liste")
print("T =", T)

# Convert List to numpy array
T = numpy.array(T)
print("Affichage de la tableau numpy")
print("T =", T)

# Convert elemnts' type to float
T = T.astype(float)
print("Affichage de la tableau des réels")
print("T =", T)

# Nombre of elements
print("Nombre d'élements de T", T.size)

# Type of elements
print("Le type d'élements", T.dtype)

# Convert table to matrix
T2 = T.reshape(2, 5)
print(T2)

# Min, Max and Moyen
print("La valeur minimale de T", T.min())
print("La valeur maximale de T", T.max())
print("La valeur moyenne de T", T.mean())

# Sum of elements
print("La somme des valeurs des élements", sum(T))

# First and last element
print("Premier élement de T", T[0])
print("Dernier élement de T", T[-1])
# T[[0,9]]

# Affichage des élements d'indice impair
print(T[1:11:2])

# Add elements
T = numpy.insert(T, 1, 20)
T = numpy.append(T, 11)
print("Affichage tableau après insertion", T)

# Delete first 4 elements
T = numpy.delete(T, range(4))
print("Affichage du tableau après Suppresion", T)

# Table division
Tl, Tr = numpy.hsplit(T, 2)
print("Premier partie du tableau", Tl)
print("Deuxième partie du tableau", Tr)

# Concatination
M = numpy.vstack((Tl, Tr))
print("Matrice après concatination")
print(M)


