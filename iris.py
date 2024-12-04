from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

#cargar el conjunto de datos iris
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
target = iris.target 

# añadir la columna de la especie al dataframe
data['species'] = target
data['species'] = data['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

#mostrar una vista previa de los datos
print(data.head())

# calcular estadisticas descriptivas
print(data.describe())

# visualizacion de las distribuciones por especie
sns.pairplot(data, hue='species', diag_kind='kde')
plt.show()

#mapa de calor para ver correlaciones
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('correlacion entre las caracteristicas')
plt.show()

from sklearn.model_selection import train_test_split

#division de los datos
x = data.drop(columns=['species']) #caracteristicas
y = target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)