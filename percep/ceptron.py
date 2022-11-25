import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
dataset = pd.read_csv('run_results.csv')
dataset.columns = ['Nombre Técnica', 'Link Técnica', 'Nombre Subtécnica', 'Link
Subtécnica']
dataset = dataset.groupby('Nombre Subtécnica').size().reset_index()
dataset.columns = ['Nombre Subtécnica', 'Cantidad']
dataset = dataset.sort_values('Cantidad', ascending=False)
dataset = dataset.head(5)
print(dataset)
X = dataset.iloc[:, 1].values.astype(int)
X = X.reshape(-1, 1)
y = dataset.iloc[:, 0].values
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)


yp = regressor.predict([[4]])
yp = yp[0].astype(int)
print(yp)
print("Técnica predecida: ", labelencoder_y.inverse_transform([yp]))
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Técnica según aparición (Decision Tree Regression)')
plt.xlabel('# de apariciones')
plt.ylabel('Técnica(Codificada)')
plt.savefig('graph.png')
plt.show()
plt.figure(figsize=(15, 10))
tree.plot_tree(regressor, filled=True, feature_names=['# de apariciones'],
class_names=labelencoder_y.classes_)
plt.savefig('tree.png')
plt.show()