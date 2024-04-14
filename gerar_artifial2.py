# import numpy as np
# import matplotlib.pyplot as plt

# # Parâmetros para a Classe 1
# mean1 = [1, 1]
# cov1 = [[0.1, 0], [0, 0.1]]
# class1 = np.random.multivariate_normal(mean1, cov1, 15)

# # Parâmetros para a Classe 0
# mean2 = [0, 0]
# cov2 = [[0.1, 0], [0, 0.1]]
# class0_1 = np.random.multivariate_normal(mean2, cov2, 10)

# mean3 = [0, 1]
# cov3 = [[0.1, 0], [0, 0.1]]
# class0_2 = np.random.multivariate_normal(mean3, cov3, 5)

# # Combinar as classes
# X_artificial = np.vstack((class1, class0_1, class0_2))
# y_artificial = np.array([1]*15 + [0]*15)


# # Concatenar rótulos com dados
# data_with_labels = np.column_stack((X_artificial, y_artificial.astype(int)))

# # Salvar no arquivo .dat
# file_path = "./dados/artificial2.data"
# np.savetxt(file_path, data_with_labels, delimiter=',',fmt='%f,%f,%d')
# print (data_with_labels)



# # # Plotar o conjunto de dados
# # plt.figure(figsize=(6, 6))
# # plt.scatter(X_artificial[:, 0], X_artificial[:, 1], c=y_artificial, cmap='coolwarm', marker='o')
# # plt.title('Artificial')
# # plt.xlabel('X1')
# # plt.ylabel('X2')
# # plt.grid(True)
# # plt.show()


import numpy as np
import pandas as pd

# Criação de dados para cada cluster
np.random.seed(42)
circles = np.random.normal(loc=0, scale=0.5, size=(8, 2)) + 2
crosses = np.random.normal(loc=1, scale=0.5, size=(7, 2)) + 2
triangles = np.random.normal(loc=-1, scale=0.5, size=(8, 2)) + 2

# Criação de rótulos para cada cluster
labels = np.array(['Circulo']*8 + ['Estrela']*7 + ['Triangulo']*8)

# Concatenação dos dados e rótulos em um DataFrame
data = pd.DataFrame(np.vstack([circles, crosses, triangles]), columns=['X1', 'X2'])
data['label'] = labels

print(data.head(50))
data.to_csv('./dados/artificial2.data', index=False)
