import numpy as np
import pandas as pd
import random
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def predict(X_train, y_train, x_test, k):
    distances = [euclidean_distance(x_test, x_train) for x_train in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
    return most_common

def knee_curve(X_train, y_train, k_range, X_val, y_val):
    scores = []
    df_final = pd.DataFrame() 
    for k in k_range:
        predictions = [predict(X_train, y_train, x_val, k) for x_val in X_val]
        accuracy = np.mean(predictions == y_val)
        dict_={k:accuracy}
        df_temp = pd.DataFrame.from_dict(dict_, orient="index", columns=["ACCURACY"]).reset_index(names=["K"])
        df_final = pd.concat([df_temp, df_final], ignore_index=True)
        
    return df_final
	
def knee_plot_curve (scores,k_range):
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, scores, marker='o', linestyle='-')
    plt.title('KNN Knee Curve')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Accuracy')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()
     
# Exemplo de uso
# Suponha que você tenha seus dados de treinamento X_train e os rótulos correspondentes y_train
# Suponha que você tenha seus dados de validação X_val e os rótulos correspondentes y_val
# Defina a faixa de valores de K que você deseja testar
# k_range = range(1, 21)  # Testando de 1 a 20 vizinhos
# knee_curve(X_train, y_train, k_range, X_val, y_val)
# Função para dividir os dados em conjuntos de treinamento e teste usando holdout
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# Função para calcular a acurácia
def accuracy_score(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    return correct_predictions / total_predictions

# Função para calcular a matriz de confusão
def confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    return matrix

# Função para dividir os dados em conjuntos de treinamento e teste usando holdout
def train_test_split_dmc(X, y, test_size=0.2):
    data = list(zip(X, y))
    random.shuffle(data)
    test_set_size = int(len(data) * test_size)
    X_train = [d[0] for d in data[test_set_size:]]
    y_train = [d[1] for d in data[test_set_size:]]
    X_test = [d[0] for d in data[:test_set_size]]
    y_test = [d[1] for d in data[:test_set_size]]
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

# Calcular a matriz de confusão
def confusion_matrix_dmc(y_true, y_pred):
    classes = np.unique(y_true)
    num_classes = len(classes)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(num_classes):
        for j in range(num_classes):
            conf_matrix[i, j] = np.sum((y_true == classes[i]) & (y_pred == classes[j]))
    return conf_matrix

def plot_decision_surface_dmc(X, y, classifier, colunas, classes):
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z).reshape(xx.shape)
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel(colunas[0])
    plt.ylabel(colunas[1])
    plt.title("Superfície de Decisão")
    plt.show()

# Função para plotar a superfície de decisão
def plot_decision_surface(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=[cmap(idx)], marker=markers[idx], label=cl)