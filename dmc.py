import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Definindo a função de distância Euclidiana
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# Criacao do Classificador DMC
class DMCClassifier:
    def __init__(self):
        self.centroids = {}

    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        for cls in self.classes:
            # Calcula o centróide para cada classe
            self.centroids[cls] = np.mean(X_train[y_train == cls], axis=0)

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            # Calcula a distância de x a cada centróide
            distances = [euclidean_distance(x, self.centroids[cls]) for cls in self.classes]
            # Encontra a classe com o centróide mais próximo
            predicted_class = self.classes[np.argmin(distances)]
            predictions.append(predicted_class)
        return predictions

def run(X,y,colunas,classes):
    # Função para dividir os dados em conjuntos de treinamento e teste usando holdout
    def train_test_split(X, y, test_size=0.2):
        data = list(zip(X, y))
        random.shuffle(data)
        test_set_size = int(len(data) * test_size)
        X_train = [d[0] for d in data[test_set_size:]]
        y_train = [d[1] for d in data[test_set_size:]]
        X_test = [d[0] for d in data[:test_set_size]]
        y_test = [d[1] for d in data[:test_set_size]]
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    # Realizações usando holdout
    n_realizations = 20
    accuracies = []

    for i in range(n_realizations):
        # Divide os dados em conjunto de treinamento e teste manualmente (80% treinamento, 20% teste)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Instancia o classificador DMC
        dmc_classifier = DMCClassifier()

        # Treina o classificador
        dmc_classifier.fit(X_train, y_train)

        # Faz previsões
        y_pred = dmc_classifier.predict(X_test)

        # Calcula a acurácia manualmente
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)
        
        #Exibe os dados de teste
        print(f"Dados de Teste na Realização {i+1}:")
        for j, (attributes, label) in enumerate(zip(X_test, y_test)):
            print(f"Amostra {j+1}: Atributos={attributes}, Rótulo={label}")


    # Calcular a matriz de confusão
    def confusion_matrix(y_true, y_pred):
        classes = np.unique(y_true)
        num_classes = len(classes)
        conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for i in range(num_classes):
            for j in range(num_classes):
                conf_matrix[i, j] = np.sum((y_true == classes[i]) & (y_pred == classes[j]))
        return conf_matrix

    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Matriz de confusão :\n{conf_matrix}")

    # Calcula a acurácia média e o desvio padrão
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    print("\nAcurácia Média:", mean_accuracy)
    print("Desvio Padrão da Acurácia:", std_accuracy)   