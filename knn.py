import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#ALUNO PEDRO WILSON FELIX M NETO KNN IRIS DATASET

# Definindo a função de distância Euclidiana
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


# Implementação do algoritmo KNN
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Calcular distâncias
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Ordenar por distância e obter os índices dos k vizinhos mais próximos
        k_indices = np.argsort(distances)[:self.k]
        # Extrair rótulos dos vizinhos mais próximos
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Retornar o rótulo mais comum
        most_common = np.argmax(np.bincount(k_nearest_labels))
        return most_common
   
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


# Realizando 20 interações
def cross_validation (k,X,y):
	n_realizations = 20
	accuracies = []
	for i in range(n_realizations):
		# Dividindo os dados em conjuntos de treinamento e teste
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

		# Normalizando os dados (opcional)
		# scaler = StandardScaler()
		# X_train = scaler.fit_transform(X_train)
		# X_test = scaler.transform(X_test)

		# Treinando o modelo KNN
		knn = KNN(k=k)
		knn.fit(X_train, y_train)

		# Fazendo previsões
		y_pred = knn.predict(X_test)

		# Calculando a acurácia
		accuracy = accuracy_score(y_test, y_pred)
		accuracies.append(accuracy)
		#conf_matrix = confusion_matrix(y_test, y_pred, num_classes=len(np.unique(y)))
		#print(f"Matriz de confusão para a realização {i}:\n{conf_matrix}")

	# Calculando média e desvio padrão da acurácia
	mean_accuracy = np.mean(accuracies)
	#Desvio padrao
	std_accuracy = np.std(accuracies)

	print(f"Acurácia média: {mean_accuracy}")
	print(f"Desvio padrão da acurácia: {std_accuracy}")

#Chamando o KNN e passando os atributos com seus rótulos
def run(X,y,colunas,classes):
	# Defina a faixa de valores de K que você deseja testar
	k_range = range(1, 21)  # Testando de 1 a 20 vizinhos
	random_realization = 42 #Numero 42 é mágico mais utilizado. rsrs
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_realization)
	X_train = X_train[:, :2]
	X_test = X_test[:, :2]
	# Chame a função knee_curve para plotar a curva do joelho para encontrar o melhor valor de K
	df_scores = knee_curve(X_train, y_train, k_range, X_test, y_test)
    #Escolha do melhor K a partir da ordenação da maior acuracia pegando seu maior valor do dicionario criado com a lista de K e acuracias a partir da curva do joelho
	scores = df_scores["ACCURACY"].tolist()
	k_optimal = df_scores.sort_values("ACCURACY", ascending=False).head(1)["K"].values[0]
	print("Optimal K", k_optimal) 
	knee_plot_curve(scores,k_range)
	print(df_scores)
	knn = KNN(k=k_optimal)
	knn.fit(X_train, y_train)
	y_pred = knn.predict(X_test)
	conf_matrix = confusion_matrix(y_test, y_pred, num_classes=len(np.unique(y)))
	print(f"Matriz de confusão para a realização {random_realization}:\n{conf_matrix}")

	# Escolhendo um par de atributos aleatório para plotar a superfície de decisão
	# print(X_train[:, :2].shape, X_train_subset.shape)
	plt.figure(figsize=(10, 6))
	plot_decision_surface(X_train, y_train, knn)
	plt.xlabel(colunas[0])
	plt.ylabel(colunas[1])
	plt.title('Superfície de Decisão para os Atributos Selecionados')
	plt.legend(loc='upper left')

	## plot da matriz de confusão com o seaborn

	plt.figure(figsize=(10,6))
	fx=sns.heatmap(conf_matrix, annot=True, fmt=".2f",cmap="GnBu")
	fx.set_title('Confusion Matrix \n');
	fx.set_xlabel('\n Predicted Values\n')
	fx.set_ylabel('Actual Values\n');
	fx.xaxis.set_ticklabels(classes)
	fx.yaxis.set_ticklabels(classes)
	plt.show()

	#Chamada das 20 realizações crosvalidando e verificando a acuracia 
	cross_validation(k_optimal, X,y)