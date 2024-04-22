from modelos.funcoes import *
from collections import defaultdict
#ALUNO PEDRO WILSON FELIX M NETO KNN - 2024.1
# Criacao do Classificador CNB

class NaiveBayesClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_probs = np.zeros(len(self.classes))
        self.feature_probs = []
        for c in self.classes:
            X_c = X[y == c]
            idx = np.where(self.classes == c)[0][0]  # Obter o índice do elemento c em self.classes
            self.class_probs[idx] = len(X_c) / len(X)
            feature_probs_c = []
            for feature in range(X.shape[1]):
                unique_values, counts = np.unique(X_c[:, feature], return_counts=True)
                feature_probs_c.append({val: (counts[i] + 1) / (len(X_c) + len(unique_values)) for i, val in enumerate(unique_values)})
            self.feature_probs.append(feature_probs_c)

    def predict(self, X):
        preds = []
        for x in X:
            class_probs_x = []
            for idx, c in enumerate(self.classes):
                idx = np.where(self.classes == c)[0][0]  # Obter o índice do elemento c em self.classes
                class_prob = np.log(self.class_probs[idx])
                for feature, val in enumerate(x):
                    if val in self.feature_probs[idx][feature]:
                        class_prob += np.log(self.feature_probs[idx][feature][val])
                class_probs_x.append(class_prob)
            preds.append(np.argmax(class_probs_x))
        return preds

def confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def run_multiple_iterations(X, y, num_iterations=20, test_size=0.2, random_state=None):
    accuracies = np.zeros(num_iterations)
    for i in range(num_iterations):
        # Dividindo os dados em treino e teste
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=i)

        # Criando e treinando o classificador Naive Bayes
        nb_classifier = NaiveBayesClassifier()
        nb_classifier.fit(X_train, y_train)

        # Fazendo previsões
        y_pred = nb_classifier.predict(X_test)

        # Calculando a acurácia
        accuracy = calculate_accuracy(y_test, y_pred)

        # Armazenando a acurácia
        accuracies[i] = accuracy
    # Calculando a média e o desvio padrão das acurácias
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    # Exibindo os resultados
    print("Acurácias:")
    print(accuracies)
    print(f"Acurácia média: {mean_accuracy:.2f}")
    print(f"Desvio padrão: {std_accuracy:.2f}")
    return accuracies

def plot_gaussians(X, y, classes, attribute1_index, attribute2_index, colunas):
    plt.figure(figsize=(10, 8))
    for idx, class_name in enumerate(classes):
        class_data = X[y == idx]
        mean = np.mean(class_data, axis=0)
        # Convertendo a lista de valores para um array NumPy
        class_data_array = np.array(class_data)
        # Calculando a matriz de covariância
        cov = np.cov(class_data_array.T)
        #cov = np.cov(class_data.T)
        samples = np.random.multivariate_normal(mean, cov, 1000)
        plt.scatter(samples[:, attribute1_index], samples[:, attribute2_index], label=f'{class_name} gaussian', alpha=0.2)
    # Configurar o plot
    plt.xlabel(f'Feature {colunas[attribute1_index]}')
    plt.ylabel(f'Feature {colunas[attribute2_index]}')
    plt.title('Gaussian Distributions')
    plt.legend()
    plt.show()

def plot_decision_surface_cnb(X, y, classifier, classes, attribute1_index, attribute2_index, colunas):
    h = .02  
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    Z = np.array(Z)  # Convertendo para um array numpy
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    for idx, class_name in enumerate(classes):
        plt.scatter(X[y == idx][:, attribute1_index], X[y == idx][:, attribute2_index], edgecolor='k', s=20, label=class_name)

    # Configurar o plot
    plt.xlabel(f'Feature {colunas[attribute1_index]}')
    plt.ylabel(f'Feature {colunas[attribute2_index]}')
    plt.title('Decision Surface')
    plt.legend()
    plt.show()

def run(X,y,colunas,classes):
    # Definindo configurações para as realizações
    num_realizacoes = 20
    test_size = 0.2   # Separando 80% para treinamento e 20% para teste
    random_realization = 42   
    # Dividindo os dados em treino e teste
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_realization)

    # Criando e treinando o classificador Naive Bayes
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.fit(X_train, y_train)

    # Chamando a função para executar as 20 realizações
    accuracies = run_multiple_iterations(X, y, num_iterations=num_realizacoes, test_size=test_size, random_state=random_realization)
    # Encontrar a realização com a melhor acurácia
    best_accuracy_index = np.argmax(accuracies)
    best_accuracy = accuracies[best_accuracy_index]
    print("Melhor acurácia ao utilizar randon_state {}, acontece na (Realização {}): {:.2f}".format(random_realization,best_accuracy_index + 1, best_accuracy))


    # Escolher um par de atributos para a superfície de decisão sepal-width e sepal-length
    attribute1_index = 0
    attribute2_index = 1
    # Dados de treinamento e teste com os atributos escolhidos
    X_train_selected = X_train[:, [attribute1_index, attribute2_index]]
    X_test_selected = X_test[:, [attribute1_index, attribute2_index]]
    print("Dados de Treinamento:")
    dados_com_labels = np.column_stack((X_train,y_train.astype(int)))
    print (dados_com_labels)
    print("Dados de teste:")
    dados_com_labels_teste = np.column_stack((X_test, y_test.astype(int)))
    print (dados_com_labels_teste)

    # Fazendo previsões
    y_pred = nb_classifier.predict(X_test)

    # Calculando a matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred, num_classes=len(np.unique(y)))
    # Exibindo a matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Plotando a superfície de decisão
    plot_decision_surface_cnb(X_train, y_train, nb_classifier, classes, attribute1_index, attribute2_index, colunas)
    # Plotando as gaussianas
    plot_gaussians(X_train, y_train, classes, attribute1_index, attribute2_index, colunas)
    # Plotar os conjuntos de dados de treinamento e teste
    plot_dataset(X_train_selected, y_train, X_test_selected, y_test, colunas, attribute1_index, attribute2_index)
