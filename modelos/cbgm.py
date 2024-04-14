from modelos.funcoes import *

import matplotlib.pyplot as plt
import seaborn as sns

# Função para plotar a superfície de decisão
def plot_decision_surface_cbgm(classifier, X, y,colunas, classes, attribute1_index, attribute2_index):
    # Define os limites do gráfico
    x_min, x_max = X[:, attribute1_index].min() - 1, X[:, attribute1_index].max() + 1
    y_min, y_max = X[:, attribute2_index].min() - 1, X[:, attribute2_index].max() + 1

    # Gera um grid de pontos para plotar a superfície de decisão
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Prepara o plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plotar a superfície de decisão
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.1, cmap='viridis')

    # Plotar os dados de treinamento
    sns.scatterplot(x=X[:, attribute1_index], y=X[:, attribute2_index], hue=y, palette='viridis', marker='o', legend=False)

    # Adicionar a legenda para os dados de treinamento
    #plt.legend(labels=np.unique(y))
    # Adicionar a legenda para os dados de treinamento
    for i, cls in enumerate(classes):
        plt.scatter([], [], label=f'Classe {cls}', color=plt.cm.viridis(i / len(classes)))
    plt.legend()

    # Configurar o plot
    plt.xlabel(f'Feature {colunas[attribute1_index]}')
    plt.ylabel(f'Feature {colunas[attribute2_index]}')
    plt.title('Superfície de Decisão')
    plt.show()


def plot_gaussians(classifier, X, y, colunas, classes, attribute1_index, attribute2_index):
    # Prepara o plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # # Plotar as gaussianas para cada classe
    # for c in np.unique(y):
    #     mean = classifier.means[c][[attribute1_index, attribute2_index]]
    #     covariance = classifier.covariances[c][np.ix_([attribute1_index, attribute2_index], [attribute1_index, attribute2_index])]
    #     plot_gaussian_2d(mean, covariance, label=f'Classe {c}')

    # Adicionar a legenda para as gaussianas
    for i, cls in enumerate(classes):
        mean = classifier.means[i][[attribute1_index, attribute2_index]]
        covariance = classifier.covariances[i][np.ix_([attribute1_index, attribute2_index], [attribute1_index, attribute2_index])]
        plot_gaussian_2d(mean, covariance, label=f'{cls}')
        
    plt.legend()

    # Configurar o plot
    plt.xlabel(f'Feature {colunas[attribute1_index]}')
    plt.ylabel(f'Feature {colunas[attribute2_index]}')
    plt.title('Gaussianas para Cada Classe')
    plt.show()

def plot_gaussian_2d(mean, cov, label):
    x, y = np.random.multivariate_normal(mean, cov, 1000).T
    plt.plot(x, y, 'o', alpha=0.2, label=label)

def plot_dataset(X_train, y_train, X_test, y_test, colunas, attribute1_index, attribute2_index):
    # Prepara o plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plotar os conjuntos de dados de treinamento e teste
    train_plot = sns.scatterplot(x=X_train[:, attribute1_index], y=X_train[:, attribute2_index], hue=y_train, palette='viridis', marker='o', legend=False)
    test_plot = sns.scatterplot(x=X_test[:, attribute1_index], y=X_test[:, attribute2_index], hue=y_test, palette='viridis', marker='x', legend=False)

    # Adicionar a legenda para os conjuntos de dados
    plt.legend(handles=[train_plot.collections[0], test_plot.collections[0]], labels=['Treinamento', 'Teste'])

    # Configurar o plot
    plt.xlabel(f'Feature {colunas[attribute1_index]}')
    plt.ylabel(f'Feature {colunas[attribute2_index]}')
    plt.title('Conjuntos de Dados de Treinamento e Teste')
    plt.show()

class GaussianNB:
    def __init__(self):
        self.classes = None
        self.means = {}
        self.covariances = {}
        self.class_priors = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        total_samples = len(X)

        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.covariances[c] = np.cov(X_c, rowvar=False) + np.eye(X_c.shape[1]) * 1e-6  # Adicionando ruído para evitar a singularidade
            self.class_priors[c] = len(X_c) / total_samples

    def predict(self, X):
        predictions = []
        for x in X:
            max_class = None
            max_posterior = -np.inf

            for c in self.classes:
                mean = self.means[c]
                covariance = self.covariances[c]
                prior = self.class_priors[c]
                posterior = self.calculate_class_posterior(x, mean, covariance, prior)

                if posterior > max_posterior:
                    max_posterior = posterior
                    max_class = c

            predictions.append(max_class)
        return predictions

    def calculate_class_posterior(self, x, mean, covariance, prior):
        # Calcula a log likelihood
        exponent = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(covariance)), (x - mean))
        log_likelihood = np.log(1 / np.sqrt((2 * np.pi) ** x.shape[0] * np.linalg.det(covariance))) + exponent

        # Combina a log likelihood com o prior
        posterior = log_likelihood + np.log(prior)
        return posterior



# class GaussianNB:
#     def fit(self, X, y):
#         self.classes = np.unique(y)
#         self.class_priors = np.zeros(len(self.classes))
#         self.means = {}
#         self.covariances = {}
        
#         # Calculate class priors
#         for i, c in enumerate(self.classes):
#             X_c = X[y == c]
#             self.class_priors[i] = len(X_c) / len(X)
#             self.means[c] = np.mean(X_c, axis=0)
#             self.covariances[c] = np.cov(X_c, rowvar=False)

#     def predict(self, X):
#         y_pred = []
#         for x in X:
#             class_scores = []
#             for i, c in enumerate(self.classes):
#                 mean = self.means[c]
#                 covariance = self.covariances[c]
#                 prior = self.class_priors[i]
#                 class_scores.append(self.calculate_class_score(x, mean, covariance, prior))
#             y_pred.append(self.classes[np.argmax(class_scores)])
#         return np.array(y_pred)

#     def calculate_class_score(self, x, mean, cov, prior):
#         # Multivariate Gaussian PDF
#         exponent = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean))
#         score = np.exp(exponent) / np.sqrt((2 * np.pi) ** len(x) * np.linalg.det(cov))
#         return score * prior

def cross_validation(X, y, classifier, num_folds=5, num_repeats=20, random_state=None):
    accuracies = []
    for _ in range(num_repeats):
        if random_state:
            np.random.seed(random_state)
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        fold_size = len(X) // num_folds
        for fold in range(num_folds):
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size
            X_test = X_shuffled[test_start:test_end]
            y_test = y_shuffled[test_start:test_end]
            X_train = np.concatenate([X_shuffled[:test_start], X_shuffled[test_end:]])
            y_train = np.concatenate([y_shuffled[:test_start], y_shuffled[test_end:]])
            classifier.fit(X_train, y_train)
            predictions = classifier.predict(X_test)
            accuracy = np.mean(predictions == y_test)
            accuracies.append(accuracy)
    return accuracies


#Chamando o KNN e passando os atributos com seus rótulos
def run(X,y,colunas,classes):
    ## Você pode alterar este valor para escolher uma realização diferente
    random_realization=42
    # Dividir os dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_realization)

    # Criar o classificador
    classifier = GaussianNB()

    # Realizar a validação cruzada
    accuracies = cross_validation(X, y, classifier, num_repeats=20, random_state=random_realization)

    # Calcular média e desvio padrão da acurácia
    mean_accuracy = np.mean(accuracies)
    std_dev_accuracy = np.std(accuracies)

    print("Média da acurácia:", mean_accuracy)
    print("Desvio padrão da acurácia:", std_dev_accuracy)
    # Encontrar a realização com a melhor acurácia
    best_accuracy_index = np.argmax(accuracies)
    best_accuracy = accuracies[best_accuracy_index]
    print("Melhor acurácia ao utilizar randon_state {}, acontece na (Realização {}): {:.2f}".format(random_realization,best_accuracy_index + 1, best_accuracy))

    #Passo 5
    # Escolher uma realização aleatória para apresentar a matriz de confusão
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=best_accuracy_index+1)
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predictions, num_classes=len(np.unique(y)))
    print("Matriz de Confusão da realizacao {}: ".format(best_accuracy_index+1))
    print(conf_matrix)
    plt.figure(figsize=(10,6))
    fx=sns.heatmap(conf_matrix, annot=True, fmt=".2f",cmap="GnBu")
    fx.set_title('Confusion Matrix \n');
    fx.set_xlabel('\n Predicted Values\n')
    fx.set_ylabel('Actual Values\n');
    fx.xaxis.set_ticklabels(classes)
    fx.yaxis.set_ticklabels(classes)
    plt.show()

    #Passo 6
    # Escolher um par de atributos para a superfície de decisão
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

    # Treinar o classificador com os atributos escolhidos
    classifier.fit(X_train_selected, y_train)

    # Plotar a superfície de decisão
    plot_decision_surface_cbgm(classifier, X_train_selected, y_train, colunas, classes, attribute1_index, attribute2_index)
    #Passo 7
    # Plotar as gaussianas sobre os dados para cada classe
    plot_gaussians(classifier, X_train_selected, y_train, colunas, classes, attribute1_index, attribute2_index)

    # Plotar os conjuntos de dados de treinamento e teste
    plot_dataset(X_train_selected, y_train, X_test_selected, y_test, colunas, attribute1_index, attribute2_index)

