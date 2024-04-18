from modelos.funcoes import *

#ALUNO PEDRO WILSON FELIX M NETO KNN - 2024.1
# Criacao do Classificador CNB

def gaussian_pdf(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))

class NaiveBayesClassifier:
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.class_probs = np.zeros(len(self.classes))
        self.mean = []
        self.var = []
        
        for c in self.classes:
            X_c = X_train[y_train == c]
            self.class_probs[c] = len(X_c) / len(X_train)
            self.mean.append(X_c.mean(axis=0))
            self.var.append(X_c.var(axis=0) + self.epsilon)  # Adiciona epsilon para regularização

    def predict(self, X_test):
        preds = []
        
        for x in X_test:
            class_probs_x = []
            
            for i, c in enumerate(self.classes):
                prior_prob = np.log(self.class_probs[i])
                likelihood = np.sum(np.log(gaussian_pdf(x, self.mean[i], self.var[i])))
                posterior_prob = prior_prob + likelihood
                class_probs_x.append(posterior_prob)
            
            preds.append(np.argmax(class_probs_x))
        
        return preds

# Função para plotar a superfície de decisão

def plot_decision_surface_cnb(X, y, classifier,colunas, classes, attribute1_index, attribute2_index):
    h = .02  # Step size na grade
    x_min, x_max = X[:, attribute1_index].min() - 1, X[:, attribute1_index].max() + 1
    y_min, y_max = X[:, attribute2_index].min() - 1, X[:, attribute2_index].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z).reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    
	 # Adicionar a legenda para os dados de treinamento
    for i, cls in enumerate(classes):
        plt.scatter([], [], label=f'Classe {cls}', color=plt.cm.viridis(i / len(classes)))
    plt.legend()
	
    # Plot dos pontos de treinamento
    plt.scatter(X[:, attribute1_index], X[:, attribute2_index], c=y, cmap=cmap_bold, edgecolor='k', s=20, label=classes)
    #plt.xlim(xx.min(), xx.max())
    #plt.ylim(yy.min(), yy.max())
    plt.title('Decision Surface for Naive Bayes Classifier')
    #plt.xlabel('Feature {}'.format(attribute1_index))
    #plt.ylabel('Feature {}'.format(attribute2_index))
    #plt.legend()
    plt.xlabel(f'Feature {colunas[attribute1_index]}')
    plt.ylabel(f'Feature {colunas[attribute2_index]}')
    plt.show()

def plot_gaussians(X, y, classifier, colunas, classes, attribute1_index, attribute2_index):
    plt.figure(figsize=(10, 6))
    
    for i, c in enumerate(classes):
        X_c = X[y == c]
        mean = classifier.mean[i]
        var = classifier.var[i]
        x_values = np.linspace(X[:, attribute1_index].min(), X[:, attribute1_index].max(), 100)
        y_values = np.linspace(X[:, attribute2_index].min(), X[:, attribute2_index].max(), 100)
        X1, X2 = np.meshgrid(x_values, y_values)
        pdf_values = gaussian_pdf(X1, mean[attribute1_index], var[attribute1_index]) * gaussian_pdf(X2, mean[attribute2_index], var[attribute2_index])
        plt.contour(X1, X2, pdf_values, cmap='viridis', alpha=0.6)
        
    plt.title('Gaussian Distributions for Naive Bayes Classifier')
    plt.xlabel('Feature {}'.format(colunas[attribute1_index]))
    plt.ylabel('Feature {}'.format(colunas[attribute2_index]))        
    plt.legend([f'{colunas[attribute1_index]}-{colunas[attribute2_index]} {c}'])
    
    plt.legend()
    plt.show()


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


#Chamando o Classificador e passando os atributos com seus rótulos
def run(X,y,colunas,classes):
    ## Você pode alterar este valor para escolher uma realização diferente
    random_realization=42
    # Dividir os dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_realization)

    # Criar o classificador
    classifier = NaiveBayesClassifier()

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
    classifier = NaiveBayesClassifier()
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

	# Supondo que você tenha treinado o classificador e tenha seus dados em X_train e y_train
	# E tenha o classificador instanciado em nb_classifier
    plot_decision_surface_cnb(X_train_selected, y_train, classifier, colunas, classes, attribute1_index, attribute2_index)

    #Passo 7
    # Plotar as gaussianas sobre os dados para cada classe
    # Supondo que você tenha treinado o classificador e tenha seus dados em X_train e y_train
    # E tenha o classificador instanciado em nb_classifier
    # E tenha a legenda dos dados em uma variável chamada "classes"
    # E queira plotar as distribuições gaussianas para os atributos de índice 0 e 1
    plot_gaussians(X_train, y_train, classifier, colunas, classes, attribute1_index, attribute2_index)

    # Plotar os conjuntos de dados de treinamento e teste
    plot_dataset(X_train_selected, y_train, X_test_selected, y_test, colunas, attribute1_index, attribute2_index)
