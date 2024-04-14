from modelos.funcoes import *
#ALUNO PEDRO WILSON FELIX M NETO KNN - 2024.1
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

# Realizando 20 interações
def cross_validation (X,y):
    n_realizations = 20
    accuracies = []
    for i in range(n_realizations):
        # Divide os dados em conjunto de treinamento e teste manualmente (80% treinamento, 20% teste)
        X_train, X_test, y_train, y_test = train_test_split_dmc(X, y, test_size=0.2)

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
        #print(f"Dados de Teste na Realização {i+1}:")
        #for j, (attributes, label) in enumerate(zip(X_test, y_test)):
        #    print(f"Amostra {j+1}: Atributos={attributes}, Rótulo={label}")
    mean_accuracy = np.mean(accuracies)
    #Desvio padrao
    std_accuracy = np.std(accuracies)
    print(f"Acurácia média: {mean_accuracy}")
    print(f"Desvio padrão da acurácia: {std_accuracy}")
    # Encontrar a realização com a melhor acurácia
    best_accuracy_index = np.argmax(accuracies)
    best_accuracy = accuracies[best_accuracy_index]
    print("Melhor acurácia acontece na (Realização {}): {:.2f}".format(best_accuracy_index + 1, best_accuracy))

def run(X,y,colunas,classes):
    random_realization = 4 #Numero 42 é mágico mais utilizado. rsrs
    X_train, X_test, y_train, y_test = train_test_split_dmc(X, y, test_size=0.2)
    dmc = DMCClassifier()
    # Treina o classificador
    dmc.fit(X_train, y_train)
    print("Dados de Treinamento:")
    print (X_train)
    print (y_train)
    # Faz previsões
    y_pred = dmc.predict(X_test)
    conf_matrix = confusion_matrix_dmc(y_test, y_pred)
    print(f"Matriz de confusão :\n{conf_matrix}")
    print("Dados de teste:")
    print(X_test)
    print(y_test)
    # Escolhendo um par de atributos aleatório para plotar a superfície de decisão
    random_features = np.random.choice(range(X.shape[1]), size=2, replace=False)
    X_train_subset = X_train[:, random_features]
    X_test_subset = X_test[:, random_features]
    dmc.fit(X_train_subset, y_train)
    agora = datetime.datetime.now()
    print('entrei nas contas da superficie de decisao :',agora)
    #plot_decision_surface(X_train_subset, y_train, dmc)
    # Dentro da função `run`, após o cálculo da média e desvio padrão da acurácia, adicione:
    plot_decision_surface_dmc(X_train_subset, y_train, dmc, colunas, classes)
    agora = datetime.datetime.now()
    print('sai das contas superficie de decisao :',agora)
    

	## plot da matriz de confusão com o seaborn
    plt.figure(figsize=(10,6))
    fx=sns.heatmap(conf_matrix, annot=True, fmt=".2f",cmap="GnBu")
    fx.set_title('Confusion Matrix DMC \n');
    fx.set_xlabel('\n Predicted Values\n')
    fx.set_ylabel('Actual Values\n');
    fx.xaxis.set_ticklabels(classes)
    fx.yaxis.set_ticklabels(classes)
    plt.show()
    #Chamada das 20 realizações crosvalidando e verificando a acuracia 
    cross_validation(X,y)
    
    

   