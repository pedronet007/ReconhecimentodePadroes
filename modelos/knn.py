from modelos.funcoes import *

#ALUNO PEDRO WILSON FELIX M NETO KNN IRIS DATASET

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

# Realizando 20 interações
def cross_validation (K,X,y):
    n_realizations = 20
    accuracies = []
    for i in range(n_realizations):
        # Dividindo os dados em conjuntos de treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        # Normalizando os dados (opcional)
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)
        #Print K
        print(f"o K utilizado para o cross_validarion foi : {K}")
        # Treinando o modelo KNN
        knn = KNN(k=K)
        knn.fit(X_train, y_train)

        # Fazendo previsões
        y_pred = knn.predict(X_test)

        # Calculando a acurácia
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        conf_matrix = confusion_matrix(y_test, y_pred, num_classes=len(np.unique(y)))
        print(f"Acurácia desta realização {accuracy}, Matriz de confusão para a realização {i}:\n{conf_matrix}")
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
    random_realization = 4 #Numero 42 é mágico mais utilizado. rsrs
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_realization)
    #Alteracao que fixava axis do X_train em duas colunas
    #X_train = X_train[:, :2]
    #X_test = X_test[:, :2]
    print("Dados de Treinamento:")
    dados_com_labels = np.column_stack((X_train,y_train.astype(int)))
    print (dados_com_labels)
    
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
    print(f"Matriz de confusão com o melhor K = {k_optimal} para o random_state= {random_realization}:\n{conf_matrix}")
    print("Dados de teste:")
    dados_com_labels_teste = np.column_stack((X_test, y_test.astype(int)))
    print (dados_com_labels_teste)
    
	# Escolhendo um par de atributos aleatório para plotar a superfície de decisão
	# print(X_train[:, :2].shape, X_train_subset.shape)
    #plt.figure(figsize=(10, 6))
    #plot_decision_surface(X_train, y_train, knn)
    #plt.xlabel(colunas[0])
    #plt.ylabel(colunas[1])
    #plt.title('Superfície de Decisão para os Atributos Selecionados')
    #plt.legend(loc='upper left')

    # Escolhendo um par de atributos aleatório para plotar a superfície de decisão
    random_features = np.random.choice(range(X.shape[1]), size=2, replace=False)
    X_train_subset = X_train[:, random_features]
    X_test_subset = X_test[:, random_features]
    knn.fit(X_train_subset, y_train)
    agora = datetime.datetime.now()
    print('entrei nas contas da superficie de decisao :',agora)
    plt.figure(figsize=(10, 6))
    plot_decision_surface(X_train_subset, y_train, knn)
    agora = datetime.datetime.now()
    print('sai das contas superficie de decisao :',agora)
    plt.xlabel(colunas[random_features[0]])
    plt.ylabel(colunas[random_features[1]])
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