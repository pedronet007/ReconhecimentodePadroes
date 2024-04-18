from modelos.funcoes import *
from collections import defaultdict
#ALUNO PEDRO WILSON FELIX M NETO KNN - 2024.1
# Criacao do Classificador CNB


class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_priors = self.compute_class_priors(y)
        self.means, self.stds = self.compute_class_stats(X, y)
    
    def predict(self, X):
        predictions = []
        for x in X:
            posterior_probs = self.calculate_posterior_probs(x)
            predictions.append(np.argmax(posterior_probs))
        return predictions
    
    def compute_class_priors(self, y):
        class_counts = defaultdict(int)
        for label in y:
            class_counts[label] += 1
        return {label: count / len(y) for label, count in class_counts.items()}
    
    def compute_class_stats(self, X, y):
        means = {}
        stds = {}
        for label in self.classes:
            X_class = X[y == label]
            means[label] = np.mean(X_class, axis=0)
            stds[label] = np.std(X_class, axis=0)
        return means, stds
    
    def calculate_likelihood(self, x, mean, std):
        exponent = -0.5 * ((x - mean) / std) ** 2
        return np.exp(exponent) / (np.sqrt(2 * np.pi) * std)
    
    def calculate_posterior_probs(self, x):
        posterior_probs = {}
        for label in self.classes:
            class_prior = self.class_priors[label]
            likelihoods = self.calculate_likelihood(x, self.means[label], self.stds[label])
            posterior_probs[label] = np.prod(likelihoods) * class_prior
        return posterior_probs
    
    def plot_gaussians(self, colunas):
        num_classes = len(self.classes)
        num_features = len(self.means[self.classes[0]])
        plt.figure(figsize=(12, 8))

        for i, label in enumerate(self.classes):
            plt.subplot(1, num_classes, i + 1)
            plt.title(f"Classe {label}")
            for j in range(num_features):
                mean = self.means[label][j]
                std = self.stds[label][j]
                x_values = np.linspace(mean - 3 * std, mean + 3 * std, 100)
                y_values = self.calculate_likelihood(x_values, mean, std)
                plt.plot(x_values, y_values, label=colunas[j])
                plt.xlabel("Valor")
                plt.ylabel("Densidade de Probabilidade")

        plt.legend()
        plt.tight_layout()
        plt.show()


def run(X,y,colunas,classes):
    random_realization=42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_realization)

    # Treinar o modelo
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    # Fazer previsões
    predictions = nb.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print("Acurácia:", accuracy)
    # Plotar as distribuições gaussianas
    nb.plot_gaussians(colunas=colunas)
