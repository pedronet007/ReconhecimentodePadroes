from modelos.funcoes import *

#ALUNO PEDRO WILSON FELIX M NETO KNN - 2024.1
# Criacao do Classificador CNB

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

    def _gaussian_pdf(self, X, mean, var):
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(X - mean) ** 2 / (2 * var))
    
    def predict(self, X_test):
        preds = []
        
        for x in X_test:
            class_probs_x = []
            
            for i, c in enumerate(self.classes):
                prior_prob = np.log(self.class_probs[i])
                likelihood = np.sum(np.log(self._gaussian_pdf(x, self.mean[i], self.var[i])))
                posterior_prob = prior_prob + likelihood
                class_probs_x.append(posterior_prob)
            
            preds.append(np.argmax(class_probs_x))
        
        return preds
