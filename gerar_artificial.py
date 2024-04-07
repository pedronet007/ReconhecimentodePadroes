from modelos.funcoes import *

def generate_artificial_dataset():
    np.random.seed(42)

    # Parâmetros para a Classe 1
    mean1 = [1, 1]
    cov1 = [[0.1, 0], [0, 0.1]]
    class1 = np.random.multivariate_normal(mean1, cov1, 10)

    # Parâmetros para a Classe 0
    mean2 = [0, 0]
    cov2 = [[0.1, 0], [0, 0.1]]
    class0_1 = np.random.multivariate_normal(mean2, cov2, 10)

    mean3 = [0, 1]
    cov3 = [0.1, 0], [0, 0.1]
    class0_2 = np.random.multivariate_normal(mean3, cov3, 10)

    mean4 = [1, 0]
    cov4 = [[0.1, 0], [0, 0.1]]
    class0_3 = np.random.multivariate_normal(mean4, cov4, 10)

    class0 = np.vstack((class0_1, class0_2, class0_3))

    # Combinar as classes
    X_artificial = np.vstack((class1, class0))
    y_artificial = np.array([1]*10 + [0]*30)

    return X_artificial, y_artificial

# Gerar dados artificiais
X_artificial, y_artificial = generate_artificial_dataset()

# Concatenar rótulos com dados
data_with_labels = np.column_stack((X_artificial, y_artificial.astype(int)))

# Salvar no arquivo .dat
file_path = "./dados/artificial.data"
np.savetxt(file_path, data_with_labels, delimiter=',',fmt='%f,%f,%d')
print (data_with_labels)