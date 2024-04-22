from modelos.cnb import *

# Carregando os dados
caminho = "./dados/artificial2.data"
colunas = ['a', 'b', 'class']
dataset = pd.read_csv(caminho, names=colunas)

# Mapeando os rótulos de classe para valores numéricos
class_mapping = {label: idx for idx, label in enumerate(np.unique(dataset['class']))}
classes = dataset["class"].unique()
dataset['class'] = dataset['class'].map(class_mapping)

# Dividindo os dados em atributos e rótulos
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


run(X,y,colunas,classes)