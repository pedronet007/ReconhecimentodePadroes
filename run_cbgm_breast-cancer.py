from modelos.cbgm import *

# Carregando os dados
caminho = "./dados/wdbc.data"
colunas = ['id','class','radius (mean)','texture (mean)','perimeter (mean)','area (mean)','smoothness (mean)','compactness (mean)','concavity (mean)','concave points (mean)','symmetry (mean)','fractal dimension (mean)','radius (standard error)','texture (standard error)','perimeter (standard error)','area (standard error)','smoothness (standard error)','compactness (standard error)','concavity (standard error)','concave points (standard error)','symmetry (standard error)','fractal dimension (standard error)','radius (worst)','texture (worst)','perimeter (worst)','area (worst)','smoothness (worst)','compactness (worst)','concavity (worst)','concave points (worst)','symmetry (worst)','fractal dimension (worst)']
dataset = pd.read_csv(caminho, names=colunas)

# Mapeando os rótulos de classe para valores numéricos
class_mapping = {label: idx for idx, label in enumerate(np.unique(dataset['class']))}
classes = dataset["class"].unique()
dataset['class'] = dataset['class'].map(class_mapping)

# Dividindo os dados em atributos e rótulos
X = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1].values


run(X,y,colunas,classes)
