from modelos.cbgm import *

# Carregando os dados#
#caminho = "./dados/wdbc.data"
#colunas = ['id','class','radius (mean)','texture (mean)','perimeter (mean)','area (mean)','smoothness (mean)','compactness (mean)','concavity (mean)','concave points (mean)','symmetry (mean)','fractal dimension (mean)','radius (standard error)','texture (standard error)','perimeter (standard error)','area (standard error)','smoothness (standard error)','compactness (standard error)','concavity (standard error)','concave points (standard error)','symmetry (standard error)','fractal dimension (standard error)','radius (worst)','texture (worst)','perimeter (worst)','area (worst)','smoothness (worst)','compactness (worst)','concavity (worst)','concave points (worst)','symmetry (worst)','fractal dimension (worst)']
#dataset = pd.read_csv(caminho, names=colunas)
# Mapeando os rótulos de classe para valores numéricos
# class_mapping = {label: idx for idx, label in enumerate(np.unique(dataset['class']))}
# classes = dataset["class"].unique()
# dataset['class'] = dataset['class'].map(class_mapping)

# # Dividindo os dados em atributos e rótulos
# X = dataset.iloc[:, 2:].values
# y = dataset.iloc[:, 1].values

# Carregando os dados
caminho = "./dados/breast-cancer.data"
colunas = ['class', 'age', 'menopause', 'tumor_size', 'inv_nodes', 'node_caps', 'deg_malig', 'breast', 'breast_quad', 'irradiat']
dataset = pd.read_csv(caminho, names=colunas)

## movimentacao para manter classe como ultima coluna
# Supondo que 'df' seja o seu DataFrame e 'col_name' seja o nome da coluna que você quer mover
col_name = 'class'
# Remover a coluna desejada do DataFrame e armazenar seus dados
coluna_a_mover = dataset.pop(col_name)
# Adicionar a coluna de volta ao DataFrame na última posição
dataset[col_name] = coluna_a_mover
colunas = ['age', 'menopause', 'tumor_size', 'inv_nodes', 'node_caps', 'deg_malig', 'breast', 'breast_quad', 'irradiat', 'class']
# Recriar o DataFrame com a nova ordem das colunas
dataset = dataset[colunas]


# Supondo que 'colunas' seja uma lista com os nomes das colunas que você quer converter
# Mapear 'yes' para 1, 'no' para 0 e '?' para 2 na coluna 'node_caps'
dataset['node_caps'] = dataset['node_caps'].map({'yes': 1, 'no': 0, '?': 2})
dataset['irradiat'] = dataset['irradiat'].map({'yes': 1, 'no': 0})
dataset['breast'] = dataset['breast'].map({'left': 1, 'right': 0})
dataset['breast_quad'] = dataset['breast_quad'].map({'left_low': 0, 'left_up': 1, 'right_low': 2, 'right_up': 3, 'central': 4, '?': 5})
dataset['menopause'] = dataset['menopause'].map({'premeno': 0, 'ge40': 1, 'lt40': 2})
# Mapear '0-4' para 0, '5-9' para 1, '10-14' para 2, '15-19' para 3 e '20-24' para 4 '30-34' para 5 na coluna 'tumor_size'
dataset['tumor_size'] = dataset['tumor_size'].map({'0-4': 0, '5-9': 1, '10-14': 2, '15-19': 3, '20-24': 4, '25-29': 5, '30-34': 6, '35-39': 7, '40-44': 8, '45-49': 9, '50-54': 10, '55-59': 11})
# Mapear '20-29': 0, '30-39': 1, '40-49': 2, '50-59': 3, '60-69': 4, '70-79': 5, '80-89': 6 na coluna 'age'
dataset['age'] = dataset['age'].map({'20-29': 0, '30-39': 1, '40-49': 2, '50-59': 3, '60-69': 4, '70-79': 5, '80-89': 6})
# Mapear '0-2' para 0, '3-5' para 1, '6-8' para 2, '9-11' para 3, '12-14' para 4, '15-17' para 5, '24-26' para 6, '27-29' para 7, '30-32' para 8 na coluna 'inv_nodes'
dataset['inv_nodes'] = dataset['inv_nodes'].map({'0-2': 0, '3-5': 1, '6-8': 2, '9-11': 3, '12-14': 4, '15-17': 5, '24-26': 6, '27-29': 7, '30-32': 8})

# Verificar se há valores ausentes
#print(dataset.isnull().sum())    

# Visualizar as primeiras linhas do DataFrame
#print(dataset.head())

# Mapeando os rótulos de classe para valores numéricos
class_mapping = {label: idx for idx, label in enumerate(np.unique(dataset['class']))}
classes = dataset["class"].unique()
dataset['class'] = dataset['class'].map(class_mapping)

# Dividindo os dados em atributos e rótulos
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(dataset)
run(X,y,colunas,classes)