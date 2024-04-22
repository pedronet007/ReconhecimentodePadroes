from modelos.cnb import *

# Carregando os dados
caminho = "./dados/column_3C.dat"
colunas = ['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius','degree_spondylolisthesis','class']
dataset = pd.read_csv(caminho, names=colunas, delimiter=' ')

# Mapeando os rótulos de classe para valores numéricos
class_mapping = {label: idx for idx, label in enumerate(np.unique(dataset['class']))}
classes = dataset["class"].unique()
dataset['class'] = dataset['class'].map(class_mapping)

# Dividindo os dados em atributos e rótulos
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


run(X,y,colunas,classes)