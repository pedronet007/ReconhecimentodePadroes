from modelos.cbgm import *

# Carregando os dados
caminho = "./dados/dermatology.data"
colunas = ['erythema','scaling','definite borders','itching','koebner phenomenon','polygonal papules','follicular papules','oral mucosal involvement','knee and elbow involvement','scalp involvement','family history','melanin incontinence','eosinophils in the infiltrate','PNL infiltrate','fibrosis of the papillary dermis','exocytosis','acanthosis','hyperkeratosis','parakeratosis','clubbing of the rete ridges','elongation of the rete ridges','thinning of the suprapapillary epidermis','spongiform pustule','munro microabcess','focal hypergranulosis','disappearance of the granular layer','vacuolisation and damage of basal layer','spongiosis','saw-tooth appearance of retes','follicular horn plug','perifollicular parakeratosis','inflammatory monoluclear inflitrate','band-like infiltrate','Age','class']
dataset = pd.read_csv(caminho, names=colunas)
# Substituindo '?' por 0
dataset.replace('?', 0, inplace=True)

# Convertendo todos os valores para tipo numérico (float)
dataset = dataset.apply(pd.to_numeric, errors='coerce')

# Mapeando os rótulos de classe para valores numéricos
class_mapping = {label: idx for idx, label in enumerate(np.unique(dataset['class']))}
classes = dataset["class"].unique()
dataset['class'] = dataset['class'].map(class_mapping)

# Dividindo os dados em atributos e rótulos
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


run(X,y,colunas,classes)
