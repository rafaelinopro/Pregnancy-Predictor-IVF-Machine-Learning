# Carga de librerias y funciones
from utils.libreries import *
from utils.functions import *
from imblearn.under_sampling import RandomUnderSampler


# carga de dataset limpio
data = pd.read_csv('/data/processed/dataset_completo_limpio.csv')

# Definicion de target
target = 'Live birth occurrence'

#carga de feauture selector
selector = pickle.load(open('model/model_metrics/selector.pkl','rb'))

# Reduccion de feautures del dataset limpio, completo. Y reestablecimiento de target.
data_fselected = data.loc[:,selector.get_feature_names_out()]
data_fselected[target] = data[target]

# Aplicacion del undersampled 1/3-2/3 al dataset con feauture selection. 
# Creacion de X e y
rus = RandomUnderSampler(sampling_strategy=0.5,random_state=42)
X_rus, y_rus = rus.fit_resample(data_fselected.loc[:, data_fselected.columns != target], data_fselected[target])

X_rus = pd.DataFrame(X_rus, columns=data_fselected.loc[:, data_fselected.columns != target].columns)
y_rus = pd.DataFrame(y_rus, columns=[target])

data_fselected_us= pd.concat([X_rus, y_rus], axis=1)

# Train test split con 0.25
X_train, y_train, X_test, y_test = train_test_split(X_rus, y_rus, test_size=0.25, random_state=42)

# Mejores parámetros del modelo seleccionado
params = "criterion= 'mse',learning_rate= 0.1, loss='exponential', max_depth= 7, n_estimators= 20"

# instancia del modelo seleccionado (GradientBoostingClassifier)
gbc = GradientBoostingClassifier(params)

# Entrenamiento del modelo con X_train e y_train
modelo = gbc.fit(X_train, y_train)
pickle.dump(modelo, open('model/model_metrics/new_model.pkl', 'wb'))

# Prediccion del modelo con X_test
y_pred = modelo.predict(X_test)



