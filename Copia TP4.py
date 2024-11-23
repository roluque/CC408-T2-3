# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# %%
# Abrimos la carpeta donde se encuentran las bases de datos
os.chdir(r"C:/Users\s1248850/OneDrive - Syngenta/Documents/Rosario Luque/Ciencias de datos/TP/TP3")
#os.chdir(r"c:\Users\s1290226\OneDrive - Syngenta\Desktop\UdeSA\Ciencia de datos\CC408-T2-3\TP4")
#os.chdir(r"C:\Users\clari\OneDrive\Documents\Tutoriales CD\CC408-T2-3\TP4")
#os.chdir(r"c:\Users\Teresa\Desktop\Cuarto año\Semestre de primavera\Ciencia de Datos\CC408-T2-3\TP4")

# Guardamos las bases de datos en en dos variables
ind_2004 = pd.read_stata("Individual_t104.dta")
hogar_2004 = pd.read_stata("Hogar_t104.dta")
ind_2024 = pd.read_excel("usu_individual_T124.xlsx")
hogar_2024 = pd.read_excel("usu_hogar_T124.xlsx")

# %%
ind_2004.shape, hogar_2004.shape, ind_2024.shape, hogar_2024.shape

# %%
hogar_2024.columns

# %% [markdown]
# 2. Descarguen la base de microdatos de la EPH correspondiente al primer trimestre de 2004 y 2024 en formato .dta y .xls, respectivamente. La base de hogares se llama Hogar_t104.dta y usu_hogar_T124.xls, respectivamente. Eliminen todas las observaciones que no corresponden a los aglomerados de Ciudad Autónoma de Buenos Aires o Gran Buenos Aires y unan ambos trimestres en una sola base. Esto es, a la base de la encuesta individual de cada año (que usaron en el TP3) unan la base de la encuesta de hogar. Asegúrese de estar usando las variables CODUSU y NRO_Hogar para el merge.

# %% [markdown]
# Para poder trabajar con las cuatro bases de datos es necesario primero pasar el nombre de todas las columnas a minúscula. Esto nos permitirá luego concatenar las bases de datos en una sola y preservar la estructura de la información.

# %%
# En la base de datos del 2024 los nombres de las columnas están en mayúsculas.
# Entonces, cambiamos los nombres de las columnas a minúsculas.
ind_2024.columns = ind_2024.columns.str.lower()
hogar_2024.columns = hogar_2024.columns.str.lower()

#Just in case, hacemos lo mismo para las bases del 2004.
ind_2004.columns = ind_2004.columns.str.lower()
hogar_2004.columns = hogar_2004.columns.str.lower()

# %% [markdown]
# Ahora eliminamos todas las observaciones que no corresponden a los aglomerados de Ciudad Autónoma de Buenos Aires o Gran Buenos Aires, y unimos ambos trimestres de hogares e individuos en una sola base.

# %%
# Contar el número de filas en el dataframe ind_2004 que son 'Ciudad de Buenos Aires' o 'Partidos del GBA' en la columna "aglomerado"
count_32_33 = ind_2004[ind_2004['aglomerado'].isin(['Ciudad de Buenos Aires', 'Partidos del GBA'])].shape[0]
print(f"El número de filas con aglomerado 32 y 33 para el año 2004 es: {count_32_33}")
# Contar el número de filas en el dataframe ind_2024 que tienen el valor 32 y 33 en la columna "aglomerado"
count_32_33_2024 = ind_2024[ind_2024['aglomerado'].isin([32, 33])].shape[0]
print(f"El número de filas con aglomerado 32 y 33 para el año 2024 es: {count_32_33_2024}")
# Contar el número de filas en el dataframe hogar_2004 que son 'Ciudad de Buenos Aires' o 'Partidos del GBA' en la columna "aglomerado"
count_32_33_hogar = hogar_2004[hogar_2004['aglomerado'].isin(['Ciudad de Buenos Aires', 'Partidos del GBA'])].shape[0]
print(f"El número de filas con aglomerado 32 y 33 en la base de hogares para el año 2004 es: {count_32_33_hogar}")
# Contar el número de filas en el dataframe hogar_2024 que tienen el valor 32 y 33 en la columna "aglomerado"
count_32_33_hogar_2024 = hogar_2024[hogar_2024['aglomerado'].isin([32, 33])].shape[0]
print(f"El número de filas con aglomerado 32 y 33 en la base de hogares para el año 2024 es: {count_32_33_hogar_2024}")


# %%
# Primero filtramos los aglomerados de Ciudad Autónoma de Buenos Aires (32) y Gran Buenos Aires (33)
ind_2004 = ind_2004[ind_2004['aglomerado'].isin(['Ciudad de Buenos Aires', 'Partidos del GBA'])]
ind_2024 = ind_2024[ind_2024['aglomerado'].isin([32, 33])]
hogar_2004 = hogar_2004[hogar_2004['aglomerado'].isin(['Ciudad de Buenos Aires', 'Partidos del GBA'])]
hogar_2024 = hogar_2024[hogar_2024['aglomerado'].isin([32, 33])]

# %% [markdown]
# Antes de unir las bases de datos, vamos a quedarnos con las variables que usamos en el TP3 y vamos a reorganizar las variables. Seguramente tengamos que hacer esto con las bases de hogares, lo iremos haciendo al paso.

# %%
# Primero hacemos el recorte de variables de interés de la base de individuos en función de lo realizado en el tp3.
ind_2004 = ind_2004[['codusu', 'nro_hogar', 'ano4', 'ch04', 'ch06', 'ch07', 'ch08', 'nivel_ed', 'estado', 'cat_inac', 'ipcf']]
ind_2024 = ind_2024[['codusu', 'nro_hogar', 'ano4', 'ch04', 'ch06', 'ch07', 'ch08', 'nivel_ed', 'estado', 'cat_inac', 'ipcf']]
print(ind_2004.columns)
print(ind_2024.columns)

# %%
# Ahora ajustamos las variables de la base de invididuos del 2004 para que tengan el mismo formato que las de 2024.
# Se convierte a numérico los valores de ch06 (edades) de los datos de 2004, convirtiendo en na los valores que no son numeros.
ind_2004['ch06'] = pd.to_numeric(ind_2004['ch06'], errors='coerce')
# Ahora convertimos las variables de 2004 al formato numérico de 2024, para facilitar posteriormente el análisis.
#ch04=genero
ind_2004['ch04'] = ind_2004['ch04'].replace({'Varón': 1, 'Mujer': 2})
#ch07= estado civil
ind_2004['ch07'] = ind_2004['ch07'].replace({'Unido': 1, 'Casado': 2,'Separado o divorciado':3, 'Viudo':4,'Ns./Nr.':0,'Soltero':5})
#ch08= estado de salud
mapeosalud = {'Obra social (incluye PAMI)': 1,'No paga ni le descuentan': 2,'Mutual/Prepaga/Servicio de emergencia': 3,'Obra social y mutual/prepaga/servicio de emergencia': 12,'Planes y seguros públicos': 3,'Ns./Nr.': 9,  'Obra social, mutual/prepaga/servicio de emergencia y planes': 123,'Obra social y planes y seguros públicos': 13,'Mutual/prepaga/servicio de emergencia/planes y seguros públi': 23}
ind_2004['ch08'] = ind_2004['ch08'].map(mapeosalud)
ind_2004['ch08'] = pd.to_numeric(ind_2004['ch08'], errors='coerce')
#nivel_ed= nivel educativo
mapeonivel = {'Primaria Incompleta (incluye educación especial)': 1,'Primaria Completa': 2,'Secundaria Incompleta': 3,'Secundaria Completa': 4,'Superior Universitaria Incompleta': 5,'Superior Universitaria Completa': 6,'Sin instrucción': 7,'Ns./ Nr.': 9  }
ind_2004['nivel_ed'] = ind_2004['nivel_ed'].map(mapeonivel)
ind_2004['nivel_ed'] = pd.to_numeric(ind_2004['nivel_ed'], errors='coerce')
#estado = estado laboral
mapeoestado = {'Entrevista individual no realizada (no respuesta al cuestion': 0,'Ocupado': 1,'Desocupado': 2,'Inactivo': 3,'Menor de 10 años': 4}
ind_2004['estado'] = ind_2004['estado'].map(mapeoestado)
ind_2004['estado'] = pd.to_numeric(ind_2004['estado'], errors='coerce')
#cat_inac= categoría de inactividad
mapeoinac = {'Jubilado / Pensionado': 1,'Rentista': 2,'Estudiante': 3,'Ama de casa': 4,'Menor de 6 años': 5,'Discapacitado': 6,'Otros': 7,0.0:0}
ind_2004['cat_inac'] = ind_2004['cat_inac'].map(mapeoinac)
ind_2004['cat_inac'] = pd.to_numeric(ind_2004['cat_inac'], errors='coerce')

# %%
# Ahora chequeamos si hay nulls en la base de datos de 2004
print(ind_2004.isnull().sum())
# Y si hay nulls en la base de datos de 2024
print(ind_2024.isnull().sum())
#Hay 135 valores faltantes para la variable ch06 en la base de datos de 2004.
#Hay 668 valores faltantes para la variable cat_inac en la base de datos de 2004.
#No hay valores faltantes en la base de datos de 2024.

# %%
# Chequeamos si hay valores negativos de IPCF en la base de datos de 2004.
print((ind_2004['ipcf']<0).sum())
print((ind_2024['ipcf']<0).sum())

# %%
# Chequeamos si hay valores negativos en edad (ch06)
print((ind_2004['ch06']<0).sum())
print((ind_2024['ch06']<0).sum())
#Hay 51 valores negativos de edad en la base del 2024.

# %%
#Eliminamos los valores de edad que sean menores a 0 en ambas bases de datos.
ind_2024 = ind_2024[ind_2024['ch06'] > 0]
ind_2004 = ind_2004[ind_2004['ch06'] > 0]

# %%
#Eliminamos los valores faltantes de la base de datos de 2004.
ind_2004 = ind_2004.dropna()
#Chequeamos que no haya valores faltantes en la base de datos de 2004.
print(ind_2004.isnull().sum())
#Chequeamos que no haya valores faltantes en la base de datos de 2024.
print(ind_2024.isnull().sum())

# %%
# Reportar las columnas con valores faltantes para la base de hogares del 2004.
missing_values = hogar_2024.isnull().sum()
missing_columns = missing_values[missing_values > 0]
if not missing_columns.empty:
    print("Columnas con valores faltantes y su cantidad de NAs:")
    print(missing_columns)
else:
    print("No hay columnas con valores faltantes.")

# %%
# Reportar las columnas con valores faltantes para la base de hogares del 2024.
missing_values_2024 = hogar_2024.isnull().sum()
missing_columns_2024 = missing_values_2024[missing_values_2024 > 0]
if not missing_columns_2024.empty:
    print("Columnas con valores faltantes y su cantidad de NAs:")
    print(missing_columns_2024)
else:
    print("No hay columnas con valores faltantes.")

# %%
# Dado que en el 2024, hay muchas variables con muchos nas, vamos a quedarnos solo con aquellas que vamos a usar para el análisis.
# Según el análisis del diseño del registro, aquellas variables que podrían constribuir a la mejora de la predicción son:
# 'ITF', 'V4', 'IX_Tot', 'IX_Men10', 'V5' e 'II7'. Por lo tanto, nos quedamos con esas variables.
hogar_2004 = hogar_2004[['codusu', 'nro_hogar', 'ano4', 'itf', 'ix_men10', 'v4', 'ix_tot', 'v5', 'ii7']]
hogar_2024 = hogar_2024[['codusu', 'nro_hogar', 'ano4', 'itf', 'ix_men10', 'v4', 'ix_tot', 'v5', 'ii7']]

# %%
# Dado que la base de datos de hogares del 2004 tiene la descripción como valor en vez del valor numérico, vamos a cambiarlo.
#itf= ingreso total familiar
hogar_2004['itf'] = pd.to_numeric(hogar_2004['itf'], errors='coerce')
#ix_men10= cantidad de menores de 10 años
hogar_2004['ix_men10'] = pd.to_numeric(hogar_2004['ix_men10'], errors='coerce')
#v4= en los últimos tres meses, las personas de este hogar han vivido de seguro de desempleo
hogar_2004['v4'] = hogar_2004['v4'].replace({'Sí': 1, 'No': 2, 'Ns./Nr.': 0, 0.0:0})
hogar_2004['v4'] = pd.to_numeric(hogar_2004['v4'], errors='coerce')
#ix_tot= cantidad total de integrantes del hogar
hogar_2004['ix_tot'] = pd.to_numeric(hogar_2004['ix_tot'], errors='coerce')
#v5= en los últimos tres meses, las personas de este hogar han vivido de subsidio o ayuda social.
hogar_2004['v5'] = hogar_2004['v5'].replace({'Sí': 1, 'No': 2, 'Ns./Nr.': 0, 0.0:0})
hogar_2004['v5'] = pd.to_numeric(hogar_2004['v5'], errors='coerce')
mapeo_ii7 = {'Otra situación':9, 'Propietario de la vivienda y el terreno':1, 'Inquilino/arrendatario de la vivienda':3, 'Ocupante por pago de impuestos/expensas':4, 'Propietario de la vivienda solamente':2, 'Ocupante gratuito (con permiso)':6, 'Ocupante en relación de dependencia':5, 'Ocupante de hecho (sin permiso)':7, 'Está en sucesión':8,0.0:0}
hogar_2004['ii7'] = hogar_2004['ii7'].map(mapeo_ii7)

# %%
#Nuevamente chequeo el número de nas para las variables seleccionadas.
print(hogar_2004.isnull().sum())
print(hogar_2024.isnull().sum())

# %%
# Elimino los valores faltantes de la base de datos de hogares del 2004.
hogar_2004 = hogar_2004.dropna()

# %%
#Chequeo que las variables itf, ix_men10, e ix_tot no tengan valores negativos.
print((hogar_2004['itf']<0).sum())
print((hogar_2004['ix_men10']<0).sum())
print((hogar_2004['ix_tot']<0).sum())

# %%
#Chequeo que las variables itf, ix_men10, e ix_tot no tengan valores negativos.
print((hogar_2024['itf']<0).sum())
print((hogar_2024['ix_men10']<0).sum())
print((hogar_2024['ix_tot']<0).sum())

# %% [markdown]
# Ahora sí, una vez eliminados los valores faltantes y los valores sin sentido, además de transformar las variables, procedemos a unir las bases de datos.

# %%
# Verificar las claves en todas las bases
claves = ["codusu", "nro_hogar"]
for df, name in zip([ind_2004, hogar_2004, ind_2024, hogar_2024],
                    ["individual_2004", "hogar_2004", "individual_2024", "hogar_2024"]):
    if not all(clave in df.columns for clave in claves):
        raise ValueError(f"Faltan claves en la base {name}")

# Unir las bases individuales y de hogar para cada año
base_2004 = pd.merge(ind_2004, hogar_2004, on=claves, how="inner")
base_2024 = pd.merge(ind_2024, hogar_2024, on=claves, how="inner")

# Agregar columna de año
base_2004["Año"] = 2004
base_2024["Año"] = 2024

# Combinar ambas bases en un solo DataFrame
base_final = pd.concat([base_2004, base_2024], ignore_index=True)

# %%
base_final.shape


# %% [markdown]
# 3. Limpien la base de datos tomando criterios que hagan sentido. Explicar cualquier decisión como el tratamiento de valores faltantes (missing values), extremos (outliers), o variables categóricas. Justifique sus decisiones.

# %%
# Check for missing values in base_final
missing_values_base_final = base_final.isnull().sum()
missing_columns_base_final = missing_values_base_final[missing_values_base_final > 0]

if not missing_columns_base_final.empty:
    print("Columnas con valores faltantes y su cantidad de NAs:")
    print(missing_columns_base_final)
else:
    print("No hay columnas con valores faltantes en base_final.")

# %%
base_final.to_excel("base_final.xlsx", index=False)

# %% [markdown]
# 4. Construya variables (mínimo 3) que no estén en la base pero que sean relevantes para predecir individuos desocupados (por ejemplo, la proporción de personas que trabajan en el hogar).

# %%
# Primero, vamos a crear las variables que utilizamos en el tp3 para la predicción del desempleo.
# La columna PEA (Población Económicamente Activa) que toma 1 si están ocupados o desocupados en ESTADO.
base_final['PEA'] = base_final['estado'].apply(lambda x: 1 if x in [1, 2] else 0)

# %%
# La columna PET (Población en Edad para Trabajar) toma 1 si están la persona tiene entre 15 y 65 años cumplidos. 
base_final['PET'] = base_final['ch06'].apply(lambda x: 1 if 15 <= x <= 65 else 0)

# %%
# Agregamos una columna llamada desocupado que tome 1 si esta desocupada
base_final['desocupado'] = base_final['estado'].apply(lambda x: 1 if x == 2 else 0)


# %%
# Ahora vamos a crear una nueva columna que contenga la proporción de personas que trabajan en el hogar.
# Para esto, vamos a dividir la cantidad de personas ocupadas en el hogar por la cantidad total de personas en el hogar.
base_final['prop_ocupados'] = base_final['PEA'] / base_final['ix_tot']

# Ahora vamos a crear una nueva columna que contenga la proporción de personas que reciben subsidio o ayuda social en el hogar.
# Para esto, vamos a dividir la cantidad de personas que reciben subsidio o ayuda social en el hogar por la cantidad total de personas en el hogar.
base_final['prop_subsidio'] = base_final['v5'] / base_final['ix_tot']

# Ahora vamos a crear una nueva columna que contenga la proporción de personas que viven de seguro de desempleo en el hogar.
# Para esto, vamos a dividir la cantidad de personas que viven de seguro de desempleo en el hogar por la cantidad total de personas en el hogar.
base_final['prop_desempleo'] = base_final['v4'] / base_final['ix_tot']

# %% [markdown]
# 5. Presenten estadísticas descriptivas de tres variables de la encuesta de hogar que ustedes creen que pueden ser relevantes para predecir la desocupación. Comenten las estadísticas obtenidas.

# %%
# Análisis descriptivo de las variables provenientes de las bases de hogares
variables_hogar = ['itf', 'ix_men10', 'v4', 'ix_tot', 'v5', 'ii7']

# Estadísticas descriptivas
descriptive_stats = base_final[variables_hogar].describe()
print(descriptive_stats)

# Visualización de la distribución de las variables
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()

for i, var in enumerate(variables_hogar):
    sns.histplot(base_final[var], kde=True, ax=axes[i])
    axes[i].set_title(f'Distribución de {var}')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Frecuencia')

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Parte II: Clasificación y regularización

# %% [markdown]
# 1. Para cada año, partan la base respondieron en una base de prueba y una de entrenamiento (X_train, y_train, X_test, y_test) utilizando
# el comando train_test_split. La base de entrenamiento debe comprender el 70% de los datos, y la semilla a utilizar (random state
# instance) debe ser 101. Establezca a desocupado como su variable dependiente en la base de entrenamiento (vector y). El resto de las variables serán las variables independientes (matriz X). Recuerden agregar la columna de unos (1).

# %% [markdown]
# 

# %% [markdown]
# Antes de hacer esto, sería importante considerar el feedback que nos dejó Nacho.
# Parte II:
# #### Creo que, si bien hicieron unas variables que tienen sentido ordinal, para los modelos hubiera creado dummies para las variables categóricas. Es más seguro.
# - Habría que preguntarle si crear dummies para absolutamente todas las variables categóricas que tiene la base final.
# #### Hay algún problema con la convergencia de la regresión logística, no termino de entender por qué.
# - Si ahora la parte de preprocesamiento de datos está bien, esto no debería volver a pasar. 
# #### Me gustan mucho las figuras de ROC de todos los modelos juntos, pero por algún motivo que no comprendo les dieron valores de Accuracy y AUC bastante peores que sus compañeras. Igual se los consideré bien porque los modelos están bien hechos (salvo el tema de las dummies).
# - Idem al anterior
# #### Definitivamente no hubiera utilizado la variable PEA para el entrenamiento de los modelos. Piensen que para crearla utilizaron la variable estado.
# - Tener en mente la posibilidad de sacar las variables madres de otras variables que creemos. Si podemos pensar entre todas que variables que podamos crear nos evitaría tener tantas variables categóricas, como en el caso de PEA, nos ahorraría tiempo.
# #### Sacar la variable estado de la base final

# %%
import statsmodels.api as sm     

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
#from sklearn.metrics import plot_roc_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler



# %%
# Agregamos una columna llamada desocupado a las bases de cada año y que tome 1 si esta desocupada
base_2004['desocupado'] = base_2004['estado'].apply(lambda x: 1 if x == 2 else 0)
base_2024['desocupado'] = base_2024['estado'].apply(lambda x: 1 if x == 2 else 0)


# %%
# Entrenaremos con el 70% de la base de datos del 2004 y el resto se usarán para testear 
# el modelo obtenido
# Split the data into training and testing sets
train2004, test2004 = train_test_split(base_2004, test_size=0.3, random_state=101)
# Define the dependent variable y and the independent variables X
y_train_2004 = train2004['desocupado']
X_train_2004 = train2004.drop(columns=['desocupado'])

y_test_2004 = test2004['desocupado']
X_test_2004 = test2004.drop(columns=['desocupado'])

# Add a column of ones to X_train and X_test
X_train_2004 = X_train_2004.assign(intercept=1)
X_test_2004 = X_test_2004.assign(intercept=1)

# Repito lo mismo para el 2024
train2024, test2024 = train_test_split(base_2024, test_size=0.3, random_state=101)
# Define the dependent variable y and the independent variables X
y_test_2024 = test2024['desocupado']
X_test_2024 = test2024.drop(columns=['desocupado'])

y_train_2024 = train2024['desocupado']
X_train_2024 = train2024.drop(columns=['desocupado'])

# %%
# Revisamos cuantas observaciones quedaron para Test y cuantas para Entrenamiento.
print(f'El conjunto de entrenamiento del 2004 tiene {len(X_train_2004)} observaciones.')
print(f'El conjunto de test del 2004 tiene {len(X_test_2004)} observaciones.')
print(f'El conjunto de entrenamiento del 2024 tiene {len(X_train_2024)} observaciones.')
print(f'El conjunto de test del 2024 tiene {len(X_test_2024)} observaciones.')

# %% [markdown]
# Expliquen brevemente cómo elegirían λ por validación cruzada (en Python es alpha). Detallen por qué no usarían el conjunto de prueba (test) para su elección

# %% [markdown]
# En validación cruzada, ¿cuáles son las implicancias de usar un k muy
# pequeño o uno muy grande? Cuando k = n (con n el número de
# muestras), ¿cuántas veces se estima el modelo?

# %% [markdown]
# Para regresión logística, implementen la penalidad, L1 como la de
# LASSO y L2 como la de Ridge con λ = 1 (como en la Tutorial 10), usando
# la opción penalty y reporten la matriz de confusión, la curva ROC, los
# valores de AUC y de Accuracy para cada año.1 ¿Cómo cambiaron los
# resultados con respecto al TP3? ¿La performance de regresión logística
# con regularización es mejor o peor?

# %%
#antes de hacer la regresión, necesitamos estandarizar las variables
# Estadisticas antes de estandarizar, base 2004
X_train_2004.describe().T

# %%
#voy a transformar la columna "codusu" en 0 porque es string y cuando corro la standarización me da error
X_train_2024['codusu'] = 0
X_test_2024['codusu'] = 0

# %%
# Estadisticas antes de estandarizar
X_train_2024.describe().T


# %%
#primero para 2004
#  Iniciamos el Standard Scaler
sc = StandardScaler()

# Estandarizamos las observaciones de entrenamiento
X_train_2004_transformed = pd.DataFrame(sc.fit_transform(X_train_2004), index=X_train_2004.index, columns=X_train_2004.columns)

# Estandarizamos las observaciones de test
X_test_2004_transformed = pd.DataFrame(sc.transform(X_test_2004), index=X_test_2004.index, columns=X_test_2004.columns)

# Estadisticas luego de estandarizar
X_test_2004_transformed.describe().T


# %%
#ahora para 2024
#  Iniciamos el Standard Scaler
sc = StandardScaler()

# Estandarizamos las observaciones de entrenamiento
X_train_2024_transformed = pd.DataFrame(sc.fit_transform(X_train_2024), index=X_train_2024.index, columns=X_train_2024.columns)

# Estandarizamos las observaciones de test
X_test_2024_transformed = pd.DataFrame(sc.transform(X_test_2024), index=X_test_2024.index, columns=X_test_2024.columns)

# Estadisticas luego de estandarizar
X_test_2024_transformed.describe().T


# %% [markdown]
# Ahora si, comenzamos con las regresiones

# %%
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
# Ridge para 2004
# prueba con alpha = 1
alpha = 1
print("Alpha:", alpha)

ridge_a1_2004 = Ridge(alpha = alpha)
ridge_a1_2004.fit(X_train_2004_transformed, y_train_2004)             
pred_a1_2004 = ridge_a1_2004.predict(X_test_2004_transformed)
ecm_a1_2004 = mean_squared_error(y_test_2004, pred_a1_2004)

print("Error cuadrático medio: ", ecm_a1_2004)   
print("Coeficientes del modelo:")
print(pd.Series(ridge_a1_2004.coef_, index = X_train_2004_transformed.columns)) 

# %%
# prueba con alpha = 1
alpha = 5
print("Alpha:", alpha)

ridge_a5_2004 = Ridge(alpha = alpha)
ridge_a5_2004.fit(X_train_2004_transformed, y_train_2004)             
pred_a5_2004 = ridge_a5_2004.predict(X_test_2004_transformed)
ecm_a5_2004 = mean_squared_error(y_test_2004, pred_a5_2004)

print("Error cuadrático medio: ", ecm_a5_2004)   
print("Coeficientes del modelo:")
print(pd.Series(ridge_a5_2004.coef_, index = X_train_2004_transformed.columns)) 


