# %%
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from yellowbrick.regressor import prediction_error
from yellowbrick.regressor import ResidualsPlot
from sklearn.model_selection import KFold, cross_validate
from yellowbrick.model_selection import FeatureImportances
from sklearn.model_selection import GridSearchCV
import pickle

base = pd.read_csv('C:/Users/mathg/OneDrive/Documents/Diretório/Projetos no VSCode/Alura/IA aumentada/flights.csv')

# %% [markdown]
# # Análise exploratória dos dados

# %%
base.describe()

# %%
base.describe(include='O')

# %%
base.info()

# %%
base.head(5)

# %%
media_atraso = base.groupby('airline')['delay'].mean().reset_index()
sns.barplot(x='airline', y='delay', data=media_atraso, color='green')
plt.title('atraso médio de voô para cada companhia aérea')
plt.xlabel('companhia aérea')
plt.ylabel('média de minutos de atraso')
plt.show()

sns.countplot(data=base, x='airline')
plt.title('Contagem de observações por companhia aérea')
plt.xlabel('companhia aérea')
plt.ylabel('total de instâncias')
plt.show()

# %%
media_atraso_schengen = base.groupby('schengen')['delay'].mean().reset_index()
sns.barplot(x='schengen', y='delay', data=media_atraso, color='green')
plt.title('atraso médio por pertencimento ou não no schengen')
plt.xlabel('tipo de voo')
plt.ylabel('média de minutos de atraso')
plt.show()

sns.countplot(data=base, x='schengen')
plt.title('total de instâncias por pertencimento ou não no schengen')
plt.xlabel('tipo de voo')
plt.ylabel('total de instâncias')
plt.show()

# %%
media_atraso_feriado = base.groupby('is_holiday')['delay'].mean().reset_index()
sns.barplot(x='is_holiday', y='delay', data=media_atraso_feriado)
plt.title('Atraso em feriados')
plt.xlabel('feriado')
plt.ylabel('media de minutos de atraso')
plt.show()

# %%

sns.countplot(data=media_atraso_feriado, x='is_holiday')
plt.title('Atraso em feriados')
plt.xlabel('feriado')
plt.ylabel('media de minutos de atraso')
plt.show()

# %%
ordem = base['aircraft_type'].value_counts().index
media_atraso_aeronave = base.groupby('aircraft_type')['delay'].mean().reset_index()
sns.barplot(x='aircraft_type', y='delay', data=media_atraso_aeronave, palette='Set2', order=ordem)
plt.xticks(rotation=70)
plt.title('media atraso de voo por aeronave')
plt.xlabel('tipo aeronave')
plt.ylabel('media atraso')

# %%
# usando a regra de Freedman-Diaconis para estabelecer a largura das barras 
# a regra é: largura é igual a 2 vezes o intervalo interquartil dos dados sobre raiz cúbica de n 

import numpy as np
def definir_largura (coluna):
    Q25, Q75 = np.percentile(base[coluna], [25, 75])
    IQR = Q75 - Q25

    largura_barra = 2 * IQR * np.power(len(base[coluna]), -1/3)

    return largura_barra


barra_arrival = definir_largura("arrival_time")


sns.histplot(data=base, x='arrival_time', kde='true', binwidth=barra_arrival)
plt.title('Histograma de voôs ao longo do dia')
plt.xlabel('hora do dia')
plt.ylabel('total de voôs')


# %%
# usando a regra de Freedman-Diaconis para estabelecer a largura das barras 
# a regra é: largura é igual a 2 vezes o intervalo interquartil dos dados sobre raiz cúbica de n 

import numpy as np
def definir_largura (coluna):
    Q25, Q75 = np.percentile(base[coluna], [25, 75])
    IQR = Q75 - Q25

    largura_barra = 2 * IQR * np.power(len(base[coluna]), -1/3)

    return largura_barra

barra_departure = definir_largura("departure_time")



sns.histplot(data=base, x='arrival_time', kde='true', binwidth=barra_departure)
plt.title('Histograma de voôs ao longo do dia')
plt.xlabel('hora do dia')
plt.ylabel('total de voôs')


# %% [markdown]
# ## Verificando a distribuição da variável target

# %%
media_atraso = base['delay'].mean()
mediana_atraso = base['delay'].median()

fig, axes = plt.subplots(1,2, figsize=(9,4))

sns.boxplot(data=base, y='delay', ax=axes[0])
axes[0].set_title('boxplot')
axes[0].axhline(y=media_atraso, color='red', linestyle='--', label='média')
axes[0].legend()


largura_barra_atraso = definir_largura('delay')
sns.histplot(data=base, x='delay', ax=axes[1], binwidth=largura_barra_atraso)
axes[1].set_title('histograma')
axes[1].axvline(x=media_atraso, color='red', linestyle='--', label='média')
axes[1].axvline(x=mediana_atraso, color='purple', linestyle='--', label='mediana')
axes[1].legend()

# %%
base.head(1)

# %% [markdown]
# # Feature engineering

# %%
# criar uma coluna de data

base['year'].astype(str)
(base['day']+1).astype(str) # tirando o dia 0 e començando por 1
base['year'].astype(str) + '-' + (base['day']+1).astype(str)
base['data'] = pd.to_datetime(base['year'].astype(str) + '-' + (base['day']+1).astype(str), format='%Y-%j')

# coluna pro final de semana
base['fds'] = base['data'].dt.weekday.isin(base['data']) # 


base.nunique()  # ver o total valores únicos
# criando dummies e associando números para as categorias 

base['schengen'].unique()
base['schengen'] = base['schengen'].replace({'non-schengen': 0, 'schengen': 1 })
base['is_holiday'].unique()
base['schengen'] = base['schengen'].replace({False: 0, True: 1 })
base['is_holiday'].unique()
variáveis_categóricas = ['airline', 'aircraft_type','origin','airline']
base_dummieficada = pd.get_dummies(data=base, columns=variáveis_categóricas, dtype=int)



# excluindo variáveis

# verifiquei no gráfico que o horário de chegada e partida pode ter uma correlação. Como pretendo fazer um modelo de regressão, ter variáveis correlacionadas não é muito interessante. 

base_dummieficada[['arrival_time', 'departure_time']].corr()    # confirma a correlação

baseFinal = base_dummieficada.drop(["flight_id", "departure_time", "day", "year", "data"], axis=1)


# %% [markdown]
# # treinando modelos

# %%
# separando

X = baseFinal.drop(['delay'], axis=1)
Y = baseFinal['delay']

X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, random_state=42)

# %%
# irei treinar um modelo simples para baseline e depois poder comparar com modelos mais complexos.

modelo_dummie = DummyRegressor(strategy='mean' )
modelo_dummie.fit(X_treino, Y_treino)   # treinando 

Y_prev_dummie = modelo_dummie.predict(X_teste)  # rodando no conjunto de teste

def calcular_metricas (dados_teste, dados_prev):
    RMSE = mean_squared_error(dados_teste, dados_prev, squared=False)
    MAE = mean_absolute_error(dados_teste, dados_prev)
    R2 = r2_score(dados_teste, dados_prev)

    metricas = {
        'Raiz do erro quadrático médio': round(RMSE, 4),
        'Erro absoluto médio': round(MAE, 4),
        "R2 score": round(R2, 4)
    }

    return metricas

calcular_metricas(Y_teste, Y_prev_dummie)


# %%
# treinando um random forest regressor

modelo_rf = RandomForestRegressor(max_depth=5, random_state=42)
modelo_rf.fit(X_treino, Y_treino)

Y_prev_rf = modelo_rf.predict(X_teste)  
calcular_metricas(Y_teste, Y_prev_rf)

# vemos que ta melhor que o modelo baseline


# agora vamos ver graficamente algumas propriedades estatísticas desse modelo treinado 



# %%
# ver graficamente a diferença entre previsão e real

visualizer_prev = prediction_error(modelo_rf, X_teste, Y_teste, title='diferença entre o previso e o real')

# %%
# ver o erro a distribuição dos resíduos

visualizer_res = ResidualsPlot(modelo_rf, title="resíduos do modelo e sua distribuição em histograma")

visualizer_res.fit(X_treino, Y_treino)  # Fit the training data to the visualizer_res
visualizer_res.score(X_teste, Y_teste)  # Evaluate the model on the test data
visualizer_res.show()  

# %%
# metendo o cross validation 

métricas = {
    'mae': 'neg_median_absolute_error',
    'rmse': 'neg_root_mean_squared_error',
    'r2': 'r2'
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
resultados_cv = cross_validate(modelo_rf, X_treino, Y_treino, cv=cv, scoring=métricas)

for métrica in métricas.keys():
    scores = resultados_cv[f'test_{métrica}']
    media_score = '{:.3f}'.format(scores.mean())
    desvpad_score = '{:.3f}'.format(scores.std())

    print(f"{métrica.upper()} Scores: {[f' {val:.3f}' for val in scores]}")
    print(f"{métrica.upper()} Média: {media_score}, Std: {desvpad_score}")
    print('----------------------------------------------------------------------------------')

# %% [markdown]
# # identificando as variáveis mais relevantes 

# %%
var_relevantes = FeatureImportances(modelo_rf, relative=False, topn=10)
var_relevantes.fit(X_treino, Y_treino)
var_relevantes.show()

# %%
# criando um dataframe e atribuindo essas importâncias de cada variável

importances = modelo_rf.feature_importances_
feature_importances = pd.DataFrame({'variavel': X_treino.columns, 'importancia': importances})
feature_importances.sort_values('importancia', ascending=False)

# %%
# rodando o modelo com as variáveis mais relevantes pra ver se muda alguma coisa

resultados_df = pd.DataFrame(index=['RMSE', 'MAE', 'R2'])
modelo_features_selecionada = RandomForestRegressor(random_state=42, max_depth=5)

for contagem in [1,5,10,15,20,25,30]: 
    selected_features  = feature_importances['variavel'].values[:contagem]
    X_treino_selecionado = X_treino[selected_features] 
    X_teste_selecionado = X_teste[selected_features]

    modelo_features_selecionada.fit(X_treino_selecionado, Y_treino)

    Y_pred = modelo_features_selecionada.predict(X_teste_selecionado)

    métricas = calcular_metricas(Y_teste, Y_pred)

    resultados_df[contagem] = list(métricas.values())
resultados_df

# percebos que de colocar mais de 15 variáveis o modelo tem uma melhora muito insignifante. Então podemos tirar essas variáveis para uma melhor eficiência.

# %%
# vendo onde o modelo se torna obsoleto

resultados_df = pd.DataFrame(index=['RMSE', 'MAE', 'R2'])
modelo_features_selecionada = RandomForestRegressor(random_state=42, max_depth=5)

for contagem in range(10, 15): 
    selected_features  = feature_importances['variavel'].values[:contagem]
    X_treino_selecionado = X_treino[selected_features] 
    X_teste_selecionado = X_teste[selected_features]

    modelo_features_selecionada.fit(X_treino_selecionado, Y_treino)

    Y_pred = modelo_features_selecionada.predict(X_teste_selecionado)

    métricas = calcular_metricas(Y_teste, Y_pred)

    resultados_df[contagem] = list(métricas.values())
resultados_df

# com 13 variáveis é o suficiente!

# %%
selected_features = feature_importances['variavel'].values[:13]
X_filtrado = X[selected_features]

X_treino, X_teste, Y_treino, Y_teste = train_test_split(X_filtrado, Y, random_state=42)

# %% [markdown]
# # otimizando os hiperparâmetros com gridsearch

# %%
cv = KFold(n_splits=5, shuffle=True, random_state=42)

parametros = {
'max_depth': [5, 10, 15, 20, 25],
'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7],
'min_samples_split': [2, 4, 6, 8],
'n_estimators': [100, 150, 200, 250]
}

modelo_otimizado = GridSearchCV(modelo_features_selecionada, param_grid=parametros, scoring="r2", cv=cv, n_jobs=-1) #n_jobs = -1 serve pra usar todos os núcleos do processador

modelo_otimizado.fit(X_treino, Y_treino)

# %%
modelo_otimizado.best_params_   # obter os melhores hiperparâmetros encontrado 

y_prev_modelo_grid = modelo_otimizado.predict(X_teste)

metricas_model_grid = calcular_metricas(Y_teste, y_prev_modelo_grid)

metricas_model_grid # verificar as métricas

resultados_df['modelo_grid'] = list(metricas_model_grid.values())
resultados_df   # comparar com os modelos separados pelas features

# %% [markdown]
# # salvando o modelo

# %%
try:
    with open('model_producao.pkl', 'wb') as file: 
        pickle.dump(modelo_otimizado.best_estimator_, file)
        print('Modelo salvo com sucesso!!!')
except Exception as e:
    print ('Ocorreu um erro ao salvar o modelo: ', str(e))


