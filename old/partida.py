import numpy as np
import pandas as pd

# Import o motor de calculo do premio
import engine

SEED = 42
np.random.seed(SEED)

# Carrega dados das empresas
df = engine.load_data()

# Pontuacao atual - antes da otimizacao
pontos_atual = engine.fn_99_premio_geral(df, debug=True)

# Carrega tabela de parametros dos indicadores
parametros = pd.read_csv('parametros_indicadores_edp_partida.csv', sep=';', decimal=',')
print(parametros.head())

# Teste Resultado com valores minimos
edp = dict(zip(parametros['indicador'].values,
               parametros['min'].values))
pontos_min = engine.fn_99_premio_geral(df=df, edp=edp, debug=True)

# Teste Resultado com valores maximo
edp = dict(zip(parametros['indicador'].values,
               parametros['max'].values))
pontos_max = engine.fn_99_premio_geral(df=df, edp=edp, debug=True)

print('\n','Premio Atual: ', pontos_atual['EDP SP'])
print('\n','Premio Valor Min: ', pontos_min['EDP SP'])
print('\n','Premio Valor Min: ', pontos_max['EDP SP'])

print('\n','---------------------Pontos Atual-----------------------')
df = pd.DataFrame.from_dict(pontos_atual, orient='index', columns=['valor'])
print(df.sort_values('valor', ascending=False).reset_index(drop=False))

print('\n','---------------------Pontos Min-------------------------')
df = pd.DataFrame.from_dict(pontos_min, orient='index', columns=['valor'])
print(df.sort_values('valor', ascending=False).reset_index(drop=False))

print('\n','---------------------Pontos Max-------------------------')
df = pd.DataFrame.from_dict(pontos_max, orient='index', columns=['valor'])
print(df.sort_values('valor', ascending=False).reset_index(drop=False))