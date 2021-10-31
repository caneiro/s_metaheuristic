import numpy as np
import pandas as pd

# Import o motor de calculo do premio
import engine

SEED = 42
np.random.seed(SEED)

# Carrega tabela de parametros dos indicadores
parametros = pd.read_csv('parametros_indicadores_edp.csv', sep=';', decimal=',')
# print(parametros.head())

########################## BUSCA LOCAL ##################################

### Geracao da vizinhanca
def gera_neighboors(s, k):
    """
    s <- solucao atual
    k <- k vizinhos
    """
    neighboors = {}
    s1 = s.copy() # <<<<<<<< vetorizar!!!
    # Cada k vizinho
    for k_ in range(k):
        # Cada indicador
        for i in s1.keys():
            # Gera uma perturbacao randomica em torno da normal
            s1[i] = np.random.normal(s[i])
        # Copia a solucao para o conjunto
        neighboors[k_] = s1.copy()
    # Retorna conjunto de vizinhos
    return neighboors

### Verifica vizinhanca valida (min - max) parametros
##### IMPLEMENTAR ####

def verifica_solucao(s, empresa):
    pontos = engine.fn_99_premio_geral(s)
    return pontos[empresa]

def escolha_solucao(s0, neighboors, tipo='max', strategy='best'):
    pontos_s0 = verifica_solucao(s0, 'EDP SP')
    improve=False
    if strategy=='best':
        pontos_k = []
        for k in neighboors.keys():
            pts = verifica_solucao(neighboors[k], 'EDP SP')
            pontos_k.append(pts)
            print(k, pts)
        pontos_k = np.array(pontos_k)
        
        if tipo=='max':
            idx_best_k = np.argsort(pontos_k)[-1]
            if pontos_k[idx_best_k] > pontos_s0:
                improve=True
                s_best = neighboors[idx_best_k]
                pts_best = pontos_k[idx_best_k]
            else:
                s_best = s0
                pts_best = pontos_s0
        else:
            idx_best_k = np.argsort(pontos_k)[0]
            if pontos_k[idx_best_k] < pontos_s0:
                improve=True
                s_best = neighboors[idx_best_k]
            else:
                s_best = s0
    return s_best, pts_best, improve

def busca_local(s0, k):

    neighboors = gera_neighboors(s0, k)

    # for i_ in range(i):
    s_best, pts_best, improve = escolha_solucao(s0, neighboors, 'max', 'best')

    return s_best, pts_best, improve




### Parametros da BUSCA LOCAL
K = 100 # k vizihos
# I = 1000 # chamadas de funcao obj
strategy = 'best' # best / first / random
# a = 0.001 # alpha -> tamanho da perturbacao
### Solucao inicial -> 'Premio Atual'


s_init = dict(zip(parametros['indicador'].values, parametros['valor'].values))
pts_init = verifica_solucao(s_init, 'EDP SP')
print('\n', 's_best', s_init, '\n', '>>> pts_init', pts_init)
print()



s_best, pts_best, improve = busca_local(s_init, K)
print('\n', 's_best', s_init, '\n', '>>> pts_init', pts_init)
print('\n', 's_best', s_best, '\n', '>>> pts_best', pts_best)
print('\n', 'improve', improve, '\n')


df = pd.DataFrame.from_dict(s_init, orient='index', columns=['init'])
df['best'] = pd.DataFrame.from_dict(s_best, orient='index', columns=['init'])
df['%'] = df.best / df.init * 100
print(df)             
            






# s = s0 ; /∗ Generate an initial solution s0
# ∗/
# While not Termination Criterion Do
# Generate (N(s)) ; /∗ Generation of candidate neighbors ∗/
# If there is no better neighbor Then Stop ;
# s = s′ ; /∗ Select a better neighbor s′ ∈ N(s) ∗/
# Endwhile
# Output Final solution found (local optima).
