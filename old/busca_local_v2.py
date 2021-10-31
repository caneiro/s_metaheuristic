import numpy as np
import pandas as pd
import time
import sys

# Import o motor de calculo do premio
import engine

SEED = 42
DEBUG = True
np.random.seed(SEED)

# Carrega tabela de parametros dos indicadores
parametros = pd.read_csv('parametros_indicadores_edp.csv', sep=';', decimal=',')
# print(parametros.head())

########################## BUSCA LOCAL ##################################

def gera_vizinhos(s, k, a):
    """
    s <- solucao atual
    k <- k vizinhos
    """
    vizinhos = {}
    s1 = s.copy() # <<<<<<<< vetorizar!!!
    # Cada k vizinho
    for k_ in range(k):
        # Cada indicador
        for i in s1.keys():
            # Gera uma perturbacao randomica em torno da normal
            if a is None:
                s1[i] = np.random.normal(s[i])
            else:
                s1[i] = s[i] + (np.random.normal(0) * a * s[i])
        # Copia a solucao para o conjunto
        vizinhos[k_] = s1.copy()
    # Retorna conjunto de vizinhos
    return vizinhos

def valida_solucoes(vizinhos):
    solucao_valida = np.zeros(len(vizinhos))
    for k in vizinhos:
        valores = np.array(list(vizinhos[k].values()))
        minimos = parametros['min'].values
        maximos = parametros['max'].values
        solucao_valida[k] = all((valores >= minimos) & (valores <= maximos)) == True
    idx_validos = np.where(solucao_valida==1)[0]
    qtd_validos = solucao_valida.sum()
    print('solucoes validas:', qtd_validos)
    vizinhos_validos = {key: vizinhos[idx] for idx, key in zip(idx_validos, range(int(qtd_validos)))}
    return vizinhos_validos

def verifica_solucao(df, s, empresa):
    pontos = engine.fn_99_premio_geral(df=df, edp=s)
    return pontos[empresa]

def escolha_solucao(df, s0, vizinhos, tipo='max', strategy='best'):
    pontos_s0 = verifica_solucao(df, s0, 'EDP SP')
    s_best = s0.copy()
    pts_best = pontos_s0
    improve=False
    pontos_k = []

    if strategy=='best':
        for k in vizinhos.keys():
            pts = verifica_solucao(df, vizinhos[k], 'EDP SP')
            pontos_k.append(pts)
            if DEBUG:
                print(k, pts)
        pontos_k = np.array(pontos_k)
        
        if tipo=='max':
            idx_best_k = np.argsort(pontos_k)[-1]
            if pontos_k[idx_best_k] > pontos_s0:
                improve=True
                s_best = vizinhos[idx_best_k]
                pts_best = pontos_k[idx_best_k]
        else:
            idx_best_k = np.argsort(pontos_k)[0]
            if pontos_k[idx_best_k] < pontos_s0:
                improve=True
                s_best = vizinhos[idx_best_k]
                pts_best = pontos_k[idx_best_k]

    elif strategy=='first':
        for k in vizinhos.keys():
            pts = verifica_solucao(df, vizinhos[k], 'EDP SP')
            if DEBUG:
                print(k, pts)
            if tipo=='max':
                if pts > pontos_s0:
                    improve=True
                    s_best = vizinhos[k]
                    pts_best = pts
                    break
            else:
                if pts < pontos_s0:
                    improve=True
                    s_best = vizinhos[k]
                    pts_best = pts
                    break

    elif strategy=='random':
        for k in vizinhos.keys():
            pts = verifica_solucao(df, vizinhos[k], 'EDP SP')
            pontos_k.append(pts)
            if DEBUG:
                print(k, pts)
        pontos_k = np.array(pontos_k)
        
        if tipo=='max':
            idx_improve_k = np.where(pontos_k>pontos_s0)[0]
            qtd_improve_k = len(idx_improve_k)
            if qtd_improve_k>=1:
                # print('qtd_random', qtd_improve_k)
                idx_random = np.random.choice(idx_improve_k, 1)
                improve=True
                s_best = vizinhos[idx_random[0]]
                pts_best = pontos_k[idx_random[0]]
                if DEBUG:
                    print(k, pts)
        else:
            idx_improve_k = np.where(pontos_k<pontos_s0)[0]
            qtd_improve_k = len(idx_improve_k)
            if qtd_improve_k>=1:
                # print('qtd_random', qtd_improve_k)
                idx_random = np.random.choice(idx_improve_k, 1)
                improve=True
                s_best = vizinhos[idx_random[0]]
                pts_best = pontos_k[idx_random[0]]

    return s_best, pts_best, improve

def busca_local(df, s_init, strategy='best', tipo='max', k=100, i=1000, a=0.01, tag=None):
    # Copia solucao inicial para solucao atual
    s_atual = s_init.copy()

    # Inicializacao de variaveis de controle e log
    pts_hist = []
    improve_hist = []
    vizinhos_validos_hist = []
    tempo_execucao_hist = []
    for i_ in range(i):
        start_time = time.time()

        # Gera vizinhanca e verifica os vizinhos validos
        vizinhos = gera_vizinhos(s_atual, k, a)
        vizinhos_validos = valida_solucoes(vizinhos)
        # Verifica se restou vizinhos validos apos validacao
        if len(vizinhos_validos) == 0:
            improve=False     
        else:
            # print(vizinhos_validos)
            s_best, pts_best, improve = escolha_solucao(df, s_atual, vizinhos_validos, tipo, strategy)
            if improve:
                s_atual = s_best.copy()
            else:
                improve=False
        # Logs
        pts_hist.append(pts_best)
        improve_hist.append(improve)
        vizinhos_validos_hist.append(len(vizinhos_validos))
        end_time = time.time()
        tempo_execucao = round(end_time - start_time, 2)
        tempo_execucao_hist.append(tempo_execucao)
        print(i_, '>>>> improve', improve, 
                  '>>>> pts_best', pts_best,
                  '>>>> duracao', tempo_execucao, 'segundos')
        print('--------------------------------------------------')

    # Grava historico de rodadas
    hist = pd.DataFrame(
        {'pts':pts_hist,
         'improve':improve_hist,
         'qtd_vizinhos_validos':vizinhos_validos_hist,
         'tempo_execucao': tempo_execucao_hist})
    hist.to_csv('./log/busca_local_hist_' + tag + '.csv', index=False)

    return s_best, pts_best, improve

#########################################################################

def main(strategy, tipo, K, I, a, tag):
    start_time = time.time()
    # Carrega dados das empresas
    df = engine.load_data()

    # Apenas referencia inicial
    s_init = dict(zip(parametros['indicador'].values, parametros['valor'].values))
    pts_init = verifica_solucao(df, s_init, 'EDP SP')
    print('\n', 's_init', s_init, '\n', '>>> pts_init', pts_init)
    print()

    # Aplica a busca local
    s_best, pts_best, improve = busca_local(df=df,
                                            s_init=s_init, 
                                            strategy=strategy,
                                            tipo=tipo,
                                            k=K,
                                            i=I,
                                            a=a,
                                            tag=tag)

    print('\n', 's_best', s_init, '\n', '>>> pts_init', pts_init)
    print('\n', 's_best', s_best, '\n', '>>> pts_best', pts_best)
    print('\n', 'improve', improve, '\n')

    # Resumo
    df = pd.DataFrame.from_dict(s_init, orient='index', columns=['init'])
    df['best'] = pd.DataFrame.from_dict(s_best, orient='index', columns=['init'])
    df['%'] = df.best / df.init * 100
    print(df)

    end_time = time.time()
    tempo_execucao = round(end_time - start_time, 2)
    print('\n', 'tempo de execução total', tempo_execucao, 'segundos')
 

if __name__ == "__main__":

    # Verifica se foram passados parametros de comando de linha
    if len(sys.argv) > 1:
        strategy = sys.argv[1]
        tipo = sys.argv[2]
        K =  int(sys.argv[3])
        I = int(sys.argv[4])
        alpha = float(sys.argv[5])
        tag = strategy + '_alpha_' + str(alpha) + '_k_' + str(K) + '_i_' + str(I) + '_teste'
        print(strategy, tipo, K, I, alpha, tag)
        main(strategy, tipo, K, I, alpha, tag)
    else:
        alphas = [0.02]
        strategies = ['first']
        k_vizinhos = [10]
        for s in strategies:
            for a in alphas:
                for k in k_vizinhos:
                    strategy = s
                    alpha = a
                    tipo = 'max'
                    K = k
                    I = 10
                    tag = strategy + '_alpha_' + str(alpha) + '_k_' + str(K) + '_i_' + str(I) + '_'
                    
                    main(strategy, tipo, K, I, alpha, tag)

                    



# alphas = [0.01, 0.02, 0.03]
# strategies = ['best', 'first', 'random']
# k_vizinhos = [10, 30, 100]
# for s in strategies:
#     for a in alphas:
#         for k in k_vizinhos:
#             strategy = s
#             alpha = a
#             tipo = 'max'
#             K = k
#             I = 100
#             tag = strategy + '_alpha_' + str(alpha) + '_k_' + str(K) + '_i_' + str(I) + '_'
            
#             main(strategy, tipo, K, I, alpha, tag)