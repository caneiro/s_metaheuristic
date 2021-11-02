##### LOCAL SEARCH #####

##### ITA | PG-CTE-S | TE-282 - Meta-heurísticas
##### Professor Dr. Angelo Passaro
##### Aluno: Rafael Caneiro de Oliveira
##### Versao: 0.1
##### Data: 22/10/2021

# conjunto de n ativos A = {a1, ..., an}
# possuem retorno = {r1, ..., rn}
# o portfolio é um vetor X = {x1, ..., xn} sendo xi a fraçao do ativo
# 0 <= xi <= 1 e Soma(xn) = 1
# restricoes de cardinalidade -> kmin e kmax ativos no portfolio
# restricoes de quantidade (fracao) de cada asset ->  dmin e dmax

import itertools
import numpy as np
import pandas as pd
import time

np.set_printoptions(linewidth=100000)

from load_data import load_port_files
from itertools import combinations
import copy
from pathlib import Path
from tqdm.auto import tqdm
import ray
import random


SEED = 42
DEBUG = False
PATH = Path.cwd()
LOG_PATH = Path(PATH, "./data/log/")

def normalize(X):
    # normalização para soma de X = 1
    total_X = X.sum()
    fator = 1 / total_X
    
    return np.round(X * fator, 6)

def initial_solution(n_assets, k=None, k_min=1, k_max=30, d_min=0.01, d_max=0.99):
    """
    Gera a solução inicial de forma aleatória.
    n_assets <- integer: quantidade de ativos no portfolio.
    k <- integer: quantidade de ativos na solucao inicial
    s -> tuple of X e Z
        X -> array of float: quantidade do ativo (percential)
        Z -> array of int: 0 ou 1 para indicação do ativo pertence ou nao a solucao
    """
    # >>>> as vezes depois da normalizacao um ativo pode ficar abaixo ou acima de dmin e dmax!!!
    # inicializa vetores
    X = np.zeros(n_assets)
    Z = np.zeros(n_assets)

    k_min = min(k_min, n_assets)
    k_max = min(k_max, n_assets)

    # gera qtd de ativos a serem selecionados
    if k is None:
        r = np.random.randint(k_min, k_max+1, 1)[0]
    else:
        r = k

    z = np.random.choice(n_assets, r, False)
    # gera os vetores e ajusta para soma = 1
    Z[z] = 1
    X[z] = np.random.uniform(d_min, d_max, r)
    X = normalize(X)
    # retorna a solucao inicial S que é uma tupla de X e Z
    s = [X, Z]
    return s

def aux_move_random_selections(s, ops=None):
    _, Z = s
    # ativos pertencentes à solucao
    Z1 = np.where(Z==1)[0]
    # ativos nao pertencentes
    Z0 = np.where(Z==0)[0]
    # seleciona um ativo aleatoriamente
    ai = np.random.choice(Z1)
    # verifica q sobrou algum ativo fora da lista >>>> melhorar!!!
    if Z0.shape[0] > 0:
        aj = np.random.choice(Z0)
    else:
        aj = ai

    # sinal da operacao
    if ops is not None:
        op = np.random.choice(ops)
    else:
        op=None
    
    return ai, aj, op

def move_idR(s, alpha, d_min, d_max):
    """
    idR -> [i]ncrease, [d]ecrease, [R]eplace
    ai é um ativo que pertence a solucao atua S
    aj é um ativo que não pertene solucao atual S
    op é 1 ou 0, sendo 1 para aumento e 0 para reducao
    se sign = 1 xi = xi * (1+q) caso contrario xi = xi * (1-q)

    """
    # Obtem dados da solucao atual
    X, Z = s
    k_in = np.sum(Z)
    # Obtem ativos e operador aleatoriamente
    ai, aj, op = aux_move_random_selections(s, [0, 1])

    # ciclo de operacao para cada tipo de operador 'aumento' ou 'reducao'
    if op == 1:
        fator = 1 + alpha
        X[ai] = X[ai] * fator
        # verifica se o valor ultrapassou o maximo de quantidade permitido e realiza 
        # a substituicao pelo ativo aj que nao fazia parte da solucao
        if X[ai] > d_max:
            Z[ai] = 0
            Z[aj] = 1
            X[aj] = d_max
            
    else:
        fator = 1 - alpha
        X[ai] = X[ai] * fator
        if X[ai] < d_min:
            Z[ai] = 0
            Z[aj] = 1
            X[aj] = d_min

    k_out = np.sum(Z)
    # print('Move: idR | k_in: {} | k_out: {}'.format(k_in, k_out))
    X = normalize(X)
    sl = [X, Z]
    return sl

def move_idID(s, alpha, d_min, d_max):
    """
    idID -> [i]ncrease, [d]ecrease, [I]nsert, [D]elete
    ai é um ativo que pertence a solucao atua S
    aj é um ativo que não pertene solucao atual S
    op é 0, 1 ou 2, sendo 0 para reducao, 1 para aumento e 2 para inserção
    se sign = 1 xi = xi * (1+q) caso contrario xi = xi * (1-q)

    """
    # Obtem dados da solucao atual
    X, Z = s
    k_in = np.sum(Z)

    # Obtem ativos e operador aleatoriamente
    ai, aj, op = aux_move_random_selections(s, [0, 1, 2])

    # ciclo de operacao para cada tipo de operador 'aumento' ou 'reducao'
    if op == 1:
        fator = 1 + alpha
        X[ai] = max(X[ai] * fator, d_max)
            
    elif op == 0:
        fator = 1 - alpha
        X[ai] = X[ai] * fator
        if X[ai] < d_min:
            # delete
            Z[ai] = 0
            X[ai] = 0
    
    elif op == 2:
        Z[ai] = 0
        X[ai] = 0
        Z[aj] = 1
        X[aj] = d_min

    k_out = np.sum(Z)
    # print('Move: idID | k_in: {} | k_out: {}'.format(k_in, k_out))

    X = normalize(X)
    sl = [X, Z]
    return sl

def move_TID(s, alpha, d_min, d_max):
    """
    TID  -> [T]ransfer, [I]nsert, [D]elete
    ai é um ativo que pertence a solucao atua S
    aj é um ativo que não pertene solucao atual S
    """
    # Obtem dados da solucao atual
    X, Z = s
    k_in = np.sum(Z)

    # Obtem ativos e operador aleatoriamente
    ai, _, _ = aux_move_random_selections(s)
    aj = np.random.choice([j for j in range(X.shape[0]) if j != ai])
    trans = X[ai] * alpha
    # special case
    if X[ai] - trans < d_min:
        X[ai] = 0
        Z[ai] = 0
        # special case II
        if Z[aj] == 0:
            X[aj] = d_min
            Z[aj] = 1
        # special case I
        else:
            X[aj] = X[ai]
    # operacao normal
    else:
        X[ai] = X[ai] - trans
        X[aj] = X[aj] + trans
        Z[aj] = 1

    k_out = np.sum(Z)
    # print('Move: TID | k_in: {} | k_out: {}'.format(k_in, k_out))

    # X = normalize(X)
    sl = [X, Z]
    return sl
   
def neighbours(s, i, alpha, move='random', d_min=0.01, d_max=0.99):
    """
    s <- solucao atual
    """
    # inicializacao de variaveis
    N = []
    moves = {
        'iDR':  move_idR,
        'idID': move_idID,
        'TID':  move_TID
    }
    choices = list(moves.keys())
    # verifica qual tipo de geracao de vizinhos sera utilizada
    if move=='random':
        move=np.random.choice(choices)
    # atribui a funcao de move determinada
    move = moves[move]
    for _ in range(i):
        sl = copy.deepcopy(s)
        s1 = move(sl, alpha, d_min, d_max)
        N.append(s1)

    return N

def cost_function(s, cov_mx):
    X, Z = s
    Z1 = np.where(Z==1)[0]
    if len(Z1) == 1:
        obj = X[Z1]
    else:
        i, j = zip(*list(combinations(Z1, 2)))
        i = np.array(i)
        j = np.array(j)
        cov_ = cov_mx[i, j]
        obj = np.sum(cov_ * X[i] * X[j])
    return obj

def portfolio_return(s, r_mean):
    X, Z = s
    Z1 = np.where(Z==1)[0]
    p_return = np.sum(X[Z1] * r_mean[Z1])
    return p_return

def validation(N, exp_return, r_mean, k_min, k_max, d_min, d_max):
    # filtra vizinhos com quantidade de ativos válidas
    Nv = [sl for sl in N if np.sum(sl[1]) >= k_min and np.sum(sl[1]) <= k_max]
    # filtra vizinhos com proporcoes de ativos válidas
    Nv = [sl for sl in Nv if all(sl[0][np.where(sl[1]==1)] >= d_min) and all(sl[0][np.where(sl[1]==1)] <= d_max)]
    # filtra vizinhos com retorno maior que o retorno experado
    Nv = [sl for sl in Nv if portfolio_return(sl, r_mean) > exp_return]

    # print('Validation | N = {} | Nv = {}'.format(len(N), len(Nv)))
    return Nv

def selection(Nv, s_best, cov_mx, strategy='best', type='min'):

    if type == 'min':
        obj_best = cost_function(s_best, cov_mx)
        improve = False

        if strategy in ['best', 'random']:
            obj_sn = []
            for n in Nv:
                obj_sn.append(cost_function(n, cov_mx))
            obj_sn = np.array(obj_sn)

            if strategy == 'best':
                idx_selected = np.argsort(obj_sn)[0]
                obj_selected = obj_sn[idx_selected]
                if obj_selected < obj_best:
                    improve = True
                    s_best = Nv[idx_selected]
                    obj_best = obj_sn[idx_selected]
            
            elif strategy == 'random':
                idx_selected = np.where(obj_sn < obj_best)[0]
                if len(idx_selected) >= 1:
                    idx_selected = np.random.choice(idx_selected, 1)[0]
                    improve = True
                    s_best = Nv[idx_selected]
                    obj_best = obj_sn[idx_selected]
        
        elif strategy == 'first':
            for n in Nv:
                obj_sn = cost_function(n, cov_mx)
                if obj_sn < obj_best:
                    improve = True
                    s_best = n
                    obj_best = obj_sn

    return s_best, obj_best, improve

@ray.remote
def local_search(params):

    k_min = params[0]
    k_max = params[1]
    d_min = params[2]
    d_max = params[3]
    iter = params[4]
    neighs = params[5]
    alpha = params[6]
    exp_return = params[7]
    move_strategy = params[8]
    selection_strategy = params[9]
    n_port = params[10]
    seed = params[11]
    early_stoping=50
    tol=0.000001

    np.random.seed(seed)
    total_time_start = time.time()

    # obtem dados do portfolio
    portfolio = load_port_files(n_port)
    
    # obtem dados do portfolio
    n_assets, r_mean, r_std, cov_mx = portfolio
    
    # dados da solucao ininial aleatoria
    s0 = initial_solution(n_assets, None, k_min, k_max, d_min, d_max)
    
    # obj_s0 = cost_function(s0, cov_mx)
    # return_s0 = portfolio_return(s0, r_mean)
    
    # print('Solucao Inicial | obj: {} | return: {} | assets: {} | X:{}'\
    #     .format(round(obj_s0, 6),
    #             round(return_s0, 6),
    #             s0[1].sum(),
    #             s0[0].sum()))

    # listas para armazenar dados das iteracoes
    l_move = []
    l_improve = []
    l_obj = []
    l_return = []
    l_assets = []
    l_X = []
    l_Z = []
    l_qN = []
    l_qNv = []
    l_iter_time = []
    l_iter = []

    # contador para realizar o early stoping
    c_early = 0
    
    # copia solucao inicial para a solucao
    s_best = s0.copy()
    obj_best = cost_function(s0, cov_mx)
    
    # tipos de operadores a serem utilizados
    moves = ['iDR', 'idID', 'TID']

    for i in range(iter):
        l_iter.append(i)
        iter_time_start = time.time()

        if move_strategy == 'best':
            s_move = []
            obj_move = []

            for move in moves:
                N = neighbours(s_best, neighs, alpha, move, d_min, d_max)
                Nv = validation(N, exp_return, r_mean, k_min, k_max, d_min, d_max)
                if len(Nv) > 0:
                    s_select, obj_select, improve = selection(Nv, s_best.copy(), cov_mx, selection_strategy)
                    s_move.append(s_select)
                    obj_move.append(obj_select)

            if len(s_move) >= 1:
                obj_move = np.array(obj_move)
                min_obj_move = np.argsort(obj_move)[0]
                sl = s_move[min_obj_move]
                obj_l = obj_move[min_obj_move]
                move = moves[min_obj_move]
            else:
                obj_l = obj_best
                sl = s_best.copy()
                improve = False                

        else:
            if move_strategy == 'random':
                move = np.random.choice(moves)

            elif move_strategy in moves:
                move = move_strategy

            N = neighbours(s_best, neighs, alpha, move, d_min, d_max)
            Nv = validation(N, exp_return, r_mean, k_min, k_max, d_min, d_max)
            if len(Nv) > 0:
                sl, obj_l, improve = selection(Nv, s_best.copy(), cov_mx, selection_strategy)
            else:
                obj_l = obj_best
                sl = s_best.copy()
                improve = False

        variation = abs((obj_l / obj_best)-1)
        if  variation <= tol:
            c_early = c_early+1
        else:
            c_early = 0

        if obj_l < obj_best:
            obj_best = obj_l
            s_best = sl.copy()

        return_best = portfolio_return(s_best, r_mean)
        z_best = np.where(s_best[1]==1)[0]
        x_best = s_best[0][z_best]

        iter_time_end = time.time()
        iter_time = round(iter_time_end - iter_time_start, 6)

        l_move.append(move)
        l_improve.append(improve)
        l_obj.append(obj_best)
        l_return.append(return_best)
        l_assets.append(x_best.shape[0])
        l_X.append(x_best)
        l_Z.append(z_best)
        l_qN.append(len(N))
        l_qNv.append(len(Nv))
        l_iter_time.append(round(iter_time, 6))

        l_X = [list(X) for X in l_X]
        l_Z = [list(Z) for Z in l_Z]



        if c_early > early_stoping:
            # print('early_stoping', i)
            break
        
        if DEBUG:
            print('iter {} | time: {} | move: {} | best: {} | return: {} | assets: {} | X: {} | qN: {} | qNv: {}' \
                .format(i,
                        iter_time,
                        move,
                        round(obj_best, 6),
                        round(return_best, 6),
                        x_best.shape[0],
                        round(s_best[0].sum(), 2),
                        len(N),
                        len(Nv)))

    l_qX = [len(X) for X in l_X]

    # log 
    log = pd.DataFrame({
        'iter':l_iter,
        'move':l_move,
        'improve':l_improve,
        'obj':l_obj,
        'return':l_return,
        'n_assets':l_assets,
        'X':l_X,
        'Z':l_Z,
        'qX':l_qX,
        'qN':l_qN,
        'qNv':l_qNv,
        'iter_time':l_iter_time
    })
    log['max_iter'] = iter
    log['neighbours'] = neighs
    log['alpha'] = alpha
    log['exp_return'] = exp_return
    log['n_port'] = n_port
    log['k_min'] = k_min
    log['k_max'] = k_max
    log['move_strategy'] = move_strategy
    log['seed'] = seed
    log['selection_strategy'] = selection_strategy
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    mh = 'local_search'
    filename = 'log_' + mh + '_' + timestamp + '.csv'
    log.to_csv(Path(LOG_PATH, filename), index=False, quotechar='"')
    total_time_end = time.time()
    total_time = total_time_end - total_time_start
    print('Local Search Time: {}'.format(round(total_time, 3)))

    return log

def debug():
    # restricoes de cardinalidade
    k_min = 10 # quantidade minima de ativos na solucao
    k_max = 10

    # restricoes de quantidade
    d_min = 0.01 # valor minimo de quantidade do ativo (percentual)
    d_max = 0.99 # valor maximo de quantidade do ativo (percentual)

    # parametros
    iter = 100 # iteracoes de busca local
    neighs = 1000 # neighbours
    alpha = 0.01 # 'step' -> quantidade de perturbacao para geracao da vizinhanca
    exp_return = 0.003 #

    portfolio = 1
    move_strategy = 'best'
    selection_strategy = 'best'
    seed = SEED

    log = local_search(iter,
                       neighs,
                       alpha,
                       exp_return,
                       portfolio,
                       k_min,
                       k_max,
                       d_min,
                       d_max,
                       move_strategy,
                       selection_strategy,
                       seed)

    log.to_csv('log.csv')

    return log
    # ['iDR', 'idID', 'TID']

def benchmarks():

    start_time = time.time()

    l_k_min = [2]
    l_k_max = [10]
    l_d_min = [0.01]
    l_d_max = [1.00]
    l_iter = [1000]
    l_neighs = [1000]
    l_alpha = [0.1]
    l_exp_return = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
    l_move_strategy = ['iDR', 'idID', 'TID', 'random', 'best']
    l_selection_strategy = ['best', 'first', 'random']
    l_portfolio = [1] # [1,2,3,4,5]
    l_seeds = list(range(100))

    parameters = [
        l_k_min, l_k_max, l_d_min, l_d_max, l_iter, l_neighs, l_alpha, l_exp_return,
        l_move_strategy, l_selection_strategy, l_portfolio, l_seeds
    ]

    parameters = list(itertools.product(*parameters))
    random.shuffle(parameters)
    print('Number of parameters combinations: {}'.format(len(parameters)))

    ray.init(num_cpus=32)

    futures = [local_search.remote(param) for param in parameters]
    logs = ray.get(futures)

    end_time = time.time()
    run_time = round(end_time - start_time, 3)
    iter_time = round(run_time / len(parameters), 3)
    print('Total time: {} | iter time: {}'.format(run_time, iter_time))

if __name__ == "__main__":
    benchmarks()
    # debug()
