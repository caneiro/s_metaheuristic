##### GUIDED LOCAL SEARCH #####

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
from numpy.core.numeric import indices
import pandas as pd
import time
from datetime import datetime

from objectives import augmented_cost_function

np.set_printoptions(linewidth=100000)

from load_data import load_port_files
from itertools import combinations, product
import copy
from pathlib import Path
import ray
import random

from functools import partial
from objectives import cost_function
from local_search import local_search
from search_space import initial_solution
from constraints import portfolio_return

DEBUG = False
SEED = 42

PATH = Path.cwd()
LOG_PATH = Path(PATH, "./data/log/")

def utility_function(Z, costs, penalties):
    z = np.where(Z==1)[0]
    penalties_ = copy.deepcopy(penalties)
    penalties_[z] = penalties_[z] + 1
    utils = np.divide(Z * costs, penalties_, out=np.zeros_like(penalties_), where=penalties_!=0)
    idx = np.argmax(utils)
    return idx

def guided_local_search(parameters):

    n_port=parameters[0]
    k=parameters[1]
    iter=parameters[2]
    neighs=parameters[3]
    alpha=parameters[4]
    lambda_=parameters[5]
    exp_return=parameters[6]
    move_strategy=parameters[7]
    selection_strategy=parameters[8]
    seed=parameters[9]
    tag=parameters[10]
    
    
    np.random.seed(seed)

    # listas para o log
    l_logs = []
    
    # obtem dados do portfolio
    port = load_port_files(n_port)
    n_assets, r_mean, r_std, cov_mx = port
    
    # gera solucao inicial
    s0 = initial_solution(port, k, alpha, exp_return)
    
    if s0 is not None:

        s_best = copy.deepcopy(s0)
        obj_best = cost_function(s_best, port)
        return_best = portfolio_return(s_best, r_mean)
        # inicializa as penalidade com zero (0)
        penalties = np.zeros(n_assets)

        # atribui o custo das features da solucao conforme o desvio padrao do retorno
        costs = 1 / r_mean * r_std

        l_move = []
        l_improve = []
        l_obj = []
        l_aug_obj = []
        l_return = []
        l_assets = []
        l_X = []
        l_Z = []
        l_Q = []
        l_qN = []
        l_qNv = []
        l_iter_time = []
        l_iter = []


        penalties = np.zeros(n_assets)
        improve_gls = False
        for i in range(iter):
            w = lambda_ * obj_best / s_best[1].sum()

            tp = penalties.sum()
            if tp<100:
                # procedimento de busca local
                sl, obj_l, improve_local = local_search(s_best,
                                                        neighs,
                                                        k,
                                                        alpha,
                                                        port,
                                                        penalties,
                                                        w,
                                                        exp_return,
                                                        move_strategy)

            # calculo funcao de utilidade e penalizacao da feature mais relevante
            util_idx = utility_function(s_best[1], costs, penalties)
            penalties[util_idx] = penalties[util_idx] + 1

            # verifica se a solucao encontrada é melhor q a best
            obj_raw = cost_function(sl, port)
            if obj_raw < obj_best:
                improve_gls = True
                obj_best = obj_raw
                s_best = copy.deepcopy(sl)
                penalties = np.zeros(n_assets)
            else:
                improve_gls = False

            # if tp>0:
                # print('tp')
            l_obj.append(obj_best)
            l_aug_obj.append(obj_l)
            l_return.append(return_best)
            X = s_best[0]
            l_X.append(X)
            Z = s_best[1]
            l_Z.append(Z)
            Q = np.sum(s_best[1])
            l_Q.append(Q)

            if DEBUG:
                print('i {0} | cost {1:.6f} | aug_cost {2:.6f} | Q {3} | gls {4} | local {5} | w {6:.6f} | tp {7} | {8} | {9}'\
                        .format(i, obj_best, obj_l, Q, improve_gls, improve_local,
                                w, tp, np.where(penalties>0)[0], np.where(Z==1)[0]))

        log = pd.DataFrame({
                'iter':list(range(iter)),
                'obj':l_obj,
                'aug_obj':l_aug_obj,
                'return':l_return,
                'X':l_X,
                'Z':l_Z,
                'Q':l_Q
        })

        log['max_iter'] = iter
        log['neighbours'] = neighs
        log['alpha'] = alpha
        log['exp_return'] = exp_return
        log['n_port'] = n_port
        log['k'] = k
        log['move_strategy'] = move_strategy
        log['seed'] = seed
        log['selection_strategy'] = selection_strategy
        log['tag'] = tag 

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        mh = 'gls'
        filename = 'log_' + mh + '_' + timestamp + '.csv'
        log.to_csv(Path(LOG_PATH, filename), index=False, quotechar='"')
    else:
        log=None

    return log

@ray.remote
def ray_guided_local_search(params):
    return guided_local_search(params)

def main():
    n_port=4
    k=10
    iter=1000
    neighs=100
    alpha=0.3
    lambda_=0.1
    exp_return=0.003
    move_strategy='random' # iDR, idID, TID
    selection_strategy='random' # best, random, first
    seed=None
    tag='testes'
    parameters = [n_port, k, iter, neighs, alpha, lambda_, 
                  exp_return, move_strategy, selection_strategy,
                  seed, tag]

    guided_local_search(parameters)

if __name__ == "__main__":
    main()

# 0.016472
# 0.032453