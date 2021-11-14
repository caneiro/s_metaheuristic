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
from numpy.core.numeric import indices
import pandas as pd
import time
from datetime import datetime

from seaborn.utils import locator_to_legend_entries

np.set_printoptions(linewidth=100000)

from load_data import load_port_files
from itertools import combinations, product
import copy
from pathlib import Path
from tqdm.auto import tqdm
import ray
import random

from functools import partial
from operators import move_idR, move_idID, move_TID, normalize

DEBUG = False
SEED = 42

PATH = Path.cwd()
LOG_PATH = Path(PATH, "./data/log2/")



def utility_function(Z, costs, penalties):
    z = np.where(Z==1)[0]
    penalties_ = copy.deepcopy(penalties)
    penalties_[z] = penalties_[z] + 1
    utils = np.divide(Z * costs, penalties_, out=np.zeros_like(penalties_), where=penalties_!=0)
    idx = np.argmax(utils)
    return idx




def selection(Nv, s_best, cov_mx, strategy='best', type='min', penalties=None, lambda_=0.001):

    if type == 'min':
        obj_best = cost_function(s_best, cov_mx, penalties, lambda_)
        improve = False

        if strategy in ['best', 'random']:
            obj_sn = []
            for n in Nv:
                obj_sn.append(cost_function(n, cov_mx, penalties, lambda_))
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
                obj_sn = cost_function(n, cov_mx, penalties, lambda_)
                if obj_sn < obj_best:
                    improve = True
                    s_best = n
                    obj_best = obj_sn

    return s_best, obj_best, improve



@ray.remote
def ray_local_search(params):
    return local_search(params)

def guided_local_search():

    global DEBUG
    DEBUG = True
    
    # parametros
    n_port = 1
    k_min = 5
    k_max = 15
    k = None
    d_min = 0.01
    d_max = 1.00
    iter = 100
    neighs = 1000
    alpha = 0.1
    # lambda_ = 0.001
    exp_return = 0.001
    move_strategy = 'best'
    selection_strategy = 'best'
    seed = None
    tag = 'gls'
    early_stoping = 10
    tol = 0.000001

    np.random.seed(seed)

    # listas para o log
    l_logs = []
    
    # obtem dados do portfolio
    port = load_port_files(n_port)
    n_assets, r_mean, r_std, cov_mx = port
    
    # gera solucao inicial
    s0 = initial_solution(n_assets, k, k_min, k_max, d_min, d_max, alpha, exp_return, r_mean, cov_mx)
    if s0 is None:
        print('Bad GLS')
    else:

        # # realiza a busca local na solucao inicial
        # s_best, obj_best, log = local_search(
        #     s0, k_min, k_max, d_min, d_max, iter, neighs, alpha, exp_return, 
        #     move_strategy,selection_strategy, cov_mx, r_mean, tag, early_stoping, tol
        # )

        s_best = copy.deepcopy(s0)
        obj_best = cost_function(s_best, cov_mx)

        # inicializa as penalidade com zero (0)
        penalties = np.zeros(n_assets)
        # atribui o custo das features da solucao conforme o inverso do retorno
        # costs = np.ones(n_assets)
        costs = r_std
    
        improve=False
        # inicializa o loop para o busca local com funcao de cus
        for i in range(iter):

            lambda_ = alpha * obj_best / s_best[1].sum()



            sl, obj_l, improve, log = local_search(
                s_best, k_min, k_max, d_min, d_max, iter, neighs, alpha, exp_return, 
                move_strategy,selection_strategy, cov_mx, r_mean, tag, early_stoping, tol,
                penalties, lambda_
            )

            if obj_l < obj_best:
                s_best = copy.deepcopy(sl)
                obj_best = obj_l
                # escapou do minimo local
                penalties = np.zeros(n_assets)
            else:
                # calculo funcao de utilidade e penalizacao da feature mais relevante
                util_idx = utility_function(sl[1], costs, penalties)
                penalties[util_idx] = penalties[util_idx] + 1
                total_penalties = np.sum(penalties)
                obj_best = cost_function(s_best, cov_mx, penalties, lambda_)
            


            log['gls_iter'] = i
            log['gls_improve'] = improve
            l_logs.append(log)


        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        mh = 'gls'
        filename = 'log_' + mh + '_' + timestamp + '.csv'
        gls_log = pd.concat(l_logs)
        gls_log.to_csv(Path(LOG_PATH, filename), index=False, quotechar='"')

        print(improve, obj_l, obj_best)

if __name__ == "__main__":
    guided_local_search()
