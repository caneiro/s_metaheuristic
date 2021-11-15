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
from objectives import augmented_cost_function

np.set_printoptions(linewidth=100000)

from load_data import load_port_files
from itertools import combinations, product
import copy
from pathlib import Path
from tqdm.auto import tqdm
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

def guided_local_search(
    n_port=1, k=5, iter=100, neighs=1000, alpha=0.1, lambda_=1, exp_return=0.001,
    move_strategy='best', selection_strategy='best', seed=None, tag='gls', 
    early_stoping=10, tol=0.000001, debug=False
    ):

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
        costs = r_std

        l_move = []
        l_improve = []
        l_obj = []
        l_aug_obj = []
        l_return = []
        l_assets = []
        l_X = []
        l_Z = []
        l_qN = []
        l_qNv = []
        l_iter_time = []
        l_iter = []

        improve=False
        for i in range(iter):

            w = lambda_ * obj_best / s_best[1].sum()

            ### procedimento de busca local
            sl, obj_l, improve_local = local_search(s_best,
                                                    neighs,
                                                    k,
                                                    alpha,
                                                    port,
                                                    penalties,
                                                    w,
                                                    exp_return,
                                                    move_strategy)

            if obj_l < obj_best:
                s_best = copy.deepcopy(sl)
                obj_best = obj_l
                improve=True
                # escapou do minimo local
                # penalties = np.zeros(n_assets)
            else:
                # calculo funcao de utilidade e penalizacao da feature mais relevante
                improve=False
                util_idx = utility_function(sl[1], costs, penalties)
                penalties[util_idx] = penalties[util_idx] + 1
                total_penalties = np.sum(penalties)
                obj_best = augmented_cost_function(s_best, port, penalties, w)
            
            return_best = portfolio_return(s_best, r_mean)

            obj_raw = cost_function(s_best, port)
            
            l_obj.append(obj_raw)
            l_aug_obj.append(obj_best)
            l_return.append(return_best)

            print(i, improve, improve_local, obj_best, obj_raw, w)

        log = pd.DataFrame({
                'iter':list(range(iter)),
                'obj':l_obj,
                'aug_obj':l_aug_obj,
                'return':l_return,
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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        mh = 'gls'
        filename = 'log_' + mh + '_' + timestamp + '.csv'
        log.to_csv(Path(LOG_PATH, filename), index=False, quotechar='"')




if __name__ == "__main__":
    guided_local_search()
