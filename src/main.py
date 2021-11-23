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
from datetime import datetime
from pathlib import Path

from guided_local_search import guided_local_search

np.set_printoptions(linewidth=100000)

PATH = Path.cwd()
LOG_PATH = Path(PATH, "./data/log/")

import ray
import random
from tqdm.auto import tqdm


from local_search import local_search
from load_data import load_port_files
from search_space import initial_solution
from guided_local_search import ray_guided_local_search

def debug():

    n_port=1
    k=1
    iter=100
    neighs=100
    alpha=0.1
    lambda_=0.1
    exp_return=0.001
    move_strategy='best'
    selection_strategy='best'
    seed=94

    parameters = [
        n_port, k, iter, neighs, alpha, lambda_,
        exp_return, move_strategy, selection_strategy, seed
    ]

    log = guided_local_search(parameters)
    print(log.head())

    return log
    # ['iDR', 'idID', 'TID']

def benchmarks(seed):

    start_time = time.time()

    l_k = list(range(1,32,1))
    l_iter = [1000]
    l_neighs = [100]
    l_alpha = [0.01]
    l_lambda = [0.1]
    l_exp_return = [
        0.0010, 0.0020, 0.0030, 0.0040, 0.0050,
        0.0060, 0.0070, 0.0080, 0.0090, 0.0100,
    ]
    # l_exp_return = [
    #     0.0005, 
    #     0.0010, 0.0020, 0.0030, 0.0040, 0.0050,
    #     0.0015, 0.0025, 0.0035, 0.0045, 0.0055,
    #     0.0060, 0.0070, 0.0080, 0.0090, 0.01,
    #     0.0065, 0.0075, 0.0085, 0.0095
    # ]
    l_move_strategy = ['iDR', 'idID', 'TID', 'random'] # ['iDR', 'idID', 'TID', 'random']
    l_selection_strategy = ['best', 'first', 'random'] # ['best', 'first', 'random']
    l_portfolio = [1]
    l_seeds = [None]
    l_tag = ['single_objective']

    parameters = [
        l_portfolio, l_k, l_iter, l_neighs, l_alpha, l_lambda, 
        l_exp_return, l_move_strategy, l_selection_strategy, l_seeds, l_tag
    ]

    parameters = list(itertools.product(*parameters))
    random.shuffle(parameters)
    print('Number of parameters combinations: {}'.format(len(parameters)))

    ray.init(num_cpus=30)

    futures = [ray_guided_local_search.remote(param) for param in parameters]
    logs = ray.get(futures)
    # log = pd.concat(logs, axis=0, ignore_index=True)
    # log.reset_index(drop=True, inplace=True)

    end_time = time.time()
    run_time = round(end_time - start_time, 3)
    iter_time = round(run_time / len(parameters), 3)
    print('Total time: {} | iter time: {}'.format(run_time, iter_time))

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # mh = 'gls'
    # filename = 'log_' + mh + '_' + timestamp + '.csv'
    # log.to_csv(Path(LOG_PATH, filename), index=False, quotechar='"')

    ray.shutdown()

def main():
    for i in tqdm(range(10)):
        benchmarks(i)

if __name__ == "__main__":
    main()
    # benchmarks(42)