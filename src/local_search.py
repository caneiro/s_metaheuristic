
import numpy as np
import pandas as pd
import copy
import time
import ray

from search_space import neighbours, selection
from constraints import validation, portfolio_return
from objectives import cost_function, augmented_cost_function

DEBUG=True

def local_search(sl, n, k, alpha, port, penalties, w, exp_return, move_strategy='random'):
    _, r_mean, _, _ = port
    Ns = neighbours(sl, n, alpha, move_strategy)
    Nsv = validation(Ns, exp_return, port, k)
    QN = len(Nsv)
    Zi = [np.where(s[1]==1) for s in Nsv]
    Xi = [s[0][np.where(s[1]==1)] for s in Nsv]
    s_best, obj_best, improve = selection(Nsv, sl, augmented_cost_function, port, penalties, w)
    return s_best, obj_best, improve

def local_search_old(
    s0, k_min, k_max, d_min, d_max, iter, neighs, alpha,
    exp_return, move_strategy, selection_strategy,
    cov_mx, r_mean, tag, early_stoping, tol,
    penalties=None, lambda_=0.001
    ):

    total_time_start = time.time()
        
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
    s_best = copy.deepcopy(s0)
    obj_best = cost_function(s_best, cov_mx, penalties, lambda_)
    
    # tipos de operadores a serem utilizados
    moves = ['iDR', 'idID', 'TID']

    for i in range(iter):
        l_iter.append(i)
        iter_time_start = time.time()

        if move_strategy == 'best':
            s_move = []
            obj_move = []

            for move in moves:
                N = neighbours(copy.deepcopy(s_best), neighs, alpha, move, d_min, d_max)
                Nv = validation(N, exp_return, r_mean, k_min, k_max, d_min, d_max)
                if len(Nv) > 0:
                    s_select, obj_select, improve = selection(Nv,
                                                              copy.deepcopy(s_best),
                                                              cov_mx,
                                                              selection_strategy,
                                                              'min',
                                                              penalties,
                                                              lambda_)
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
                sl = copy.deepcopy(s_best)
                improve = False                

        else:
            if move_strategy == 'random':
                move = np.random.choice(moves)

            elif move_strategy in moves:
                move = move_strategy

            N = neighbours(copy.deepcopy(s_best), neighs, alpha, move, d_min, d_max)
            Nv = validation(N, exp_return, r_mean, k_min, k_max, d_min, d_max)
            if len(Nv) > 0:
                sl, obj_l, improve = selection(Nv,
                                               copy.deepcopy(s_best),
                                               cov_mx,
                                               selection_strategy,
                                               'min',
                                               penalties,
                                               lambda_)
            else:
                obj_l = obj_best
                sl = copy.deepcopy(s_best)
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
    log['k_min'] = k_min
    log['k_max'] = k_max
    log['move_strategy'] = move_strategy
    log['selection_strategy'] = selection_strategy
    log['tag'] = tag
    log_cols = [c for c in log.columns if c not in ['X', 'Z']] + ['X', 'Z']
    log = log[log_cols]

    total_time_end = time.time()
    total_time = total_time_end - total_time_start
    print('Local Search Time: {}'.format(round(total_time, 3)))

    return s_best, obj_best, improve, log

@ray.remote
def ray_local_search(params):
    return local_search(params)