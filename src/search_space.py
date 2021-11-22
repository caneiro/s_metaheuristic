import numpy as np
import copy
import time

from numpy.core.fromnumeric import argmin

from operators import move_idR, move_idID, move_TID, normalize
from constraints import validation
from objectives import cost_function
from objectives import augmented_cost_function

def neighbours(s, i, alpha, move):
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

    if move=='random_random':
        for _ in range(i):
            move=np.random.choice(choices)
            f_move = moves[move]    
            sl = copy.deepcopy(s)
            s1 = f_move(sl, alpha)
            N.append(copy.deepcopy(s1))

    elif move=='random':
        move=np.random.choice(choices)
        f_move = moves[move]    
        for _ in range(i):
            sl = copy.deepcopy(s)
            s1 = f_move(sl, alpha)
            N.append(copy.deepcopy(s1))

    else:
        # atribui a funcao de move determinada
        f_move = moves[move]
        for _ in range(i):
            sl = copy.deepcopy(s)
            s1 = f_move(sl, alpha)
            N.append(copy.deepcopy(s1))

    return N

def selection(S, s_best, cost_function, port, penalties, w, strategy='best', type='min'):
    QN = len(S)
    Zi = [np.where(s[1]==1) for s in S]
    Xi = [s[0][np.where(s[1]==1)] for s in S]
    if len(S)==0:
        # print('Sem vizinhos válidos')
        obj_best=None
        improve=None
    else:
        obj_best = cost_function(s_best, port, penalties, w)
        improve = False
        Cost = []
        for s in S:
            Cost.append(cost_function(s, port, penalties, w))

        Cost = np.array(Cost)
        mCost = Cost.mean()
        
        if strategy=='best':
            idx = np.argmin(Cost) if type=='min' else np.argmax(Cost)

        else:
            idx = np.where(Cost<obj_best) if type=='min' else np.where(Cost>obj_best)
            if strategy=='random':
                idx = np.choice(idx, 1)
            else:
                idx = idx[0]

        if type=='min':
            if Cost[idx]<obj_best:
                improve = True
                s_best = S[idx]
                obj_best = Cost[idx]
        else:
            if Cost[idx]>obj_best:
                improve = True
                s_best = S[idx]
                obj_best = Cost[idx]

    return s_best, obj_best, improve

def initial_solution(port, k, alpha, exp_return):
    """
    Gera a solução inicial de forma aleatória.
    n_assets <- integer: quantidade de ativos no portfolio.
    k <- integer: quantidade de ativos na solucao inicial
    s -> tuple of X e Z
        X -> array of float: quantidade do ativo (percential)
        Z -> array of int: 0 ou 1 para indicação do ativo pertence ou nao a solucao
    """
    # obtem dados do portfolio
    n_assets, _, _, _ = port

    # >>>> as vezes depois da normalizacao um ativo pode ficar abaixo ou acima de dmin e dmax!!!
    # inicializa vetores
    X = np.zeros(n_assets)
    Z = np.zeros(n_assets)

    # gera qtd de ativos a serem selecionados
    if k is None:
        k=n_assets

    z = np.random.choice(n_assets, k, False)
    # gera os vetores e ajusta para soma = 1
    Z[z] = 1
    X[z] = np.random.uniform(0.01, 1, k)
    X = normalize(X, Z)
    # retorna a solucao inicial S que é uma tupla de X e Z
    sl = [X, Z]
    # sum_X = X.sum()
    # sum_Z = Z.sum()
    # X_selected = X[np.where(Z==1)]

    Ns = neighbours(sl, 10000, alpha, 'random_random')
    Nsv = validation(Ns, exp_return, port, k)
    QN = len(Ns)
    Zi = [np.where(s[1]==1) for s in Ns]
    Xi = [s[0][np.where(s[1]==1)] for s in Ns]
    # print(k, exp_return, len(Nsv))
    if len(Nsv) == 0:
        s0 = None
    else:
        s0, _, _ = selection(Nsv, sl, augmented_cost_function, port, None, None)
    return s0

