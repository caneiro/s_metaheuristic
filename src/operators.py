import numpy as np

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

def normalize(X, Z):
    # normalização para soma de X = 1
    Z1 = np.where(Z==1)[0]
    E = np.zeros(X.shape[0])
    E[Z1] = 0.1
    fator = (1 - np.sum(E)) / np.sum(X)
    X = (X * fator) + E
    checkX = X.sum()
    return np.round(X, 6)

def move_idR(s, alpha):
    """
    idR -> [i]ncrease, [d]ecrease, [R]eplace
    ai é um ativo que pertence a solucao atua S
    aj é um ativo que não pertene solucao atual S
    op é 1 ou 0, sendo 1 para aumento e 0 para reducao
    se sign = 1 xi = xi * (1+q) caso contrario xi = xi * (1-q)

    """
    d_min=0.1
    d_max=1
    # Obtem dados da solucao atual
    X, Z = s
    
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
            X[ai] = 0
            Z[aj] = 1
            X[aj] = d_max
            
    else:
        fator = 1 - alpha
        X[ai] = X[ai] * fator
        if X[ai] < d_min:
            Z[ai] = 0
            X[ai] = 0
            Z[aj] = 1
            X[aj] = d_min

    X = normalize(X, Z)
    sl = [X, Z]
    return sl

def move_idID(s, alpha):
    """
    idID -> [i]ncrease, [d]ecrease, [I]nsert, [D]elete
    ai é um ativo que pertence a solucao atua S
    aj é um ativo que não pertene solucao atual S
    op é 0, 1 ou 2, sendo 0 para reducao, 1 para aumento e 2 para inserção
    se sign = 1 xi = xi * (1+q) caso contrario xi = xi * (1-q)

    """
    d_min=0.1
    d_max=1
    # Obtem dados da solucao atual
    X, Z = s

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
        # Z[ai] = 0
        # X[ai] = 0
        Z[aj] = 1
        X[aj] = d_min

    X = normalize(X, Z)
    sl = [X, Z]
    return sl

def move_TID(s, alpha):
    """
    TID  -> [T]ransfer, [I]nsert, [D]elete
    ai é um ativo que pertence a solucao atua S
    aj é um ativo que não pertene solucao atual S
    """
    d_min=0.1
    # Obtem dados da solucao atual
    X, Z = s

    # Obtem ativos e operador aleatoriamente
    ai, _, _ = aux_move_random_selections(s)
    aj = np.random.choice([j for j in range(X.shape[0]) if j != ai])
    trans = X[ai] * alpha
    # special case
    if X[ai] - trans < d_min:
        # special case II
        if Z[aj] == 0:
            X[aj] = d_min
            Z[aj] = 1
        # special case I
        else:
            X[aj] = X[ai]
        X[ai] = 0
        Z[ai] = 0
    # operacao normal
    else:
        X[ai] = X[ai] - trans
        X[aj] = X[aj] + trans
        Z[aj] = 1

    sl = [X, Z]
    return sl