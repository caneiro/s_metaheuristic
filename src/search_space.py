


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


def initial_solution(n_assets, k, k_min, k_max, d_min, d_max, alpha, exp_return, r_mean, cov_mx):
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
    s0 = [X, Z]
    # sum_X = X.sum()
    # sum_Z = Z.sum()
    # X_selected = X[np.where(Z==1)]

    Ns = neighbours(copy.deepcopy(s0), 1000, alpha, 'random', d_min, d_max)
    Nsv = validation(copy.deepcopy(Ns), exp_return, r_mean, k_min, k_max, d_min, d_max)
    if len(Nsv)>=1:
        sl, _, _ = selection(copy.deepcopy(Nsv), s0.copy(), cov_mx, 'best', 'min')
        return sl
    else:
        print('>>> Bad Initial Solution | k_min: {} | k_max {} | d_min {} | d_max {} | alpha {} | exp_retur {}'\
            .format(k_min, k_max, d_min, d_max, alpha, exp_return))
        return None
