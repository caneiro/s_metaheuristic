


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
    Nv = [sl for sl in Nv if portfolio_return(sl, r_mean) >= exp_return]

    # print('Validation | N = {} | Nv = {}'.format(len(N), len(Nv)))
    return Nv