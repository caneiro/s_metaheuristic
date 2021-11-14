
def debug():
    global DEBUG
    DEBUG = True
    # restricoes de cardinalidade
    k_min = 2 # quantidade minima de ativos na solucao
    k_max = 15
    k=7

    # restricoes de quantidade
    d_min = 0.01 # valor minimo de quantidade do ativo (percentual)
    d_max = 1.00 # valor maximo de quantidade do ativo (percentual)

    # parametros
    iter = 1000 # iteracoes de busca local
    neighs = 100 # neighbours
    alpha = 0.1 # 'step' -> quantidade de perturbacao para geracao da vizinhanca
    exp_return = 0.0025 #

    portfolio = 1
    move_strategy = 'best'
    selection_strategy = 'best'
    seed = None

    tag = 'ccef'

    parameters = [
        k_min, k_max, d_min, d_max, iter, neighs, alpha, exp_return,
        move_strategy, selection_strategy, portfolio, seed, tag, k
    ]

    log = local_search(parameters)
    # log.to_csv('log.csv')

    return log
    # ['iDR', 'idID', 'TID']

def benchmarks():
    
    global DEBUG
    DEBUG = False

    start_time = time.time()
    l_k = [7]
    l_k_min = [2]
    l_k_max = [15]
    l_d_min = [0.01]
    l_d_max = [1.00]
    l_iter = [1000]
    l_neighs = [100]
    l_alpha = [0.1]
    l_exp_return = [
        0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0045, 0.0050,
        0.0055, 0.0060, 0.0065, 0.0070, 0.0075, 0.0080, 0.0085, 0.0090, 0.0095, 0.01
    ]

    l_move_strategy = ['iDR', 'idID', 'TID', 'random', 'best']
    l_selection_strategy = ['best', 'first', 'random']
    l_portfolio = [1] # [1,2,3,4,5]
    l_seeds = list(range(10))

    tag = ['ccef']

    parameters = [
        l_k_min, l_k_max, l_d_min, l_d_max, l_iter, l_neighs, l_alpha, l_exp_return,
        l_move_strategy, l_selection_strategy, l_portfolio, l_seeds, tag, l_k
    ]

    parameters = list(itertools.product(*parameters))
    random.shuffle(parameters)
    print('Number of parameters combinations: {}'.format(len(parameters)))

    ray.init(num_cpus=32)

    futures = [ray_local_search.remote(param) for param in parameters]
    logs = ray.get(futures)

    end_time = time.time()
    run_time = round(end_time - start_time, 3)
    iter_time = round(run_time / len(parameters), 3)
    print('Total time: {} | iter time: {}'.format(run_time, iter_time))
