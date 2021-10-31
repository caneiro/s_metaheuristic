

alphas = [0.01, 0.02, 0.03]
strategies = ['best', 'first', 'random']
k_vizinhos = [10, 30, 100]
for s in strategies:
    for a in alphas:
        for k in k_vizinhos:
            print(s, 'max', k, 1000, a)
            