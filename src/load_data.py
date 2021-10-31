##### LOAD DATA ####

##### ITA | PG-CTE-S | TE-282 - Meta-heurísticas
##### Professor Dr. Angelo Passaro
##### Aluno: Rafael Caneiro de Oliveira
##### Versao: 0.1
##### Data: 22/10/2021

##### Carrega os dados dos arquivos de portfolios disponiveis em:
##### http://people.brunel.ac.uk/~mastjjb/jeb/orlib/portinfo.html

##### Estrutura dos arquivos
# There are currently 5 data files.

# These data files are the test problems used in the paper:
# Chang, T.-J., Meade, N., Beasley, J.E. and Sharaiha, Y.M., 
# "Heuristics for cardinality constrained portfolio optimisation" 
# Comp. & Opns. Res. 27 (2000) 1271-1302.

# The test problems are the files:
# port1, port2, ..., port5

# The format of these data files is:
# number of assets (N)
# for each asset i (i=1,...,N):
#    mean return, standard deviation of return
# for all possible pairs of assets:
#    i, j, correlation between asset i and asset j

# The unconstrained efficient frontiers for each of these
# data sets are available in the files:
# portef1, portef2, ..., portef5

# The format of these files is:
# for each of the calculated points on the unconstrained frontier:
#    mean return, variance of return

# The largest file is port5 of size 400Kb (approximately)
# The entire set of files is of size 950Kb (approximately).

from pathlib import Path
import time
import numpy as np

PATH = Path.cwd()
RAW_PATH = Path(PATH, "./data/raw/") 
print(RAW_PATH)

def load_port_files(n_port):
    start_time = time.time()
    filepath = Path(RAW_PATH, 'port' + str(n_port) + '.txt')
    with open(filepath) as fp:
        # quantidade de ativos no portfolio
        n_assets = int(fp.readline())
        # armazena as estatisticas do ativo
        r_mean = []
        r_std = []
        for n in range(n_assets):
            line = fp.readline()
            r_mean.append(float(line.strip().split()[0]))
            r_std.append(float(line.strip().split()[1]))

        # obtem o restante da matriz de covariancia
        cnt = 32
        i = []
        j = []
        cov = []
        line = fp.readline()
        while line:
            i.append(int(line.strip().split(' ')[0]))
            j.append(int(line.strip().split(' ')[1]))
            cov.append(float(line.strip().split(' ')[2]))
            line = fp.readline()
    fp.close()
    # # retorna dataframe com estatisticas dos ativos do portfolio
    # df_stats = pd.DataFrame({'port':n_port,
    #                          'i':[i_+1 for i_ in range(n_assets)],
    #                          'r_mean':r_mean,
    #                          'r_std':r_std})
    # print(df_stats.shape)

    # # retorna dataframe com matriz de covariancia dos ativos do portfolio
    # df_cov_mx = pd.DataFrame({'port':n_port,
    #                          'i':i,
    #                          'j':j,
    #                          'cov':cov})
    # print(df_cov_mx.shape)
    end_time = time.time()
    exec_time = round(end_time - start_time, 3)
    # print('>>> Arquivo port{}.txt | {} ativos | tempo: {} seg'.format(n_port, n_assets, exec_time))
    r_mean = np.array(r_mean)
    r_std = np.array(r_std)
    cov_mx = np.zeros((n_assets, n_assets))
    for i, j, cov in zip(i, j, cov):
        cov_mx[i-1, j-1] = cov
    return n_assets, r_mean, r_std, cov_mx

def main():
    for p in range(1,6,1):
        n_assets, r_mean, r_std, cov_mx = load_port_files(p)
        print(cov_mx)

if __name__ == "__main__":
    main()