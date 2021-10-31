import numpy as np
import pandas as pd

################ FUNCOES DE CALCULO DOS INDICADORES COMPOSTOS ######################
def calc_perdas_desvio(x):
    if x['PERDAS APURADO'] == 'ND' or x['PERDAS REFERENCIAL'] == 'ND':
        r = 'ND'
    else:
        r = float(str.replace(x['PERDAS APURADO'], ',', '.')) / \
            float(str.replace(x['PERDAS REFERENCIAL'], ',', '.'))
        return str(r)

def calc_dec_desvio(x):
    if x['DEC APURADO'] == 'ND' or x['DEC ANEEL'] == 'ND':
        r = 'ND'
    else:
        r = float(str.replace(x['DEC APURADO'], ',', '.')) / \
            float(str.replace(x['DEC ANEEL'], ',', '.'))
        return str(r)

def calc_fec_desvio(x):
    if x['FEC APURADO'] == 'ND' or x['FEC ANEEL'] == 'ND':
        r = 'ND'
    else:
        r = float(str.replace(x['FEC APURADO'], ',', '.')) / \
            float(str.replace(x['FEC ANEEL'], ',', '.'))
        return str(r)

def calc_dec_evolucao(x):
    if x['DEC INTERNO'] == 'ND' or x['DEC INTERNO ANTERIOR'] == 'ND':
        r = 'ND'
    else:
        r = ((float(str.replace(x['DEC INTERNO'], ',', '.')) / \
              float(str.replace(x['DEC INTERNO ANTERIOR'], ',', '.')))-1)*100
        return str(r)

def calc_fec_evolucao(x):
    if x['FEC INTERNO'] == 'ND' or x['FEC INTERNO ANTERIOR'] == 'ND':
        r = 'ND'
    else:
        r = ((float(str.replace(x['FEC INTERNO'], ',', '.')) / \
              float(str.replace(x['FEC INTERNO ANTERIOR'], ',', '.')))-1)*100
        return str(r)

################ CARREGAMENTO E TRATAMENTO DOS DADOS DO PREMIO #####################
def load_data():
    ## Carrega o arquivo com os dados do Premio ABRADEE de 2021
    df = pd.read_csv('data.csv', sep=';', decimal=',', na_values='', keep_default_na=False)

    ## Calculo dos indicadores compostos
    df["PERDAS DESVIO"] = df.apply(calc_perdas_desvio, axis=1)
    df["DEC DESVIO"] = df.apply(calc_dec_desvio, axis=1)
    df["FEC DESVIO"] = df.apply(calc_fec_desvio, axis=1)
    df["DEC EVOLUCAO"] = df.apply(calc_dec_evolucao, axis=1)
    df["FEC EVOLUCAO"] = df.apply(calc_fec_evolucao, axis=1)

    ## Transposição do Arquivo do formato 'wide' para 'long'
    df = df.melt(id_vars='EMPRESA',
                var_name='INDICADOR',
                value_name='VALOR')

    ## Verifica se existem valores NA - Nao aplicavel
    df['NA'] = df.VALOR.apply(lambda x: 1 if x=='NA' else 0)
    df.loc[df.NA==1, 'VALOR'] = np.NaN

    ## Verifica se existem valores ND - Nao disponivel
    df['ND'] = df.VALOR.apply(lambda x: 1 if x=='ND' else 0)
    df.loc[df.ND==1, 'VALOR'] = np.NaN

    ## Converte os valores para numero
    df.VALOR = df.VALOR.str.replace(',','.').astype(np.float32)

    ## Renomeia os nomes de colunas
    df.columns = ['empresa', 'indicador', 'valor', 'na', 'nd']

    ## Carrega as informacoe adicionais dos indicadores
    detalhes = pd.read_csv('indicadores.csv', sep=';', decimal=',')
    df = df.merge(detalhes, on='indicador', how='right')
    
    ## Classificao
    df.sort_values(['indicador', 'empresa'], inplace=True)

    # print(df.info())
    # print(df.head())
    df.to_csv('loaded_data.csv')
    return df

####################### FUNCOES DE CALCULO DO PREMIO ##############################
def fn_01_rank_quartil(valores, direcao, debug=False):
    """ 
        Calcula o Ranking e Quartis das Empresas para um indicador.
        
        Parameters:
        
            <- valores (array of float): valor apurado para um determinado
            indicador para todas as empresas concorrentes;

            <- direcao: string: informacao da direcao do indicador
            'maior melhor' ou 'menor melhor'

        Returns:
            -> rank (list of int): indices para o ranking do indicador
            
            -> quartis (list of int): valores dos quartis para o indicador das 
            empresas informadas

    """
    if debug:
        print('fn_01_rank_quartil')
    # Calcula a qtd de empresas na lista e a qtd de empresas
    # para os primeiros quartis
    n_empresas = valores.shape[0]
    n_prim_quartil = int(4 * np.ceil(n_empresas/16))
    # print('n_prim_quartil', n_prim_quartil)

    # Verifica valores nulos da lista
    idx_nulos = np.argwhere(np.isnan(valores))

    # Ordenacao e obtencao dos indices
    # [::-1]
    idx = np.argsort(valores)

    # Verifica a direcao do indicador e realiza a inversao
    # dos indices caso necessario.
    # Os valores nulos sao colocados por ultimo.
    if direcao == 'maior melhor':
        idx_sem_nulos = idx[:-idx_nulos.shape[0]]
        idx = np.concatenate([idx_sem_nulos[::-1],
                              idx_nulos.reshape(-1)])

    # Obtem a posicao do ranking a partir dos indices
    rank = np.argsort(idx)

    # Inicializa um array para os quartis
    quartis = np.zeros(idx.shape)

    # Atribuicao dos quartis
    for i in range(n_empresas):
        if rank[i] < n_prim_quartil:
            quartis[i] = 1
        elif rank[i] < n_prim_quartil * 2:
            quartis[i] = 2
        elif rank[i] < n_prim_quartil * 3:
            quartis[i] = 3
        else:
            quartis[i] = 4
    
    # Ajuste de quartil em funcao de empresas com
    # mesmo valor de indicador apurado e 
    # posicionadas em quartis diferentes
    for i in range(valores.shape[0]-1):
        # print(i, valores[idx[i]], valores[idx[i+1]])
        if valores[idx[i]] == valores[idx[i+1]]:
            quartis[idx[i+1]] = quartis[idx[i]]    
    
    # print(rank)
    # print(quartis)
    unique, counts = np.unique(quartis, return_counts=True)
    # print(np.asarray((unique, counts)).T)

    return rank, quartis

def fn_02_verifica_extremos(valores, debug=False):
    """ 
        Verifica a ocorrencia de valores extremos de maximo ou minimo.
        Este processo deve ser repetido até que não se encontre mais
        valores extremos na amostra.
        
        Parameters:
        
            <- valores (array of float): valor apurado para um determinado
            indicador para todas as empresas concorrentes;

        Returns:
            -> idx_ext (array of int): indices dos valores que foram 
            considerados valores extremos inferiores ou superiores.
    """
    if debug:
        print('fn_02_verifica_extremos')
    # Inicializacao do vetor que ira armazenar a informacao extremo ou nao
    # 0 = normal
    # 1 = extremo inferior
    # 2 = extremo superior
    idx_ext = np.zeros(valores.shape[0])

    # Loop para verificar a ocorrencia valores extremos em mais de uma vez no calculo
    for k in range(valores.shape[0]):
        encontrado = False
        nao_extremos = valores[np.where(idx_ext == 0)]
        # Calculo das estatisticas e limites para definicao de valores extremos
        q1 = np.percentile(nao_extremos[~np.isnan(nao_extremos)], 25)
        q3 = np.percentile(nao_extremos[~np.isnan(nao_extremos)], 75)
        iqr = q3 - q1
        lim_inf = q1 - (1.5 * iqr)
        lim_sup = q3 + (1.5 * iqr)

        # Itera para cada um dos valores e verifica se e extremo superior ou inferior
        for i in range(valores.shape[0]):
            if valores[i] != np.NaN:
                if idx_ext[i] == 0 and valores[i] < lim_inf:
                    idx_ext[i] = 1
                    encontrado = True
                elif idx_ext[i] == 0 and valores[i] > lim_sup:
                    idx_ext[i] = 2
                    encontrado = True
            # Log
            # print(k, i, encontrado, iqr, lim_inf, lim_sup, 
            #     valores[i], idx_ext[i], nao_extremos.shape[0])

        # Se na rodada atual nao for encontrado um valor extremo quebra ao loop
        if not encontrado:
            break
    return idx_ext
        
def fn_03_notas_extremos(idx_ext, rank, direcao, debug=False):
    """ 
        A partir dos indices de valores extremos calcula as notas.
        
        Parameters:
            <- idx_ext (array of int): indices dos valores que foram 
            considerados extremos.

            <- rank (array of int): indices para o ranking do indicador

            <- direcao (string): direcao do indicador: "maior melhor" ou 
            "menor melhor"

        Returns:
            -> notas_ext (array of float): notas para os indicadores 
            identificados como extremos.
    """
    if debug:
        print('fn_03_notas_extremos')
    # Inicializa o array para atribuicao das notas
    notas_ext = np.zeros(idx_ext.shape[0])
    count_ext_sup = 0
    count_ext_inf = 0
    idx = np.argsort(rank)

    # Itera para cada um dos indices e verifica se e valor extremo.
    for i in range(idx.shape[0]):
        if direcao == 'maior melhor':
            if idx_ext[idx[i]] == 2:
                notas_ext[idx[i]] = 10 - 0.1 * count_ext_sup
                count_ext_sup += 1
        else:        
            if idx_ext[idx[i]] == 1:
                notas_ext[idx[i]] = 10 - 0.1 * count_ext_inf
                count_ext_inf += 1
            

    for i in range(idx.shape[0]-1, 0, -1):
        if direcao == 'maior melhor':
            if idx_ext[idx[i]] == 1:
                notas_ext[idx[i]] = 6.1 + 0.1 * count_ext_inf
                count_ext_inf += 1
        else:        
            if idx_ext[idx[i]] == 2:
                notas_ext[idx[i]] = 6.1 + 0.1 * count_ext_sup
                count_ext_sup += 1

    return notas_ext

def fn_04_notas(idx_ext, rank, quartis, valores, direcao, notas_ext, debug=False):
    """ 
        Calculo de notas para os indicadores nao considerados extremos.
        
        Parameters:
            <- idx_ext (array of int): indices dos valores que foram 
            considerados extremos.

            <- rank (array of int): indices para o ranking do indicador

            <- direcao (string): direcao do indicador: "maior melhor" ou 
            "menor melhor"

        Returns:
            -> notas_ext (array of float): notas para os indicadores 
            identificados como extremos.
    """
    if debug:
        print('fn_04_notas')
    # Inicializa o array para atribuicao das notas
    notas = np.zeros(valores.shape[0])
    nota_max = {1:10, 2:9, 3:8, 4:7}

    idx = np.argsort(rank)
    
    for i in range(idx.shape[0]):
        if not np.isnan(valores[idx[i]]):
            if idx_ext[idx[i]] == 0:
                # Armazena o valor atual
                valor_atual = valores[idx[i]]
                # Armazena o quartil do valor atual
                quartil_atual = int(quartis[idx[i]])

                # Calculo dos valores minimos e maximos para o quartil atual
                valor_min = np.min(valores[np.where(quartis==quartil_atual)])
                valor_max = np.max(valores[np.where(quartis==quartil_atual)])

                # Quantidade de valores extremos min ou max no quartil atual
                count_ext_inf = np.sum(idx_ext[(idx_ext==1) & (quartis==quartil_atual)])
                count_ext_sup = np.sum((idx_ext==2) & (quartis==quartil_atual))

                # Verifica intervaloes validos e calculo da faixa e suintervaloes
                intervalos_validos = 10 - (count_ext_inf + count_ext_sup)
                faixa = (valor_max - valor_min) / intervalos_validos
                subintervalos = np.zeros(int(intervalos_validos))

                # Itera para cada subintervalo para o calculo da faixa
                if direcao == 'maior melhor':
                    for s in range(subintervalos.shape[0]):
                        subintervalos[s] = round(valor_max - (faixa * s),2)

                    # Obtencao dos subintervalo interior e superior
                    subintervalo_sup = subintervalos[:-1]
                    subintervalo_inf = subintervalos[1:]

                    # Obtencao da posicao do valor atual em realacao aos subintervalos
                    idx_subint_inf = valor_atual >= subintervalo_inf
                    idx_subint_sup = valor_atual < subintervalo_sup
                    subintervalo_idx = np.argmax(idx_subint_inf * idx_subint_sup)

                else:

                    for s in range(subintervalos.shape[0]):
                        subintervalos[s] = round(valor_min + (faixa * s), 2)

                    # Obtencao dos subintervalo interior e superior
                    subintervalo_inf = subintervalos[:-1]
                    subintervalo_sup = subintervalos[1:]

                    # Obtencao da posicao do valor atual em realacao aos subintervalos
                    idx_subint_inf = valor_atual < subintervalo_inf
                    idx_subint_sup = valor_atual >= subintervalo_sup
                    subintervalo_idx = np.argmax(idx_subint_inf * idx_subint_sup)

                # Atribuicao da Nota:
                notas[idx[i]] = nota_max[quartil_atual] - subintervalo_idx * 0.1


            else:
                notas[idx[i]] = notas_ext[idx[i]]
        else:
            notas[idx[i]] = np.nan

        # print(i, idx_ext[idx[i]], valores[idx[i]], quartis[idx[i]], notas[idx[i]])
        # print(subintervalo_inf)
        # print(subintervalo_sup)

    return notas

def fn_05_pontos(notas, peso, debug=False):
    """ 
        Calculo dos pontos da categoria geral para cada empresa.
        
        Parameters:
            <- notas (array of float): notas de cada empresa para um 
            indicador

            <- peso (float): peso do indicador

        Returns:
            -> pontos (array of float): pontos total de cada empresa.
    """
    if debug:
        print('fn_05_pontos')
    return notas * peso

def fn_aux_etapas_calculo(df, indicador, valores=None, debug=False):
    if debug:
        print('>>>> ' + indicador + ' <<<<<')
    if valores is None:
        valores = df[df.indicador==indicador].valor.values
    direcao_indicador = df[df.indicador==indicador].direcao_indicador.values[0]
    peso_geral = df[df.indicador==indicador].peso_geral.values[0]

    rank, quartis = fn_01_rank_quartil(valores=valores, direcao=direcao_indicador, debug=debug)
    idx_ext = fn_02_verifica_extremos(valores, debug)
    notas_ext = fn_03_notas_extremos(idx_ext, rank, direcao_indicador, debug)
    notas = fn_04_notas(idx_ext, rank, quartis, valores, direcao_indicador, notas_ext, debug)
    pontos = fn_05_pontos(notas, peso_geral, debug)
    return pontos

############################# PRINCIPAL ############################################
def fn_99_premio_geral(df, edp=None, debug=False):
    """
    """
    # Modifica indicadores da EDP SP
    if edp is not None:
        for indicador in edp.keys():
            # Filtra o dataframe para atualizar os indicadores
            mask = (df.indicador==indicador) & (df.empresa=='EDP SP')
            df.loc[mask, 'valor'] = edp[indicador]
            # Ajuste DEC Interno
            mask = (df.indicador=='DEC INTERNO') & (df.empresa=='EDP SP')
            df.loc[mask, 'valor'] = edp['DEC APURADO']
            # Ajuste FEC Interno
            mask = (df.indicador=='FEC INTERNO') & (df.empresa=='EDP SP')
            df.loc[mask, 'valor'] = edp['FEC APURADO']

    # Calculo indicador composto DEC
    pontos_dec = {}
    for i in ["DEC DESVIO", "DEC EVOLUCAO", "DEC INTERNO"]:
        pontos_dec[i] = fn_aux_etapas_calculo(df=df, indicador=i, debug=debug)
    
    dec = np.sum([dec for dec in pontos_dec.values()], axis=0)
    pontos_dec_final = {'DEC':fn_aux_etapas_calculo(df=df, indicador='DEC', valores=dec, debug=debug)}

    # Calculo indicador composto FEC
    pontos_fec = {}
    for i in ["FEC DESVIO", "FEC EVOLUCAO", "FEC INTERNO"]:
        pontos_fec[i] = fn_aux_etapas_calculo(df=df, indicador=i, debug=debug)
    
    fec = np.sum([fec for fec in pontos_fec.values()], axis=0)
    pontos_fec_final = {'FEC':fn_aux_etapas_calculo(df=df, indicador='FEC', valores=fec, debug=debug)}

    # Calculo demais indicadores composto    
    mask = (df.sub_indicador==0) & \
           (df.criterio_desempate==0) & \
           (~df.indicador.isin(['DEC', 'FEC']))
    indicadores = df[mask].indicador.drop_duplicates().values

    pontos_demais = {}
    for i in indicadores:
        pontos_demais[i] = fn_aux_etapas_calculo(df=df, indicador=i, debug=debug)

    # Calculo final todos os indicadores
    pontos_geral = pontos_dec_final
    pontos_geral.update(pontos_fec_final)
    pontos_geral.update(pontos_demais)
    pontos_final = np.round(np.sum([pto for pto in pontos_geral.values()], axis=0),2)
    # print(pontos_final)

    # Resultado por empresa
    empresas = df.empresa.drop_duplicates().sort_values().values
    # print(empresas)

    pontos_empresas = dict(zip(empresas, pontos_final))
    # print(pontos_empresas)

    # result_df = pd.DataFrame.from_dict(pontos_empresas,
    #                                    orient='index',
    #                                    columns=['valor'])
    # result_df.to_csv('result.csv')
    return pontos_empresas

if __name__ == "__main__":
    fn_99_premio_geral(edp=None, debug=True)

    # df = load_data()
    # print(df.info())

    # print(df[df.empresa=='EDP SP'][['indicador', 'valor']])