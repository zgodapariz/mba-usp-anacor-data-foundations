# -*- coding: utf-8 -*-
"""
Etapa 2 — Tabelas de contingencia e procedimentos estatisticos.

Saidas
    resultados/contingencia/*.csv     uma tabela por par
    resultados/analise_pares.xlsx     planilha com todos os pares e abas de apoio
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comum import (NOMES, TIPOS, DADOS, SAIDA, ALFA, ORDEM_L5,
                   benjamini_hochberg, v_de_cramer, apelido)

PASTA_CT = os.path.join(SAIDA, 'contingencia')


def ordenar(tabela):
    """Coloca as categorias agrupadas na ordem Baixo -> Neutro -> Alto."""
    if set(tabela.index) <= set(ORDEM_L5):
        tabela = tabela.reindex([c for c in ORDEM_L5 if c in tabela.index])
    if set(tabela.columns) <= set(ORDEM_L5):
        tabela = tabela[[c for c in ORDEM_L5 if c in tabela.columns]]
    return tabela


def analisar():
    os.makedirs(PASTA_CT, exist_ok=True)
    base = pd.read_csv(os.path.join(DADOS, 'base_agrupada.csv'))

    registros = []
    for var_a, var_b in itertools.combinations(NOMES, 2):
        tabela = ordenar(pd.crosstab(base[var_a], base[var_b]))
        if tabela.shape[0] < 2 or tabela.shape[1] < 2:
            continue

        qui2, p, gl, esperadas = chi2_contingency(tabela, correction=False)
        n = int(tabela.values.sum())
        v = v_de_cramer(qui2, n, *tabela.shape)

        marca = '%s__%s' % (apelido(var_a), apelido(var_b))
        caminho = os.path.join(PASTA_CT, 'ct_%s.csv' % marca)
        tabela.to_csv(caminho, encoding='utf-8-sig')

        registros.append(dict(
            Variável_1=var_a, Variável_2=var_b,
            Tipo_1=TIPOS[var_a], Tipo_2=TIPOS[var_b],
            Dimensão='%d×%d' % tabela.shape, N=n,
            Qui_quadrado=round(qui2, 4), Graus_de_liberdade=int(gl),
            p_valor=p, V_de_Cramér=round(v, 4),
            Inércia_total=round(qui2 / n, 4),
            Esperadas_menor_5_pct=round(100.0 * (esperadas < 5).sum()
                                        / esperadas.size, 1),
            Esperada_mínima=round(float(esperadas.min()), 3),
            Arquivo_contingência=os.path.basename(caminho)))

    pares = pd.DataFrame(registros)
    pares['p_valor_FDR'] = benjamini_hochberg(pares.p_valor.values)
    pares['Significativa_FDR'] = np.where(pares.p_valor_FDR < ALFA, 'sim', 'não')
    pares = pares.sort_values('p_valor').reset_index(drop=True)
    pares.insert(0, 'Par', range(1, len(pares) + 1))

    retidas = pares[pares.Significativa_FDR == 'sim'].copy()

    resumo = pd.DataFrame([
        ('Respondentes', len(base)),
        ('Variáveis analisadas', len(NOMES)),
        ('Pares testados', len(pares)),
        ('Pares com p < 0,05 sem ajuste', int((pares.p_valor < ALFA).sum())),
        ('Pares significativos após FDR', len(retidas)),
        ('Nível de significância', ALFA),
        ('Método de correção', 'Benjamini–Hochberg (FDR)'),
        ('Família da correção', 'todos os pares testados, em uma única aplicação'),
        ('Agrupamento', 'escalas de 5 pontos em 1-2 / 3 / 4-5'),
    ], columns=['Item', 'Valor'])

    destino = os.path.join(SAIDA, 'analise_pares.xlsx')
    colunas_ordem = ['Par', 'Variável_1', 'Variável_2', 'Dimensão', 'N',
                     'Qui_quadrado', 'Graus_de_liberdade', 'p_valor',
                     'p_valor_FDR', 'Significativa_FDR', 'V_de_Cramér',
                     'Inércia_total', 'Esperadas_menor_5_pct',
                     'Esperada_mínima', 'Tipo_1', 'Tipo_2',
                     'Arquivo_contingência']
    with pd.ExcelWriter(destino, engine='openpyxl') as escritor:
        resumo.to_excel(escritor, sheet_name='Resumo', index=False)
        retidas[colunas_ordem].to_excel(
            escritor, sheet_name='Pares retidos (FDR)', index=False)
        pares[colunas_ordem].to_excel(
            escritor, sheet_name='Todos os pares', index=False)
        pd.read_csv(os.path.join(SAIDA, 'dicionario_variaveis.csv')).to_excel(
            escritor, sheet_name='Dicionário de variáveis', index=False)
        for aba in escritor.book.worksheets:
            for coluna in aba.columns:
                largura = max(len(str(c.value or '')) for c in coluna)
                aba.column_dimensions[coluna[0].column_letter].width = \
                    min(max(largura + 2, 10), 58)
            aba.freeze_panes = 'A2'

    pares.to_pickle(os.path.join(SAIDA, 'pares.pkl'))
    print('Pares testados ................. %d' % len(pares))
    print('p < 0,05 sem ajuste ............ %d' % (pares.p_valor < ALFA).sum())
    print('Significativos após FDR ........ %d' % len(retidas))
    print('Tabelas de contingência salvas . %d' % len(pares))
    print('\nPares retidos:')
    for _, x in retidas.iterrows():
        print('  χ²=%7.2f  p=%.6f  FDR=%.5f  V=%.2f  %s  %s × %s'
              % (x.Qui_quadrado, x.p_valor, x.p_valor_FDR, x['V_de_Cramér'],
                 x['Dimensão'], x['Variável_1'], x['Variável_2']))
    return pares


if __name__ == '__main__':
    analisar()
