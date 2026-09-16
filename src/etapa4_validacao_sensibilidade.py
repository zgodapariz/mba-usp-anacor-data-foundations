# -*- coding: utf-8 -*-
"""
Etapa 4 — Validacao e analise de sensibilidade.


Saidas
    resultados/validacao_e_sensibilidade.xlsx
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comum import (VARIAVEIS, NOMES, DADOS, SAIDA, ALFA, AGRUPAMENTO,
                   benjamini_hochberg, v_de_cramer, nivel_da_resposta)

PREP = 'Preparação para IA Generativa'


def testar_todos(base):
    """Qui-quadrado e V de Cramer para todos os pares de uma base."""
    linhas = []
    for a, b in itertools.combinations(NOMES, 2):
        tabela = pd.crosstab(base[a], base[b])
        if tabela.shape[0] < 2 or tabela.shape[1] < 2:
            continue
        qui2, p, gl, _ = chi2_contingency(tabela, correction=False)
        n = int(tabela.values.sum())
        linhas.append(dict(Variável_1=a, Variável_2=b, Qui_quadrado=qui2,
                           Graus_de_liberdade=int(gl), p_valor=p, N=n,
                           V_de_Cramér=v_de_cramer(qui2, n, *tabela.shape),
                           Dimensão='%d×%d' % tabela.shape))
    saida = pd.DataFrame(linhas)
    saida['p_valor_FDR'] = benjamini_hochberg(saida.p_valor.values)
    return saida


def montar_base(original, excecao=True, agrupar_prep=True):
    """Aplica o agrupamento sob uma dada configuracao de regras."""
    base = pd.DataFrame(index=original.index)
    for _, nome, tipo in VARIAVEIS:
        serie = original[nome]
        if tipo != 'L5' or (nome == PREP and not agrupar_prep):
            base[nome] = serie
            continue
        candidata = serie.map(lambda v: AGRUPAMENTO.get(nivel_da_resposta(v)))
        base[nome] = (serie if excecao and candidata.dropna().nunique() < 3
                      else candidata)
    return base


def executar():
    original = pd.read_csv(os.path.join(DADOS, 'base_original.csv'))

    # ---- 1. validacao contra a analise sem agrupamento -------------------
    sem_agrupar = testar_todos(original).sort_values('p_valor')
    selecionados = sem_agrupar[sem_agrupar.p_valor_FDR < ALFA].copy()
    selecionados.insert(0, 'Par_Tabela_2', range(1, len(selecionados) + 1))

    # ---- 2. sensibilidade as regras de agrupamento -----------------------
    sensibilidade = []
    for excecao in (False, True):
        for prep in (False, True):
            resultado = testar_todos(montar_base(original, excecao, prep))
            retidos = resultado[resultado.p_valor_FDR < ALFA]
            sensibilidade.append(dict(
                Regra_de_exceção='sim' if excecao else 'não',
                Preparação_agrupada='sim' if prep else 'não',
                Pares_testados=len(resultado),
                p_menor_005_sem_ajuste=int((resultado.p_valor < ALFA).sum()),
                Significativos_FDR=len(retidos),
                Com_V_igual_1=int((retidos['V_de_Cramér'] > 0.999).sum())))
    sensibilidade = pd.DataFrame(sensibilidade)

    # ---- 3. familia unica versus familia em duas etapas ------------------
    agrupada = pd.read_csv(os.path.join(DADOS, 'base_agrupada.csv'))
    completo = testar_todos(agrupada)
    chaves = set(zip(selecionados['Variável_1'], selecionados['Variável_2']))
    subconjunto = completo[completo.apply(
        lambda x: (x['Variável_1'], x['Variável_2']) in chaves, axis=1)].copy()
    subconjunto['p_FDR_família_reduzida'] = benjamini_hochberg(
        subconjunto.p_valor.values)
    subconjunto = subconjunto.rename(
        columns={'p_valor_FDR': 'p_FDR_família_completa'})
    subconjunto['Retida_família_completa'] = np.where(
        subconjunto['p_FDR_família_completa'] < ALFA, 'sim', 'não')
    subconjunto['Retida_família_reduzida'] = np.where(
        subconjunto['p_FDR_família_reduzida'] < ALFA, 'sim', 'não')
    subconjunto = subconjunto.sort_values('p_valor')

    comparacao = pd.DataFrame([
        ('Pares selecionados sem agrupamento (Tabela 2)', len(selecionados)),
        ('Desses, submetidos ao agrupamento', len(subconjunto)),
        ('Retidos — FDR sobre a família completa (%d pares)' % len(completo),
         int((subconjunto.Retida_família_completa == 'sim').sum())),
        ('Retidos — FDR sobre a família reduzida (%d pares)' % len(subconjunto),
         int((subconjunto.Retida_família_reduzida == 'sim').sum())),
    ], columns=['Critério', 'Pares retidos'])

    destino = os.path.join(SAIDA, 'validacao_e_sensibilidade.xlsx')
    with pd.ExcelWriter(destino, engine='openpyxl') as escritor:
        comparacao.to_excel(escritor, sheet_name='Comparação de famílias',
                            index=False)
        sensibilidade.to_excel(escritor, sheet_name='Sensibilidade das regras',
                               index=False)
        selecionados.to_excel(escritor, sheet_name='Validação (sem agrupar)',
                              index=False)
        subconjunto.to_excel(escritor, sheet_name='Selecionados após agrupar',
                             index=False)
        for aba in escritor.book.worksheets:
            for coluna in aba.columns:
                largura = max(len(str(c.value or '')) for c in coluna)
                aba.column_dimensions[coluna[0].column_letter].width = \
                    min(max(largura + 2, 10), 58)
            aba.freeze_panes = 'A2'

    print('VALIDAÇÃO — análise sem agrupamento')
    print('  pares testados ............... %d' % len(sem_agrupar))
    print('  p < 0,05 sem ajuste .......... %d' % (sem_agrupar.p_valor < ALFA).sum())
    print('  significativos após FDR ...... %d' % len(selecionados))
    print('\nSENSIBILIDADE')
    print(sensibilidade.to_string(index=False))
    print('\nCOMPARAÇÃO DE FAMÍLIAS')
    print(comparacao.to_string(index=False))
    return list(chaves)


if __name__ == '__main__':
    executar()
