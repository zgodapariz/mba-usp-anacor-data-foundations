# -*- coding: utf-8 -*-
"""
Etapa 1 — Preparacao dos dados.

Regra de agrupamento
    niveis 1 e 2 -> "Baixo (1-2)"
    nivel  3     -> "Neutro (3)"
    niveis 4 e 5 -> "Alto (4-5)"

Regra de excecao
    O agrupamento NAO e aplicado quando resultaria em menos de tres categorias
    com observacoes. Nas variaveis em que apenas os niveis 3, 4 e 5 foram
    observados, a fusao dos niveis 4 e 5 deixaria a variavel com duas
    categorias — uma delas com um unico respondente — produzindo separacao
    perfeita artificial e V de Cramer igual a 1,000. Nesses casos preservam-se
    as categorias originalmente observadas.

Saidas
    dados/base_original.csv    variaveis analisadas, traduzidas, sem agrupar
    dados/base_agrupada.csv    a mesma base apos o agrupamento
    resultados/dicionario_variaveis.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comum import (VARIAVEIS, TIPOS, DADOS, SAIDA, AGRUPAMENTO,
                   nivel_da_resposta, traduzir)

ARQUIVO_BRUTO = os.path.join(DADOS, 'survey_bruto.xlsx')


def preparar():
    os.makedirs(SAIDA, exist_ok=True)
    bruto = pd.read_excel(ARQUIVO_BRUTO)
    colunas = list(bruto.columns)
    if len(colunas) != len(VARIAVEIS):
        raise SystemExit('survey_bruto.xlsx deve conter exatamente as %d '
                         'colunas analisadas, na ordem de VARIAVEIS'
                         % len(VARIAVEIS))

    original = pd.DataFrame(index=bruto.index)
    rotulo_origem = {}
    for ordem, (_, nome, _) in enumerate(VARIAVEIS):
        rotulo_origem[nome] = str(colunas[ordem]).strip()
        original[nome] = bruto[colunas[ordem]].map(
            lambda v, n=nome: traduzir(v, n))

    agrupada = pd.DataFrame(index=bruto.index)
    dicionario = []
    for _, nome, tipo in VARIAVEIS:
        serie = original[nome]
        if tipo != 'L5':
            agrupada[nome] = serie
            regra = 'não se aplica (categórica)'
        else:
            candidata = serie.map(
                lambda v: AGRUPAMENTO.get(nivel_da_resposta(v)))
            if candidata.dropna().nunique() < 3:
                agrupada[nome] = serie
                regra = 'agrupamento não aplicado (restariam <3 categorias)'
            else:
                agrupada[nome] = candidata
                regra = 'agrupada em 1-2 / 3 / 4-5'
        dicionario.append(dict(
            Variável=nome, Pergunta_original=rotulo_origem[nome],
            Tipo=tipo, Regra=regra,
            Respostas=int(serie.notna().sum()),
            Categorias_originais=int(serie.nunique()),
            Categorias_finais=int(agrupada[nome].nunique())))

    original.to_csv(os.path.join(DADOS, 'base_original.csv'),
                    index=False, encoding='utf-8-sig')
    agrupada.to_csv(os.path.join(DADOS, 'base_agrupada.csv'),
                    index=False, encoding='utf-8-sig')
    dic = pd.DataFrame(dicionario)
    dic.to_csv(os.path.join(SAIDA, 'dicionario_variaveis.csv'),
               index=False, encoding='utf-8-sig')

    print('Respondentes ............ %d' % len(original))
    print('Variáveis analisadas .... %d' % len(VARIAVEIS))
    print('  escalas de 5 pontos ... %d' % sum(1 for _, _, t in VARIAVEIS if t == 'L5'))
    print('  agrupadas ............. %d'
          % (dic.Regra == 'agrupada em 1-2 / 3 / 4-5').sum())
    print('  exceção aplicada ...... %d'
          % (dic.Regra == 'agrupamento não aplicado (restariam <3 categorias)').sum())
    excecoes = dic[dic.Regra.str.startswith('agrupamento não')]
    for _, linha in excecoes.iterrows():
        print('      · %s' % linha['Variável'])
    return original, agrupada


if __name__ == '__main__':
    preparar()
