# -*- coding: utf-8 -*-
"""Executa o pipeline completo, na ordem."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import etapa1_preparar_dados as e1
import etapa2_contingencia_estatisticas as e2
import etapa3_anacor_graficos as e3
import etapa4_validacao_sensibilidade as e4


def principal():
    for titulo, funcao in (
            ('ETAPA 1 — preparação e agrupamento dos dados', e1.preparar),
            ('ETAPA 2 — tabelas de contingência e estatísticas', e2.analisar),
            ('ETAPA 4 — validação e sensibilidade', e4.executar)):
        print('\n' + '=' * 70)
        print(titulo)
        print('=' * 70)
        resultado = funcao()
        if funcao is e4.executar:
            referencia = resultado
    print('\n' + '=' * 70)
    print('ETAPA 3 — ANACOR e mapas perceptuais')
    print('=' * 70)
    e3.executar(pares_extra=referencia)
    print('\nConcluído. Saídas em resultados/.')


if __name__ == '__main__':
    principal()
