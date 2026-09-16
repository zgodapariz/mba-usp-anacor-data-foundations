# -*- coding: utf-8 -*-
"""
Etapa 3 — Analise de Correspondencia Simples (ANACOR) e mapas perceptuais.

Saidas
    resultados/mapas/*.png
    resultados/anacor/*.csv        coordenadas, inercia, contribuicoes, cos²
    resultados/anacor/relatorio_anacor.txt
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comum import DADOS, SAIDA, anacor, apelido
from etapa2_contingencia_estatisticas import ordenar

PASTA_MAPAS = os.path.join(SAIDA, 'mapas')
PASTA_ANACOR = os.path.join(SAIDA, 'anacor')
AZUL, VERMELHO = '#1f4e79', '#b3312c'


# ---------------------------------------------------------------- rotulagem
def _declutter(ax, itens, iteracoes=200):
    """
    Afasta verticalmente rotulos sobrepostos.

    Os rotulos sao anotacoes com textcoords='offset points'; o deslocamento e
    aplicado sobre o proprio offset em pontos, e nao sobre a posicao em
    coordenadas de dados, que set_position() sobrescreveria.
    """
    figura = ax.figure
    figura.canvas.draw()
    textos = [t for t, _ in itens]
    if len(textos) < 2:
        return
    por_ponto = 72.0 / figura.dpi
    for _ in range(iteracoes):
        ren = figura.canvas.get_renderer()
        caixas = [t.get_window_extent(renderer=ren) for t in textos]
        moveu = False
        for i in range(len(textos)):
            for j in range(i + 1, len(textos)):
                a, b = caixas[i], caixas[j]
                if not a.overlaps(b):
                    continue
                moveu = True
                passo = ((min(a.y1, b.y1) - max(a.y0, b.y0)) / 2 + 1.0) * por_ponto
                acima = 1 if (a.y0 + a.y1) >= (b.y0 + b.y1) else -1
                for texto, sinal in ((textos[i], acima), (textos[j], -acima)):
                    dx, dy = texto.xyann
                    texto.xyann = (dx, dy + sinal * passo)
        if not moveu:
            break
        figura.canvas.draw()


def _alinhar(ax, itens, margem=6.0):
    """Vira para dentro do quadro os rotulos que extrapolariam a area util."""
    figura = ax.figure
    figura.canvas.draw()
    ren = figura.canvas.get_renderer()
    quadro = ax.get_window_extent()
    mudou = False
    for texto, (px, _) in itens:
        largura = texto.get_window_extent(renderer=ren).width
        ponto = ax.transData.transform((px, 0))[0]
        cabe_direita = ponto + largura + margem <= quadro.x1
        cabe_esquerda = ponto - largura - margem >= quadro.x0
        dx, dy = texto.xyann
        passo = abs(dx) or 8
        if not cabe_direita and not cabe_esquerda:
            alvo, novo_dx = 'center', 0
        elif cabe_direita:
            alvo, novo_dx = 'left', passo
        else:
            alvo, novo_dx = 'right', -passo
        if texto.get_horizontalalignment() != alvo or dx != novo_dx:
            texto.set_horizontalalignment(alvo)
            texto.xyann = (novo_dx, dy)
            mudou = True
    if mudou:
        figura.canvas.draw()


# ---------------------------------------------------------------- geometria
def _quadro(xs, ys, folga=1.30):
    """Janela quadrada centrada na nuvem — preserva a isometria dos eixos."""
    cx = (xs.max() + xs.min()) / 2
    cy = (ys.max() + ys.min()) / 2
    meio = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2
    return (cx - max(meio * folga, 0.05), cx + max(meio * folga, 0.05)), \
           (cy - max(meio * folga, 0.05), cy + max(meio * folga, 0.05))


def _ponto_extremo(xs, ys):
    """Categoria rara projetada muito longe do restante da nuvem."""
    r = np.sqrt(xs ** 2 + ys ** 2)
    ordenado = np.sort(r)[::-1]
    return len(ordenado) >= 4 and ordenado[2] > 0 and ordenado[0] > 3.0 * ordenado[2]


def mapa_perceptual(res, nome_linha, nome_coluna, arquivo):
    F, G = res['F'], res['G']
    if F.shape[1] < 2:
        return False                                   # solucao unidimensional

    xs = np.concatenate([F.iloc[:, 0].values, G.iloc[:, 0].values])
    ys = np.concatenate([F.iloc[:, 1].values, G.iloc[:, 1].values])
    nomes = list(F.index) + list(G.index)

    if _ponto_extremo(xs, ys):
        r = np.sqrt(xs ** 2 + ys ** 2)
        dentro = r <= np.sort(r)[::-1][2] * 1.8
        limites = _quadro(xs[dentro], ys[dentro], folga=1.45)
        fora = [(n, a, b) for n, a, b, d in zip(nomes, xs, ys, dentro) if not d]
    else:
        limites, fora = _quadro(xs, ys), []

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    ax.axhline(0, color='#8c8c8c', lw=0.8, zorder=1)
    ax.axvline(0, color='#8c8c8c', lw=0.8, zorder=1)
    ax.scatter(F.iloc[:, 0], F.iloc[:, 1], s=80, marker='o', color=AZUL,
               edgecolors='white', linewidths=0.8, label=nome_linha, zorder=4)
    ax.scatter(G.iloc[:, 0], G.iloc[:, 1], s=90, marker='^', color=VERMELHO,
               edgecolors='white', linewidths=0.8, label=nome_coluna, zorder=4)

    itens = []
    for df, cor, dy in ((F, AZUL, 8), (G, VERMELHO, -13)):
        for nome, (x, y) in zip(df.index, df.iloc[:, :2].values):
            if fora and not (limites[0][0] <= x <= limites[0][1]
                             and limites[1][0] <= y <= limites[1][1]):
                continue
            itens.append((ax.annotate(
                nome, (x, y), fontsize=8.5, color=cor, zorder=6,
                xytext=(8, dy), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.18', fc='white',
                          ec='none', alpha=0.8)), (x, y)))

    ax.set_xlim(*limites[0])
    ax.set_ylim(*limites[1])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.16, linestyle=':')
    ax.tick_params(labelsize=8)
    ax.set_xlabel('Dimensão 1 (%.1f%% da inércia)' % res['perc'][0], fontsize=9.5)
    ax.set_ylabel('Dimensão 2 (%.1f%% da inércia)' % res['perc'][1], fontsize=9.5)
    ax.legend(fontsize=8.5, loc='upper center', framealpha=0.93,
              bbox_to_anchor=(0.5, 1.14), ncol=1)

    if fora:
        def _br(v):
            return ('%.2f' % (v if abs(v) >= 0.005 else 0.0)).replace('.', ',')
        linhas = ['Fora da escala (categoria de massa reduzida):']
        linhas += ['   %s  (%s; %s)' % (n, _br(a), _br(b)) for n, a, b in fora]
        ax.text(0.015, 0.015, '\n'.join(linhas), transform=ax.transAxes,
                fontsize=7.5, va='bottom', ha='left', color='#444444', zorder=7,
                bbox=dict(boxstyle='round,pad=0.35', fc='#f4f4f4',
                          ec='#c9c9c9', lw=0.6))

    fig.tight_layout()
    _alinhar(ax, itens)
    _declutter(ax, itens)
    fig.savefig(arquivo, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return True


# ---------------------------------------------------------------- execucao
def executar(pares_extra=()):
    os.makedirs(PASTA_MAPAS, exist_ok=True)
    os.makedirs(PASTA_ANACOR, exist_ok=True)
    base = pd.read_csv(os.path.join(DADOS, 'base_agrupada.csv'))
    pares = pd.read_pickle(os.path.join(SAIDA, 'pares.pkl'))

    alvo = pares[pares.Significativa_FDR == 'sim'].copy()
    alvo['Motivo'] = 'significativa após FDR'
    if len(pares_extra):
        chaves = set(zip(alvo['Variável_1'], alvo['Variável_2']))
        extra = pares[pares.apply(
            lambda x: (x['Variável_1'], x['Variável_2']) in set(pares_extra)
            and (x['Variável_1'], x['Variável_2']) not in chaves, axis=1)].copy()
        extra['Motivo'] = 'referência (selecionada sem agrupamento)'
        alvo = pd.concat([alvo, extra], ignore_index=True)

    relatorio = ['ANACOR — Análise de Correspondência Simples',
                 'Agrupamento: escalas de 5 pontos em 1-2 / 3 / 4-5',
                 'Correção de FDR aplicada à família completa dos %d pares testados.'
                 % len(pares), '']
    feitos = 0
    for _, linha in alvo.iterrows():
        a, b = linha['Variável_1'], linha['Variável_2']
        tabela = ordenar(pd.crosstab(base[a], base[b]))
        tabela = tabela.loc[tabela.sum(axis=1) > 0, tabela.sum(axis=0) > 0]
        res = anacor(tabela)
        marca = '%s__%s' % (apelido(a), apelido(b))

        relatorio += ['=' * 78, '%s  ×  %s' % (a, b),
                      '[%s]' % linha['Motivo'], '=' * 78,
                      '', 'Tabela de contingência (N = %d)' % res['n'],
                      tabela.astype(int).to_string(), '',
                      'χ² = %.4f   gl = %d   p = %.6f   p-FDR = %.6f   V = %.4f'
                      % (linha.Qui_quadrado, linha.Graus_de_liberdade,
                         linha.p_valor, linha.p_valor_FDR, linha['V_de_Cramér']),
                      'Inércia total = %.4f' % res['inercia_total'],
                      'Células com esperada < 5: %.0f%%' % linha.Esperadas_menor_5_pct,
                      '', 'Decomposição da inércia']
        decomposicao = pd.DataFrame(
            {'Inércia': res['autovalores'], '% da inércia': res['perc'],
             '% acumulado': np.cumsum(res['perc'])}, index=res['dims'])
        relatorio += [decomposicao.round(4).to_string(), '',
                      'Coordenadas principais — linhas',
                      pd.concat([res['massa_l'].rename('Massa'), res['F']],
                                axis=1).round(4).to_string(), '',
                      'Coordenadas principais — colunas',
                      pd.concat([res['massa_c'].rename('Massa'), res['G']],
                                axis=1).round(4).to_string(), '',
                      'Contribuições absolutas — linhas',
                      res['contrib_l'].round(4).to_string(), '',
                      'Contribuições absolutas — colunas',
                      res['contrib_c'].round(4).to_string(), '',
                      'Qualidade de representação (cos²) — linhas',
                      res['cos2_l'].round(4).to_string(), '',
                      'Qualidade de representação (cos²) — colunas',
                      res['cos2_c'].round(4).to_string(), '']

        decomposicao.round(6).to_csv(
            os.path.join(PASTA_ANACOR, 'inercia_%s.csv' % marca), encoding='utf-8-sig')
        res['F'].round(6).to_csv(
            os.path.join(PASTA_ANACOR, 'coord_linhas_%s.csv' % marca), encoding='utf-8-sig')
        res['G'].round(6).to_csv(
            os.path.join(PASTA_ANACOR, 'coord_colunas_%s.csv' % marca), encoding='utf-8-sig')
        if mapa_perceptual(res, a, b,
                           os.path.join(PASTA_MAPAS, 'mapa_%s.png' % marca)):
            feitos += 1

    with open(os.path.join(PASTA_ANACOR, 'relatorio_anacor.txt'), 'w',
              encoding='utf-8') as arquivo:
        arquivo.write('\n'.join(relatorio))

    print('Pares submetidos à ANACOR ... %d' % len(alvo))
    print('Mapas perceptuais gerados ... %d' % feitos)
    return alvo


if __name__ == '__main__':
    executar()
