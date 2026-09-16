# -*- coding: utf-8 -*-
"""
Modulo comum: mapeamento de variaveis, traducao de categorias, regras de
agrupamento, estatisticas e nucleo da ANACOR.
"""
import os
import re
import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS = os.path.join(RAIZ, 'dados')
SAIDA = os.path.join(RAIZ, 'resultados')

ALFA = 0.05

# =====================================================================
# 1. VARIAVEIS ANALISADAS
# =====================================================================
# indice da coluna no arquivo original -> (nome curto em portugues, tipo)
#   tipo 'L5'  = escala ordinal de 5 pontos  -> sujeita a agrupamento
#   tipo 'CAT' = categorica nominal/ordinal  -> usada como medida
VARIAVEIS = [
    (5,  'Senioridade', 'CAT'),
    (6,  'Área funcional', 'CAT'),
    (7,  'Porte da organização', 'CAT'),
    (8,  'Setor de atuação', 'CAT'),
    (9,  'Tempo de iniciativa formal', 'CAT'),
    (10, 'Maturidade: integração e consolidação de dados', 'L5'),
    (11, 'Maturidade: governança e curadoria de dados', 'L5'),
    (12, 'Maturidade: gestão da qualidade de dados', 'L5'),
    (13, 'Maturidade: segurança e conformidade de dados', 'L5'),
    (14, 'Maturidade: acessibilidade e democratização de dados', 'L5'),
    (15, 'Maturidade: gestão e catalogação de metadados', 'L5'),
    (16, 'Maturidade: operacionalização de IA/ML', 'L5'),
    (17, 'Maturidade: analytics de autoatendimento', 'L5'),
    (18, 'Maturidade: gestão de produtos de dados', 'L5'),
    (19, 'Arquitetura da plataforma de dados', 'CAT'),
    (20, 'Modelo operacional da organização de dados', 'CAT'),
    (22, 'Preparação para IA Generativa', 'L5'),
    (23, 'Tendência do investimento', 'CAT'),
    (24, 'Direcionador estratégico do investimento', 'CAT'),
    (25, 'Definição de prioridades', 'CAT'),
    (26, 'Estimativa de investimento anual', 'CAT'),
    (27, 'Importância: valor de negócio esperado', 'L5'),
    (28, 'Importância: conformidade regulatória', 'L5'),
    (29, 'Importância: potencial de redução de custos', 'L5'),
    (30, 'Importância: habilitação de IA/analytics', 'L5'),
    (31, 'Importância: patrocínio executivo', 'L5'),
    (32, 'Importância: viabilidade técnica', 'L5'),
    (33, 'Importância: melhoria da qualidade dos dados', 'L5'),
    (34, 'Importância: urgência das áreas de negócio', 'L5'),
    (35, 'Importância: otimização de nuvem/infraestrutura', 'L5'),
    (36, 'Importância: necessidades de governança e padronização', 'L5'),
    (37, 'Percepção de ROI', 'CAT'),
    (39, 'Alinhamento entre expectativa e valor entregue', 'CAT'),
    (41, 'Desafio: técnico', 'L5'),
    (42, 'Desafio: relacionado aos dados', 'L5'),
    (43, 'Desafio: financeiro', 'L5'),
    (44, 'Desafio: processo e governança', 'L5'),
    (45, 'Desafio: pessoas e cultura', 'L5'),
    (47, 'Foco futuro da Data Foundation', 'CAT'),
]

NOMES = [n for _, n, _ in VARIAVEIS]
TIPOS = {n: t for _, n, t in VARIAVEIS}

# =====================================================================
# 2. TRADUCAO DAS CATEGORIAS
# =====================================================================
TRADUCAO = {
    # senioridade
    'Analyst / Specialist / Engineer': 'Analista / Especialista / Engenheiro',
    'Team Lead / Coordinator': 'Líder de equipe / Coordenador',
    'Manager / Senior Manager': 'Gerente / Gerente sênior',
    'Director / Head of Department': 'Diretor / Head de departamento',
    'VP / C-Level Executive': 'VP / Executivo C-level',
    # area funcional
    'Business Strategy': 'Estratégia de negócio', 'Data & Analytics': 'Dados e analytics',
    'Digital Services': 'Serviços digitais', 'Finance': 'Finanças', 'HR': 'RH',
    'IT / Infrastructure': 'TI / Infraestrutura',
    'Marketing & Sales': 'Marketing e vendas',
    'Operations / Supply Chain': 'Operações / Cadeia de suprimentos',
    # porte
    'Up to 500': 'Até 500 funcionários', '501–2,000': 'De 501 a 2.000 funcionários',
    '2,001–10,000': 'De 2.001 a 10.000 funcionários',
    'More than 10,000': 'Mais de 10.000 funcionários',
    # setor
    'Agro': 'Agronegócio', 'Biotechnology': 'Biotecnologia',
    'Consulting / Professional Services': 'Consultoria / Serviços profissionais',
    'Consumer Packaged Goods (CPG)': 'Bens de consumo embalados [CPG]',
    'Financial Services / Insurance': 'Serviços financeiros / Seguros',
    'Food services': 'Serviços de alimentação',
    'Healthcare / Pharmaceuticals': 'Saúde / Farmacêutico',
    'Manufacturing / Industrial': 'Manufatura / Industrial',
    'Technology / Software': 'Tecnologia / Software',
    # tempo de iniciativa
    '1–3 years': 'De 1 a 3 anos', '3–5 years': 'De 3 a 5 anos',
    'More than 5 years': 'Mais de 5 anos',
    # arquitetura
    'Data Lake': 'Data Lake', 'Data Lakehouse': 'Data Lakehouse',
    'Data Mesh': 'Data Mesh', 'Hybrid Architecture': 'Arquitetura híbrida',
    'Traditional Data Warehouse': 'Data Warehouse tradicional',
    # modelo operacional
    'Centralized data team': 'Equipe de dados centralizada',
    'Data mesh / domain-oriented model': 'Data mesh / orientado a domínios',
    'Federated model': 'Modelo federado', 'Hybrid model': 'Modelo híbrido',
    'Mostly decentralized': 'Predominantemente descentralizado',
    # tendencia de investimento
    'Increasing significantly': 'Crescimento significativo',
    'Increasing slightly': 'Crescimento leve', 'Stable': 'Estável',
    # direcionador estrategico
    'Offensive driver (new value creation)': 'Ofensivo (criação de novo valor)',
    'Balanced mix of offensive and defensive drivers': 'Equilibrado (ofensivo + defensivo)',
    'Defensive driver (risk/cost reduction)': 'Defensivo (redução de risco e custo)',
    # priorizacao
    'Ad-hoc based on urgent requests': 'Ad hoc, por demandas urgentes',
    'By individual business units': 'Por unidades de negócio',
    'Centrally by a dedicated data/IT team': 'Centralmente, por equipe dedicada',
    'No formal prioritization process': 'Sem processo formal',
    'Through a cross-functional steering committee': 'Comitê diretivo multifuncional',
    # investimento anual
    'Less than €100k': 'Menos de €100 mil', '€100k–€500k': 'De €100 mil a €500 mil',
    '€500k–€2M': 'De €500 mil a €2 milhões', '€2M–€10M': 'De €2 milhões a €10 milhões',
    'More than €10M': 'Mais de €10 milhões', 'I do not know': 'Não sei / não informado',
    # ROI
    'Strongly agree': 'Concordo totalmente', 'Agree': 'Concordo', 'Neutral': 'Neutro',
    'Disagree': 'Discordo', 'Strongly disagree': 'Discordo totalmente',
    'Too early to measure / I do not know': 'Cedo demais para medir / não sei',
    # alinhamento
    'Value exceeded expectations': 'Superou as expectativas',
    'Value largely aligned with expectations': 'Amplamente alinhado',
    'Value somewhat below expectations': 'Um pouco abaixo',
    'Value significantly below expectations': 'Significativamente abaixo',
    # foco futuro
    'Expanding data integration': 'Ampliação da integração de dados',
    'Improving self-service analytics': 'Melhoria do autosserviço analítico',
    'Infrastructure cost optimization': 'Otimização dos custos de infraestrutura',
    'Strengthening governance and quality': 'Fortalecimento da governança e da qualidade',
    'Supporting Generative AI platforms': 'Suporte a plataformas de IA Generativa',
    # preparacao para IA Generativa (escala verbal de 5 pontos)
    'Not prepared': '1 (Não preparado)', 'Slightly prepared': '2 (Pouco preparado)',
    'Moderately prepared': '3 (Moderadamente preparado)',
    'Well prepared': '4 (Bem preparado)', 'Fully prepared': '5 (Totalmente preparado)',
    # escalas numericas
    '1 (Nascent)': '1 (Nascente)', '2 (Developing)': '2 (Em desenvolvimento)',
    '3 (Defined)': '3 (Definido)', '4 (Managed)': '4 (Gerenciado)',
    '5 (Optmized)': '5 (Otimizado)',
    '1 (Not Important)': '1 (Nada importante)', '3 (Neutral)': '3 (Neutro)',
    '5 (Extremely Important)': '5 (Extremamente importante)',
    '1 (Not a Challenge)': '1 (Não é um desafio)', '5 (Major Challenge)': '5 (Grande desafio)',
}

# rotulo dos niveis 2 e 4, que no formulario aparecem sem texto
NIVEL_SOLTO = {
    'Maturidade': {'2': '2 (Em desenvolvimento)', '4': '4 (Gerenciado)'},
    'Importância': {'2': '2 (Pouco importante)', '4': '4 (Muito importante)'},
    'Desafio': {'2': '2 (Pouco desafiador)', '4': '4 (Desafiador)'},
}

ORDEM_L5 = ['Baixo (1–2)', 'Neutro (3)', 'Alto (4–5)']
AGRUPAMENTO = {1: 'Baixo (1–2)', 2: 'Baixo (1–2)', 3: 'Neutro (3)',
               4: 'Alto (4–5)', 5: 'Alto (4–5)'}

_PREP_NIVEL = {'Not prepared': 1, 'Slightly prepared': 2,
               'Moderately prepared': 3, 'Well prepared': 4, 'Fully prepared': 5}


def nivel_da_resposta(valor):
    """Extrai o nivel 1..5 de uma resposta de escala ordinal de 5 pontos."""
    if pd.isna(valor):
        return None
    texto = str(valor).strip()
    if texto in _PREP_NIVEL:
        return _PREP_NIVEL[texto]
    achado = re.match(r'\s*([1-5])', texto)
    return int(achado.group(1)) if achado else None


def traduzir(valor, nome_variavel):
    """Traduz a categoria para portugues, rotulando niveis 2 e 4 sem texto."""
    if pd.isna(valor):
        return np.nan
    texto = str(valor).strip()
    if texto in TRADUCAO:
        return TRADUCAO[texto]
    familia = nome_variavel.split(':')[0]
    if familia in NIVEL_SOLTO and texto in NIVEL_SOLTO[familia]:
        return NIVEL_SOLTO[familia][texto]
    return texto


# =====================================================================
# 3. ESTATISTICAS
# =====================================================================
def benjamini_hochberg(p_valores):
    """Correcao de Benjamini-Hochberg. Devolve os p-valores ajustados."""
    p = np.asarray(p_valores, dtype=float)
    m = len(p)
    ordem = np.argsort(p)
    ajustado = np.minimum.accumulate(
        (p[ordem] * m / np.arange(1, m + 1))[::-1])[::-1]
    saida = np.empty(m)
    saida[ordem] = np.minimum(ajustado, 1.0)
    return saida


def v_de_cramer(qui2, n, linhas, colunas):
    """V de Cramer a partir da estatistica qui-quadrado."""
    k = min(linhas, colunas)
    return float(np.sqrt(qui2 / (n * (k - 1)))) if k > 1 and n > 0 else np.nan


def anacor(tabela):
    """
    Analise de Correspondencia Simples por decomposicao em valores singulares
    da matriz de residuos padronizados de Pearson (GREENACRE, 2007, cap. 8).
    """
    N = tabela.values.astype(float)
    n = N.sum()
    P = N / n
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    Dr = np.diag(1.0 / np.sqrt(r))
    Dc = np.diag(1.0 / np.sqrt(c))
    S = Dr @ (P - np.outer(r, c)) @ Dc
    U, sigma, Vt = np.linalg.svd(S, full_matrices=False)
    k = min(len(r), len(c)) - 1
    U, sigma, Vt = U[:, :k], sigma[:k], Vt[:k, :]

    autovalores = sigma ** 2
    inercia = autovalores.sum()
    F = Dr @ U @ np.diag(sigma)                  # coordenadas principais linhas
    G = Dc @ Vt.T @ np.diag(sigma)               # coordenadas principais colunas
    dims = ['Dim %d' % (i + 1) for i in range(k)]

    def _frac(coords, massa):
        num = massa[:, None] * coords ** 2
        contrib = num / np.where(autovalores == 0, np.nan, autovalores)
        total = (coords ** 2).sum(axis=1, keepdims=True)
        cos2 = coords ** 2 / np.where(total == 0, np.nan, total)
        return contrib, cos2

    contrib_l, cos2_l = _frac(F, r)
    contrib_c, cos2_c = _frac(G, c)
    return dict(
        n=int(n), autovalores=autovalores, inercia_total=inercia, dims=dims,
        perc=100 * autovalores / inercia if inercia > 0 else autovalores,
        F=pd.DataFrame(F, index=tabela.index, columns=dims),
        G=pd.DataFrame(G, index=tabela.columns, columns=dims),
        massa_l=pd.Series(r, index=tabela.index),
        massa_c=pd.Series(c, index=tabela.columns),
        contrib_l=pd.DataFrame(contrib_l, index=tabela.index, columns=dims),
        contrib_c=pd.DataFrame(contrib_c, index=tabela.columns, columns=dims),
        cos2_l=pd.DataFrame(cos2_l, index=tabela.index, columns=dims),
        cos2_c=pd.DataFrame(cos2_c, index=tabela.columns, columns=dims),
    )


def apelido(nome):
    """Identificador curto e seguro para nomes de arquivo."""
    texto = (nome.lower()
             .replace('ç', 'c').replace('ã', 'a').replace('á', 'a')
             .replace('â', 'a').replace('é', 'e').replace('ê', 'e')
             .replace('í', 'i').replace('ó', 'o').replace('ô', 'o')
             .replace('ú', 'u'))
    texto = re.sub(r'[^a-z0-9]+', '_', texto).strip('_')
    return texto[:46]
