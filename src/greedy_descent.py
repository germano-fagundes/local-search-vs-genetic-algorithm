
from __future__ import annotations

import random
from collections import Counter

from avaliador import OrcamentoEsgotado, ResultadoBusca
from problema import ProblemaManutencao
from repair_problem import RepairProblem

VARIANTES = ("best", "first", "random")


def _descida_best(p: ProblemaManutencao, s, h, C, H, movs, operador, rng, poda, estat):
    while True:
        melhor_h, melhor_mov = h, None
        for p1, p2 in movs:
            viz, start = p.aplicar(s, operador, p1, p2)
            hv = p.avaliar_vizinho(viz, start, C, H, melhor_h if poda else None)
            if hv < melhor_h:
                melhor_h, melhor_mov = hv, (viz, start)
        if melhor_mov is None:
            return s, h
        viz, start = melhor_mov
        C, H = p.vetores(viz, start, C, H)
        s, h = viz, melhor_h
        estat["passos"] += 1


def _descida_first(p: ProblemaManutencao, s, h, C, H, movs, operador, rng, poda, estat):
    ordem = movs[:]
    rng.shuffle(ordem)
    total = len(ordem)
    falhas = pos = 0
    while falhas < total:
        p1, p2 = ordem[pos]
        pos = pos + 1 if pos + 1 < total else 0
        viz, start = p.aplicar(s, operador, p1, p2)
        hv = p.avaliar_vizinho(viz, start, C, H, h if poda else None)
        if hv < h:
            C, H = p.vetores(viz, start, C, H)
            s, h = viz, hv
            estat["passos"] += 1
            falhas = 0
        else:
            falhas += 1
    return s, h


def _descida_random(p: ProblemaManutencao, s, h, C, H, movs, operador, rng, poda, estat):
    total = len(movs)
    falhas = 0
    while falhas < total:
        viz, start = p.sucessor_aleatorio(s, operador, rng)
        hv = p.avaliar_vizinho(viz, start, C, H, h if poda else None)
        if hv < h:
            C, H = p.vetores(viz, start, C, H)
            s, h = viz, hv
            estat["passos"] += 1
            falhas = 0
        else:
            falhas += 1
    return s, h


DESCIDAS = {"best": _descida_best, "first": _descida_first, "random": _descida_random}


def greedy_descent(instancia: RepairProblem, orcamento: int, seed: int,
                   variante: str = "first", operador: str = "insertion", poda: bool = True,
                   passo_historico: int = 500) -> ResultadoBusca:
    """Executa GD com reinício aleatório até esgotar `orcamento` avaliações.

    extras do resultado:
      passos           -- passos de melhoria aceitos, somados em todas as descidas;
      otimos_locais    -- quantas descidas chegaram ao fim (ótimos locais atingidos);
      otimos_distintos -- quantos desses ótimos locais são permutações distintas;
      h_otimos         -- {h: ocorrências} dos ótimos locais atingidos;
      conjunto_otimos  -- as permutações distintas dos ótimos locais.
    """
    rng = random.Random(seed)
    p = ProblemaManutencao(instancia, orcamento, passo_historico)
    movs = p.movimentos(operador)
    descer = DESCIDAS[variante]

    estat = {"passos": 0}
    h_otimos: Counter = Counter()
    distintos: set[tuple[int, ...]] = set()
    try:
        while True:
            s = p.solucao_aleatoria(rng)
            h, C, H = p.avaliar(s)
            s, h = descer(p, s, h, C, H, movs, operador, rng, poda, estat)
            h_otimos[h] += 1
            distintos.add(tuple(s))
    except OrcamentoEsgotado:
        pass  # o melhor global já está no Avaliador

    return p.resultado({
        "variante": variante, "operador": operador, "passos": estat["passos"],
        "otimos_locais": sum(h_otimos.values()), "otimos_distintos": len(distintos),
        "h_otimos": dict(h_otimos), "conjunto_otimos": distintos,
    })


# Configuração usada no protocolo T3 (escolhida pelos experimentos T2 e de variantes).
VARIANTE_T3 = "first"
OPERADOR_T3 = "insertion"


def executar(instancia: RepairProblem, orcamento: int, seed: int) -> ResultadoBusca:
    """Ponto de entrada no formato esperado por experimentos_t3.py"""
    return greedy_descent(instancia, orcamento, seed, VARIANTE_T3, OPERADOR_T3)
