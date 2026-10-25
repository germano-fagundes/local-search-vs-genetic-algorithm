from __future__ import annotations

import math
import random

from avaliador import OrcamentoEsgotado, ResultadoBusca
from problema import ProblemaManutencao
from repair_problem import RepairProblem


def calibrar_t0(s: list[int], C: list[int], H: list[int], h: int, problema: ProblemaManutencao,
                vizinhanca: str, rng: random.Random, amostras: int = 200,
                p0: float = 0.80) -> tuple[float, float]:
    """Estima T_0 a partir dos aumentos de custo de `amostras` vizinhos de s"""
    pioras = []
    for _ in range(amostras):
        viz, start = problema.sucessor_aleatorio(s, vizinhanca, rng)
        h_viz = problema.avaliar_vizinho(viz, start, C, H)
        if h_viz > h:
            pioras.append(h_viz - h)
    dh_medio = sum(pioras) / len(pioras) if pioras else 1.0
    return -dh_medio / math.log(p0), dh_medio


def simulated_annealing(instancia: RepairProblem, orcamento: int, seed: int,
                        vizinhanca: str = "insertion", p0: float = 0.80, rho: float = 1e-3,
                        amostras_calibracao: int = 200, janela: int = 1000,
                        passo_historico: int = 500) -> ResultadoBusca:
    """Executa uma corrida de SA até esgotar `orcamento` avaliações de h"""
    rng = random.Random(seed)
    problema = ProblemaManutencao(instancia, orcamento, passo_historico)

    s = problema.solucao_aleatoria(rng)
    h, C, H = problema.avaliar(s)

    T0, dh_medio = calibrar_t0(s, C, H, h, problema, vizinhanca, rng, amostras_calibracao, p0)
    Tf = rho * T0
    alpha = (Tf / T0) ** (1.0 / max(1, problema.restante))
    T = T0

    taxa_aceitacao: list[float | None] = []
    temperatura: list[float] = []
    pioras_propostas = pioras_aceitas = 0
    exp = math.exp
    u = rng.random

    try:
        while problema.restante > 0:
            viz, start = problema.sucessor_aleatorio(s, vizinhanca, rng)
            h_viz = problema.avaliar_vizinho(viz, start, C, H)
            dh = h_viz - h
            aceita = dh <= 0
            if not aceita:
                pioras_propostas += 1
                if u() < exp(-dh / T):
                    pioras_aceitas += 1
                    aceita = True
            if aceita:
                C, H = problema.vetores(viz, start, C, H)
                s, h = viz, h_viz
            T *= alpha

            if problema.avaliacoes % janela == 0:
                taxa_aceitacao.append(pioras_aceitas / pioras_propostas if pioras_propostas else None)
                temperatura.append(T)
                pioras_propostas = pioras_aceitas = 0
    except OrcamentoEsgotado:
        pass

    return problema.resultado({
        "T0": T0, "Tf": Tf, "alpha": alpha, "dh_medio": dh_medio,
        "vizinhanca": vizinhanca, "janela": janela,
        "taxa_aceitacao": taxa_aceitacao, "temperatura": temperatura,
    })


def executar(instancia: RepairProblem, orcamento: int, seed: int) -> ResultadoBusca:
    """Ponto de entrada no formato esperado por experimentos_t3.py"""
    return simulated_annealing(instancia, orcamento, seed)


if __name__ == "__main__":
    import sys

    from pathlib import Path

    arquivo = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "instancias" / "manutencao_pequena.txt")
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
    r = simulated_annealing(RepairProblem(arquivo), B, seed=0)
    print(f"h* = {r.melhor_h}  pi* = {r.melhor_pi}")
    print(f"T0 = {r.extras['T0']:.2f}  Tf = {r.extras['Tf']:.3f}  alpha = {r.extras['alpha']:.8f}")
