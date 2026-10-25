from __future__ import annotations

import math
import random

from avaliador import Avaliador, OrcamentoEsgotado, ResultadoBusca
from neighborhood_operators import Neighborhoods
from repair_problem import RepairProblem

VIZINHANCAS = {
    "swap": Neighborhoods.swap,
    "insertion": Neighborhoods.insertion,
    "two_opt": Neighborhoods.two_opt,
}


def sortear_vizinho(pi: list[int], vizinhanca: str, rng: random.Random) -> tuple[list[int], int]:
    """Sorteia um vizinho uniforme usando o RNG da execução"""

    n = len(pi) - 1
    p1 = rng.randint(1, n)
    p2 = rng.randint(1, n - 1)
    if p2 >= p1:
        p2 += 1
    return VIZINHANCAS[vizinhanca](pi, p1, p2)


def calibrar_t0(pi: list[int], C: list[int], H: list[int], h: int, avaliador: Avaliador,
                vizinhanca: str, rng: random.Random, amostras: int = 200,
                p0: float = 0.80) -> tuple[float, float]:
    """Estima T_0 a partir dos aumentos de custo de `amostras` vizinhos de pi"""
    pioras = []
    for _ in range(amostras):
        viz, start = sortear_vizinho(pi, vizinhanca, rng)
        h_viz, _, _ = avaliador.avaliar_incremental(viz, start, C, H)
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
    av = Avaliador(instancia, orcamento, passo_historico)

    pi = [0] + rng.sample(range(1, instancia.n_actives + 1), instancia.n_actives)
    h, C, H = av.avaliar_completo(pi)

    T0, dh_medio = calibrar_t0(pi, C, H, h, av, vizinhanca, rng, amostras_calibracao, p0)
    Tf = rho * T0
    alpha = (Tf / T0) ** (1.0 / max(1, av.restante))
    T = T0

    taxa_aceitacao: list[float | None] = []
    temperatura: list[float] = []
    pioras_propostas = pioras_aceitas = 0
    exp = math.exp
    u = rng.random

    try:
        while av.restante > 0:
            viz, start = sortear_vizinho(pi, vizinhanca, rng)
            h_viz, C_viz, H_viz = av.avaliar_incremental(viz, start, C, H)
            dh = h_viz - h
            if dh <= 0:
                pi, h, C, H = viz, h_viz, C_viz, H_viz
            else:
                pioras_propostas += 1
                if u() < exp(-dh / T):
                    pioras_aceitas += 1
                    pi, h, C, H = viz, h_viz, C_viz, H_viz
            T *= alpha

            if av.avaliacoes % janela == 0:
                taxa_aceitacao.append(pioras_aceitas / pioras_propostas if pioras_propostas else None)
                temperatura.append(T)
                pioras_propostas = pioras_aceitas = 0
    except OrcamentoEsgotado:
        pass

    return av.resultado({
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
