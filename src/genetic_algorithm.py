from __future__ import annotations

import random
from statistics import mean

from avaliador import OrcamentoEsgotado, ResultadoBusca
from problema import ProblemaManutencao
from repair_problem import RepairProblem


def aptidao(h: int | float) -> float:
    return 1.0 / (1.0 + h)


def selecao_torneio(populacao: list[list[int]], custos: list[int], k: int,
                    rng: random.Random) -> int:
    candidatos = rng.sample(range(len(populacao)), k)
    return min(candidatos, key=custos.__getitem__)


def selecao_roleta(populacao: list[list[int]], custos: list[int],
                   rng: random.Random) -> int:
    pesos = [aptidao(h) for h in custos]
    return rng.choices(range(len(populacao)), weights=pesos, k=1)[0]


def diversidade_unica(populacao: list[list[int]]) -> float:
    if not populacao:
        return 0.0
    return len({tuple(ind) for ind in populacao}) / len(populacao)


def genetic_algorithm(
    instancia: RepairProblem,
    orcamento: int,
    seed: int,
    tam_populacao: int = 100,
    pc: float = 0.85,
    pm: float = 0.20,
    k_torneio: int = 3,
    elitismo: int = 2,
    selecao: str = "torneio",
    operador_mutacao: str = "insertion",
    passo_historico: int = 500,
) -> ResultadoBusca:
    rng = random.Random(seed)
    problema = ProblemaManutencao(instancia, orcamento, passo_historico)

    populacao: list[list[int]] = []
    custos: list[int] = []
    historico_media: list[float] = []
    historico_diversidade: list[float] = []

    geracoes = 0
    cruzamentos = 0
    filhos_melhores_ambos = 0
    filhos_melhores_um = 0

    def registrar_janelas(custos_atuais: list[int], pop_atual: list[list[int]]) -> None:
        alvos = problema.avaliacoes // passo_historico
        if len(historico_media) < alvos and custos_atuais:
            media_h = mean(custos_atuais)
            div = diversidade_unica(pop_atual)
            while len(historico_media) < alvos:
                historico_media.append(media_h)
                historico_diversidade.append(div)

    try:
        for _ in range(tam_populacao):
            ind = problema.solucao_aleatoria(rng)
            h_ind = problema.h(ind)
            populacao.append(ind)
            custos.append(h_ind)
            registrar_janelas(custos, populacao)

        while problema.restante > 0:
            ordem = sorted(range(tam_populacao), key=custos.__getitem__)
            nova_pop = [populacao[i][:] for i in ordem[:elitismo]]
            novos_custos = [custos[i] for i in ordem[:elitismo]]

            while len(nova_pop) < tam_populacao:
                if selecao == "torneio":
                    i1 = selecao_torneio(populacao, custos, k_torneio, rng)
                    i2 = selecao_torneio(populacao, custos, k_torneio, rng)
                    while i2 == i1 and tam_populacao > 1:
                        i2 = selecao_torneio(populacao, custos, k_torneio, rng)
                elif selecao == "roleta":
                    i1 = selecao_roleta(populacao, custos, rng)
                    i2 = selecao_roleta(populacao, custos, rng)
                    while i2 == i1 and tam_populacao > 1:
                        i2 = selecao_roleta(populacao, custos, rng)
                else:
                    raise ValueError(f"Operador de seleção desconhecido: {selecao}")

                pai1, pai2 = populacao[i1], populacao[i2]
                h_pai1, h_pai2 = custos[i1], custos[i2]

                fez_cruzamento = rng.random() < pc
                if fez_cruzamento:
                    filho = problema.cruzamento(pai1, pai2, rng)
                else:
                    filho = pai1[:]

                if rng.random() < pm:
                    filho = problema.mutacao(filho, rng, operador_mutacao)

                h_filho = problema.h(filho)
                if fez_cruzamento:
                    cruzamentos += 1
                    if h_filho < min(h_pai1, h_pai2):
                        filhos_melhores_ambos += 1
                    if h_filho < max(h_pai1, h_pai2):
                        filhos_melhores_um += 1

                nova_pop.append(filho)
                novos_custos.append(h_filho)
                registrar_janelas(custos, populacao)

            populacao, custos = nova_pop, novos_custos
            geracoes += 1
            registrar_janelas(custos, populacao)

    except OrcamentoEsgotado:
        pass

    n_pontos = orcamento // passo_historico
    if custos:
        media_final = mean(custos)
        div_final = diversidade_unica(populacao)
        while len(historico_media) < n_pontos:
            historico_media.append(media_final)
            historico_diversidade.append(div_final)

    return problema.resultado({
        "tam_populacao": tam_populacao,
        "pc": pc,
        "pm": pm,
        "k_torneio": k_torneio,
        "elitismo": elitismo,
        "selecao": selecao,
        "operador_mutacao": operador_mutacao,
        "geracoes": geracoes,
        "cruzamentos": cruzamentos,
        "filhos_melhores_ambos": filhos_melhores_ambos,
        "filhos_melhores_um": filhos_melhores_um,
        "diversidade_final": historico_diversidade[-1] if historico_diversidade else 0.0,
        "historico_media_populacao": historico_media[:n_pontos],
        "historico_diversidade": historico_diversidade[:n_pontos],
    })


def executar(instancia: RepairProblem, orcamento: int, seed: int) -> ResultadoBusca:
    return genetic_algorithm(instancia, orcamento, seed)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    arquivo = (sys.argv[1] if len(sys.argv) > 1
               else str(Path(__file__).resolve().parents[1] / "instancias" / "manutencao_pequena.txt"))
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
    r = genetic_algorithm(RepairProblem(arquivo), B, seed=0)
    print(f"h* = {r.melhor_h}  geracoes = {r.extras['geracoes']}  "
          f"div_final = {r.extras['diversidade_final']:.2%}")
