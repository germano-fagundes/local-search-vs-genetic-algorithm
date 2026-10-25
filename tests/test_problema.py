
import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from avaliador import OrcamentoEsgotado  # noqa: E402
from greedy_descent import VARIANTES, greedy_descent  # noqa: E402
from problema import ProblemaManutencao  # noqa: E402
from repair_problem import RepairProblem  # noqa: E402
from solution import Solution  # noqa: E402

ARQUIVOS = ["manutencao_pequena.txt", "manutencao_media.txt", "manutencao_grande.txt"]
OPERADORES = ["swap", "insertion", "2opt"]
MOVIMENTOS = 1000


def instancia(nome_arquivo: str) -> RepairProblem:
    return RepairProblem(str(RAIZ / "instancias" / nome_arquivo))


def test_t1_valores_de_referencia():
    assert ProblemaManutencao(instancia("manutencao_pequena.txt"), 1).h(list(range(13))) == 4366
    figura = instancia("manutencao_figura.txt")
    assert ProblemaManutencao(figura, 1).h([0, 1, 4, 2, 3]) == 39
    assert ProblemaManutencao(figura, 1).h([0, 1, 4, 3, 2]) == 22


def test_incremental_igual_ao_completo():
    """h_incremental(s, mov) == h_completo(s') em 1000 movimentos por vizinhança e instância."""
    rng = random.Random(0)
    for arquivo in ARQUIVOS:
        inst = instancia(arquivo)
        p = ProblemaManutencao(inst, 10**7)
        s = p.solucao_aleatoria(rng)
        h, C, H = p.avaliar(s)
        for op in OPERADORES:
            for _ in range(MOVIMENTOS):
                viz, start = p.sucessor_aleatorio(s, op, rng)
                h_inc = p.avaliar_vizinho(viz, start, C, H)
                assert h_inc == Solution(viz, inst).final_cost, (arquivo, op, viz)
                if rng.random() < 0.5:  # caminha, para testar a partir de estados variados
                    C, H = p.vetores(viz, start, C, H)
                    s, h = viz, h_inc
                    _, C_ref, H_ref = p.avaliar(s)
                    assert (C, H) == (C_ref, H_ref)


def test_poda_nunca_esconde_melhora_nem_registra_falso_melhor():
    rng = random.Random(1)
    inst = instancia("manutencao_media.txt")
    p = ProblemaManutencao(inst, 10**7)
    s = p.solucao_aleatoria(rng)
    h, C, H = p.avaliar(s)
    for _ in range(MOVIMENTOS):
        viz, start = p.sucessor_aleatorio(s, "insertion", rng)
        exato = Solution(viz, inst).final_cost
        podado = p.avaliar_vizinho(viz, start, C, H, limite=h)
        assert (podado == exato) if exato < h else (podado >= h)
        assert p.avaliador.melhor_h == Solution(p.avaliador.melhor_pi, inst).final_cost


def test_orcamento_para_exatamente_em_B():
    p = ProblemaManutencao(instancia("manutencao_pequena.txt"), 7)
    for _ in range(7):
        p.h(list(range(13)))
    try:
        p.h(list(range(13)))
    except OrcamentoEsgotado:
        assert p.avaliacoes == 7
    else:
        raise AssertionError("avaliação além do orçamento não foi bloqueada")


def test_tamanho_das_vizinhancas_e_vizinhos_distintos():
    p = ProblemaManutencao(instancia("manutencao_pequena.txt"), 1)
    s = p.solucao_aleatoria(random.Random(2))
    n = p.n
    esperado = {"swap": n * (n - 1) // 2, "insertion": (n - 1) ** 2, "2opt": n * (n - 1) // 2}
    for op in OPERADORES:
        vizinhos = [tuple(v) for v, _ in p.sucessores(s, op)]
        assert len(vizinhos) == len(set(vizinhos)) == esperado[op] == p.tamanho_vizinhanca(op)
        assert tuple(s) not in vizinhos


def test_cruzamento_ox_e_mutacao_geram_permutacoes():
    rng = random.Random(3)
    p = ProblemaManutencao(instancia("manutencao_media.txt"), 1)
    for _ in range(500):
        s1, s2 = p.solucao_aleatoria(rng), p.solucao_aleatoria(rng)
        for filho in (p.cruzamento(s1, s2, rng), p.mutacao(s1, rng)):
            assert filho[0] == 0 and sorted(filho[1:]) == list(range(1, p.n + 1))


def test_gd_respeita_orcamento_e_acha_otimo_na_pequena():
    inst = instancia("manutencao_pequena.txt")
    for var in VARIANTES:
        r = greedy_descent(inst, 20_000, seed=0, variante=var)
        assert r.avaliacoes == 20_000
        assert r.melhor_h == Solution(r.melhor_pi, inst).final_cost
        assert r.melhor_h == 1875


if __name__ == "__main__":
    for nome, teste in list(globals().items()):
        if nome.startswith("test_"):
            teste()
            print(f"ok  {nome}")
