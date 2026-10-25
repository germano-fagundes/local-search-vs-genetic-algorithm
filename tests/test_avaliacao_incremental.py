"""Testes da função objetivo, da avaliação incremental e do contador de orçamento.

Uso (a partir da raiz):  python -m pytest tests/   ou   python tests/test_avaliacao_incremental.py
"""
import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from avaliador import Avaliador, OrcamentoEsgotado  # noqa: E402
from repair_problem import RepairProblem  # noqa: E402
from simulated_annealing import VIZINHANCAS, sortear_vizinho  # noqa: E402
from solution import Solution  # noqa: E402


def instancia(nome_arquivo: str) -> str:
    return str(RAIZ / "instancias" / nome_arquivo)


ARQUIVOS = ["manutencao_pequena.txt", "manutencao_media.txt", "manutencao_grande.txt"]
MOVIMENTOS = 1000


def test_t1_valores_de_referencia():
    pequena = RepairProblem(instancia("manutencao_pequena.txt"))
    figura = RepairProblem(instancia("manutencao_figura.txt"))
    av = Avaliador(pequena, 10)
    assert av.avaliar_completo(list(range(13)))[0] == 4366
    assert Avaliador(figura, 10).avaliar_completo([0, 1, 4, 2, 3])[0] == 39
    assert Avaliador(figura, 10).avaliar_completo([0, 1, 4, 3, 2])[0] == 22


def test_incremental_igual_ao_completo():
    """h_incremental(s, mov) == h_completo(s') em 1000 movimentos por vizinhança e instância."""
    rng = random.Random(0)
    for arquivo in ARQUIVOS:
        inst = RepairProblem(instancia(arquivo))
        av = Avaliador(inst, 10**7)
        pi = [0] + rng.sample(range(1, inst.n_actives + 1), inst.n_actives)
        h, C, H = av.avaliar_completo(pi)
        for viz in VIZINHANCAS:
            for _ in range(MOVIMENTOS):
                vizinho, start = sortear_vizinho(pi, viz, rng)
                h_inc, C_inc, H_inc = av.avaliar_incremental(vizinho, start, C, H)
                assert h_inc == Solution(vizinho, inst).final_cost, (arquivo, viz, vizinho)
                if rng.random() < 0.5:  # caminha, para testar a partir de estados variados
                    pi, h, C, H = vizinho, h_inc, C_inc, H_inc


def test_orcamento_para_exatamente_em_B():
    inst = RepairProblem(instancia("manutencao_pequena.txt"))
    av = Avaliador(inst, 7)
    for _ in range(7):
        av.avaliar_completo(list(range(13)))
    try:
        av.avaliar_completo(list(range(13)))
    except OrcamentoEsgotado:
        assert av.avaliacoes == 7
    else:
        raise AssertionError("avaliação além do orçamento não foi bloqueada")


if __name__ == "__main__":
    for nome, teste in list(globals().items()):
        if nome.startswith("test_"):
            teste()
            print(f"ok  {nome}")
