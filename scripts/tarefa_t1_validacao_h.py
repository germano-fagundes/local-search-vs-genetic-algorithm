import math
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from problema import ProblemaManutencao  # noqa: E402
from repair_problem import RepairProblem  # noqa: E402
from solution import Solution  # noqa: E402

CASOS = [
    # (arquivo, permutação sem a base, h esperado, descrição)
    ("manutencao_figura.txt", [1, 4, 2, 3], 39, "Figura 1, rota (1,4,2,3)"),
    ("manutencao_figura.txt", [1, 4, 3, 2], 22, "Figura 1, rota ótima (1,4,3,2)"),
    ("manutencao_pequena.txt", list(range(1, 13)), 4366, "n=12, identidade (1,...,12)"),
]


def empates_de_arredondamento(inst: RepairProblem) -> int:
    total = inst.n_actives + 1
    return sum(1 for i in range(total) for j in range(total)
               if abs(math.hypot(inst.x[i] - inst.x[j], inst.y[i] - inst.y[j]) % 1 - 0.5) < 1e-9)


def main() -> None:
    ok = True
    print(f"{'caso':<34}{'esperado':>9}{'Solution':>10}{'h(s)':>8}")
    for arquivo, perm, esperado, descricao in CASOS:
        inst = RepairProblem(str(RAIZ / "instancias" / arquivo))
        s = [0] + perm
        h_sol = Solution(s, inst).final_cost
        h_prob = ProblemaManutencao(inst, orcamento=1).h(s)
        ok &= h_sol == h_prob == esperado
        print(f"{descricao:<34}{esperado:>9}{h_sol:>10}{h_prob:>8}"
              f"  {'OK' if h_sol == h_prob == esperado else 'FALHOU'}")

    for arquivo in ("manutencao_pequena.txt", "manutencao_media.txt", "manutencao_grande.txt"):
        empates = empates_de_arredondamento(RepairProblem(str(RAIZ / "instancias" / arquivo)))
        ok &= empates == 0
        print(f"empates .5 em t_ij ({arquivo}): {empates}")

    print("T1 validada." if ok else "T1 FALHOU.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
