import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from repair_problem import RepairProblem
from solution import Solution

little_instance = RepairProblem(str(RAIZ / "instancias" / "manutencao_pequena.txt"))
identity_permutation = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

solution1 = Solution(identity_permutation, little_instance)

print(f"Identity permutation final cost: {solution1.final_cost}")

figure_instance = RepairProblem(str(RAIZ / "instancias" / "manutencao_figura.txt"))
figure_permutation1 = [0, 1, 4, 2, 3]
figure_permutation2 = [0, 1, 4, 3, 2]

solution2 = Solution(figure_permutation1, figure_instance)
solution3 = Solution(figure_permutation2, figure_instance)

print(f"Figure initial permutation final cost: {solution2.final_cost}")
print(f"Figure optimal permutation final cost: {solution3.final_cost}")