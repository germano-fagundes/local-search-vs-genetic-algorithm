from __future__ import annotations

import random
from collections.abc import Iterator

from avaliador import Avaliador, ResultadoBusca
from neighborhood_operators import Neighborhoods
from repair_problem import RepairProblem

OPERADORES = {
    "swap": Neighborhoods.swap,
    "insertion": Neighborhoods.insertion,
    "2opt": Neighborhoods.two_opt,
}
OPERADORES["two_opt"] = OPERADORES["2opt"]  # nome usado em neighborhood_operators.py


class ProblemaManutencao:
    """Instância + contador de orçamento + operadores de vizinhança e genéticos."""

    def __init__(self, instancia: RepairProblem | str, orcamento: int,
                 passo_historico: int = 500) -> None:
        self.instancia = RepairProblem(instancia) if isinstance(instancia, str) else instancia
        self.n = self.instancia.n_actives
        self.avaliador = Avaliador(self.instancia, orcamento, passo_historico)

    # ------------------------------------------------------------------ orçamento
    @property
    def avaliacoes(self) -> int:
        return self.avaliador.avaliacoes

    @property
    def restante(self) -> int:
        return self.avaliador.restante

    def resultado(self, extras: dict | None = None) -> ResultadoBusca:
        return self.avaliador.resultado(extras)

    # ------------------------------------------------------------------ avaliação
    def solucao_aleatoria(self, rng: random.Random) -> list[int]:
        return [0] + rng.sample(range(1, self.n + 1), self.n)

    def h(self, s: list[int]) -> int:
        """h(s) completo; conta 1 avaliação."""
        return self.avaliador.avaliar_completo(s)[0]

    def avaliar(self, s: list[int]) -> tuple[int, list[int], list[int]]:
        """h(s) completo com os vetores C (conclusão) e H (atraso acumulado); conta 1."""
        return self.avaliador.avaliar_completo(s)

    def avaliar_vizinho(self, viz: list[int], start: int, C: list[int], H: list[int],
                        limite: float | None = None) -> int:
        """h do vizinho recalculando só o sufixo [start, n]; conta 1.

        Com `limite`, pode devolver um valor parcial >= limite (ver Avaliador).
        """
        return self.avaliador.avaliar_incremental(viz, start, C, H, limite)

    def vetores(self, s: list[int], start: int, C: list[int],
                H: list[int]) -> tuple[list[int], list[int]]:
        """C e H de um vizinho aceito (não conta avaliação: h já foi contado)."""
        return self.avaliador.vetores(s, start, C, H)

    # ------------------------------------------------------------------ vizinhanças
    def movimentos(self, operador: str) -> list[tuple[int, int]]:
        """Todos os pares (p1, p2) que geram vizinhos distintos.

        swap e 2opt: p1 < p2, n(n-1)/2 vizinhos. insertion: p1 != p2, exceto
        p2 = p1 - 1 (mover p1 para p1-1 é o mesmo que mover p1-1 para p1), o que
        dá (n-1)^2 vizinhos distintos.
        """
        n = self.n
        if operador == "insertion":
            return [(i, j) for i in range(1, n + 1) for j in range(1, n + 1)
                    if j != i and j != i - 1]
        if operador in OPERADORES:
            return [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]
        raise ValueError(f"operador desconhecido: {operador}")

    def tamanho_vizinhanca(self, operador: str) -> int:
        n = self.n
        return (n - 1) ** 2 if operador == "insertion" else n * (n - 1) // 2

    def aplicar(self, s: list[int], operador: str, p1: int, p2: int) -> tuple[list[int], int]:
        """Vizinho gerado pelo movimento (p1, p2) e a primeira posição alterada."""
        return OPERADORES[operador](s, p1, p2)

    def sucessores(self, s: list[int], operador: str) -> Iterator[tuple[list[int], int]]:
        """Itera sobre toda a vizinhança de s, devolvendo (vizinho, start)."""
        for p1, p2 in self.movimentos(operador):
            yield OPERADORES[operador](s, p1, p2)

    def sucessor_aleatorio(self, s: list[int], operador: str,
                           rng: random.Random) -> tuple[list[int], int]:
        """Sorteia (p1, p2) distintos uniformemente e devolve (vizinho, start).

        Os índices são sorteados com o RNG da execução (não com o `random`
        global usado por padrão em neighborhood_operators.py).
        """
        p1 = rng.randint(1, self.n)
        p2 = rng.randint(1, self.n - 1)
        if p2 >= p1:
            p2 += 1
        return OPERADORES[operador](s, p1, p2)

    # ------------------------------------------------------------------ operadores genéticos
    def cruzamento(self, s1: list[int], s2: list[int], rng: random.Random) -> list[int]:
        """Order Crossover (OX): copia um trecho de s1 e completa na ordem de s2."""
        n = self.n
        a, b = sorted(rng.sample(range(1, n + 1), 2))
        filho = [0] * (n + 1)
        filho[a:b + 1] = s1[a:b + 1]
        usados = set(s1[a:b + 1])
        resto = [g for g in s2[b + 1:] + s2[1:b + 1] if g not in usados]
        posicoes = list(range(b + 1, n + 1)) + list(range(1, a))
        for pos, gene in zip(posicoes, resto):
            filho[pos] = gene
        return filho

    def mutacao(self, s: list[int], rng: random.Random, operador: str = "swap") -> list[int]:
        """Mutação = um sucessor aleatório na vizinhança `operador`."""
        return self.sucessor_aleatorio(s, operador, rng)[0]
