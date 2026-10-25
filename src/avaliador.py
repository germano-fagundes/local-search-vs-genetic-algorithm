from __future__ import annotations

from dataclasses import dataclass, field

from repair_problem import RepairProblem


class OrcamentoEsgotado(Exception):
    """Levantada quando se tenta avaliar além do orçamento B."""


@dataclass
class ResultadoBusca:
    """Contrato de retorno comum a todos os algoritmos plugados no runner T3.

    historico: melhor h global após 500, 1000, 1500, ... avaliações gastas.
    extras:    diagnósticos específicos do algoritmo (ex.: taxa de aceitação do SA).
    """
    melhor_pi: list[int]
    melhor_h: int
    avaliacoes: int
    historico: list[int]
    extras: dict = field(default_factory=dict)


class Avaliador:
    """Envolve uma instância e controla o orçamento de avaliações e o histórico."""

    def __init__(self, instancia: RepairProblem, orcamento: int, passo_historico: int = 500) -> None:
        self.inst = instancia
        self.n = instancia.n_actives
        self.orcamento = orcamento
        self.passo_historico = passo_historico
        self.avaliacoes = 0
        self.melhor_h = float("inf")
        self.melhor_pi: list[int] | None = None
        self.historico: list[int] = []

    @property
    def restante(self) -> int:
        return self.orcamento - self.avaliacoes

    def _contar(self, h: int, pi: list[int]) -> None:
        if self.avaliacoes >= self.orcamento:
            raise OrcamentoEsgotado
        self.avaliacoes += 1
        if h < self.melhor_h:
            self.melhor_h = h
            self.melhor_pi = pi[:]
        if self.avaliacoes % self.passo_historico == 0:
            self.historico.append(self.melhor_h)

    def avaliar_completo(self, pi: list[int]) -> tuple[int, list[int], list[int]]:
        """h(pi) do zero. Retorna (h, C, H) com os vetores acumulados."""
        total = len(pi)
        C = [0] * total
        H = [0] * total
        self._preencher(pi, C, H, 1)
        h = H[-1]
        self._contar(h, pi)
        return h, C, H

    def avaliar_incremental(self, pi_novo: list[int], start: int,
                            C_atual: list[int], H_atual: list[int]) -> tuple[int, list[int], list[int]]:
        """h(pi_novo) reaproveitando o prefixo [0, start-1] da solução corrente.

        Retorna (h, C, H) do vizinho; as listas são novas (a solução corrente
        não é alterada), de modo que o chamador só as adota se aceitar o movimento.
        """
        C = C_atual[:start] + [0] * (len(pi_novo) - start)
        H = H_atual[:start] + [0] * (len(pi_novo) - start)
        self._preencher(pi_novo, C, H, start)
        h = H[-1]
        self._contar(h, pi_novo)
        return h, C, H

    def _preencher(self, pi: list[int], C: list[int], H: list[int], start: int) -> None:
        t = self.inst.displ_time
        p = self.inst.exec_time
        d = self.inst.due
        w = self.inst.weight
        c = C[start - 1]
        acc = H[start - 1]
        anterior = pi[start - 1]
        for k in range(start, len(pi)):
            j = pi[k]
            c += t[anterior][j] + p[j]
            atraso = c - d[j]
            if atraso > 0:
                acc += w[j] * atraso
            C[k] = c
            H[k] = acc
            anterior = j

    def resultado(self, extras: dict | None = None) -> ResultadoBusca:
        historico = self.historico[:]
        # completa o histórico caso o algoritmo pare antes de esgotar B
        while len(historico) < self.orcamento // self.passo_historico:
            historico.append(self.melhor_h)
        return ResultadoBusca(self.melhor_pi[:], int(self.melhor_h), self.avaliacoes,
                              historico, extras or {})
