from repair_problem import RepairProblem


class Solution:
    def __init__(self, pi: list[int], instance: RepairProblem) -> None:
        if pi[0] != 0:
            raise ValueError('The first element of a permutation must be 0 - the crew starts from their base')
        self.pi = pi
        self.conclusion_time = []
        self.acc_cost = []
        self.final_cost = 0

        self.calc_conclusion_time_and_cost(instance)

    def calc_conclusion_time_and_cost(self, instance: RepairProblem, start: int = 1):
        total_nodes = len(self.pi)
        if start < 1 or start >= total_nodes:
            raise ValueError('Start value for evaluation must be in the interval [1, total_nodes - 1]')

        if not self.conclusion_time:
            self.conclusion_time = [0] * total_nodes
            self.acc_cost = [0] * total_nodes

        for i in range(start, total_nodes):
            current = self.pi[i]
            last = self.pi[i-1]

            self.conclusion_time[i] = self.conclusion_time[i-1] + instance.displ_time[last][current] + instance.exec_time[current]
            self.acc_cost[i] = self.acc_cost[i-1] + instance.weight[current] * max(0, (self.conclusion_time[i] - instance.due[current]))

        self.final_cost = self.acc_cost[-1]