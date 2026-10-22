import math

class RepairProblem:
    def __init__(self, file_name) -> None:
        self.x = []
        self.y = []
        self.duration = []
        self.due = []
        self.weight = []
        self.distance = []

        self.parse(file_name)
        self.calc_distances()

    def parse(self, file_name: str):
        with open(file_name, 'r', encoding="utf-8") as file:
            lines = file.read().splitlines()

        # number of actives
        self.n_actives = int(lines[0])

        # arrays pre-allocation
        self.x = [0] * (self.n_actives+1)
        self.y = [0] * (self.n_actives+1)
        self.duration = [0] * (self.n_actives+1)
        self.due = [0] * (self.n_actives+1)
        self.weight = [0] * (self.n_actives+1)

        # crew's base coordinates
        base_coords = lines[1].split()
        self.x[0] = int(base_coords[0])
        self.y[0] = int(base_coords[1])

        # actives' informations
        for i in range(2, len(lines)):
            info = lines[i].split()
            self.x[i-1] = int(info[0])
            self.y[i-1] = int(info[1])
            self.duration[i-1] = int(info[2])
            self.due[i-1] = int(info[3])
            self.weight[i-1] = int(info[4])

    def calc_distances(self):
        total_nodes = self.n_actives+1
        
        # distance matrix pre-allocation
        self.distance = [[0] * total_nodes for _ in range(total_nodes)]

        for i in range(total_nodes):
            for j in range(total_nodes):
                if i != j:
                    dx = self.x[i]-self.x[j]
                    dy = self.y[i]-self.y[j]
                    self.distance[i][j] = round(math.hypot(dx, dy))