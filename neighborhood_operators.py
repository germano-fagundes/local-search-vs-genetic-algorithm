from random import randrange

class Neighborhoods:
    @classmethod
    def check_parameters(cls, total_nodes: int, p1: int | None=None, p2: int | None=None):
        if (isinstance(p1, int) and (p1 > total_nodes or p1 <= 0 or p1)) or (isinstance(p2, int) and (p2 > total_nodes or p2 <= 0)) or (p1 == p2):
            raise ValueError('The positions for swapping actives must be different from each other, greater than zero and less than the amount of the instance.')

        if p1 == None:
            p1 = randrange(1, total_nodes)
            if isinstance(p2, int):
                while p1 == p2:
                    p1 = randrange(1, total_nodes)
        
        if p2 == None:
            p2 = randrange(1, total_nodes)
            while p1 == p2:
                p2 = randrange(1, total_nodes)

        return { "p1": p1, "p2": p2 }

    @classmethod
    def swap(cls, pi: list[int], p1: int | None = None, p2: int | None = None):
        parameters = cls.check_parameters(len(pi), p1, p2)
        p1, p2 = parameters["p1"], parameters["p2"]

        aux = pi[p1]
        pi[p1] = pi[p2]
        pi[p2] = aux

        return pi

    @classmethod
    def insertion(cls, pi: list[int], active: int | None=None, position: int | None=None):
        parameters = cls.check_parameters(len(pi), active, position)
        active, position = parameters["p1"], parameters["p2"]

        diff = abs(active - position)
        