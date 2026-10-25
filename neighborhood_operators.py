from random import randrange
class Neighborhoods:
    @classmethod
    def check_parameters(cls, total_nodes: int, p1: int | None=None, p2: int | None=None):
        if (isinstance(p1, int) and (p1 >= total_nodes or p1 <= 0)) or (isinstance(p2, int) and (p2 >= total_nodes or p2 <= 0)) or (isinstance(p1, int) and isinstance(p2, int)and p1 == p2):
            raise ValueError('The neighborhood parameters must be different from each other, greater than zero and less than the amount of the instance.')

        if p1 is None:
            p1 = randrange(1, total_nodes)
            if isinstance(p2, int):
                while p1 == p2:
                    p1 = randrange(1, total_nodes)
        
        if p2 is None:
            p2 = randrange(1, total_nodes)
            while p1 == p2:
                p2 = randrange(1, total_nodes)

        return p1, p2

    @classmethod
    def swap(cls, pi: list[int], p1: int | None = None, p2: int | None = None):
        p1, p2 = cls.check_parameters(len(pi), p1, p2)
        new_pi = pi[:]

        new_pi[p1], new_pi[p2] = new_pi[p2], new_pi[p1]

        evaluation_start = min(p1, p2)
        return new_pi, evaluation_start

    @classmethod
    def insertion(cls, pi: list[int], active: int | None=None, position: int | None=None):
        active, position = cls.check_parameters(len(pi), active, position)

        new_pi = pi[:]
        value = new_pi.pop(active)
        new_pi.insert(position, value)

        evaluation_start = min(active, position)
        return new_pi, evaluation_start

    @classmethod
    def two_opt(cls, pi: list[int], begin: int | None=None, end: int | None=None):
        begin, end = cls.check_parameters(len(pi), begin, end)

        if begin > end:
            begin, end = end, begin

        new_pi = pi[:]

        new_pi[begin:end+1] = reversed(new_pi[begin:end+1])

        evaluation_start = begin
        return new_pi, evaluation_start