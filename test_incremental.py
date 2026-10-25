import time
import random
from repair_problem import RepairProblem
from solution import Solution
from neighborhood_operators import Neighborhoods

def test_incremental_evaluation():
    # load small instance for testing
    try:
        instance = RepairProblem("manutencao_pequena.txt")
    except FileNotFoundError:
        print("File 'manutencao_pequena.txt' not found. Aborting.")
        return

    # create initial random permutation starting at base 0
    n = instance.n_actives
    pi = [0] + random.sample(range(1, n + 1), n)

    # initial parent solution
    parent_sol = Solution(pi, instance)

    operators = [Neighborhoods.swap, Neighborhoods.insertion, Neighborhoods.two_opt]
    match_count = 0
    time_complete = 0.0
    time_incremental = 0.0

    print("Starting 1000 random tests...")

    for _ in range(1000):
        # select random neighborhood operator
        op = random.choice(operators)
        
        # apply movement
        new_pi, start = op(parent_sol.pi)

        # complete evaluation
        t0 = time.perf_counter()
        sol_complete = Solution(new_pi, instance)
        time_complete += (time.perf_counter() - t0)

        # incremental evaluation
        t0 = time.perf_counter()
        
        # instantiate without __init__
        sol_inc = Solution.__new__(Solution)
        sol_inc.pi = new_pi
        
        # copy parent state
        sol_inc.conclusion_time = parent_sol.conclusion_time[:]
        sol_inc.acc_cost = parent_sol.acc_cost[:]
        
        # evaluate from cut point
        sol_inc.calc_conclusion_time_and_cost(instance, start)
        time_incremental += (time.perf_counter() - t0)

        # assertions and validation
        assert sol_complete.final_cost == sol_inc.final_cost, f"Cost error in {op.__name__}! {sol_complete.final_cost} != {sol_inc.final_cost}"
        assert sol_complete.conclusion_time == sol_inc.conclusion_time, f"Time error in {op.__name__}!"
        assert sol_complete.acc_cost == sol_inc.acc_cost, f"Delay error in {op.__name__}!"

        match_count += 1
        
        # set new parent for next iteration
        parent_sol = sol_complete 

    # print results
    print(f"Success: {match_count} / 1000 movements matched perfectly.")
    print(f"Accumulated time (Complete):    {time_complete:.5f} seconds")
    print(f"Accumulated time (Incremental): {time_incremental:.5f} seconds")

    speedup = time_complete / time_incremental if time_incremental > 0 else 0
    print(f"Estimated speedup: {speedup:.2f}x faster")


if __name__ == "__main__":
    test_incremental_evaluation()