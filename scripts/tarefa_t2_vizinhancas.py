from __future__ import annotations

import argparse
import csv
import random
import time
from statistics import mean, median

from experimentos_t3 import INSTANCIAS, RAIZ, SEED_BASE  # põe src/ no sys.path
from plot_graficos import ROTULO_INST, num

from greedy_descent import greedy_descent  # noqa: E402
from problema import ProblemaManutencao  # noqa: E402
from repair_problem import RepairProblem  # noqa: E402

OPERADORES = ("swap", "insertion", "2opt")
ROTULO_OP = {"swap": "Swap", "insertion": "Insertion", "2opt": "2-opt"}
FIGURAS = RAIZ / "relatorio" / "figuras"
RESULTADOS = RAIZ / "resultados"


def parte_a(runs: int) -> list[dict]:
    linhas = []
    for nome, (arquivo, orcamento, _) in INSTANCIAS.items():
        inst = RepairProblem(str(RAIZ / "instancias" / arquivo))
        for op in OPERADORES:
            for run in range(runs):
                t0 = time.perf_counter()
                r = greedy_descent(inst, orcamento, SEED_BASE + run, "first", op)
                tempo = time.perf_counter() - t0
                h_otimos = r.extras["h_otimos"]
                n_otimos = r.extras["otimos_locais"]
                media_otimos = (sum(h * c for h, c in h_otimos.items()) / n_otimos
                                if n_otimos else float("nan"))
                linhas.append({"instancia": nome, "operador": op, "run_id": run,
                               "h_melhor": r.melhor_h, "otimos_locais": n_otimos,
                               "h_medio_otimos": f"{media_otimos:.1f}",
                               "passos": r.extras["passos"], "tempo_s": f"{tempo:.3f}"})
            ult = [l for l in linhas if l["instancia"] == nome and l["operador"] == op]
            print(f"  {nome:<8} {op:<9} mediana h* = {median(l['h_melhor'] for l in ult):>9}")
    return linhas


def otimo_local(p: ProblemaManutencao, op: str, seed: int) -> list[int]:
    """Um ótimo local de op, obtido por First Improvement a partir de uma solução aleatória."""
    rng = random.Random(seed)
    s = p.solucao_aleatoria(rng)
    h, C, H = p.avaliar(s)
    melhorou = True
    while melhorou:
        melhorou = False
        for viz, start in p.sucessores(s, op):
            hv = p.avaliar_vizinho(viz, start, C, H)
            if hv < h:
                C, H = p.vetores(viz, start, C, H)
                s, h, melhorou = viz, hv, True
                break
    return s


def parte_b() -> list[dict]:
    linhas = []
    for nome, (arquivo, _, _) in INSTANCIAS.items():
        inst = RepairProblem(str(RAIZ / "instancias" / arquivo))
        for op in OPERADORES:
            p = ProblemaManutencao(inst, orcamento=10**9)
            s = otimo_local(p, op, SEED_BASE)
            h, C, H = p.avaliar(s)
            vizinhos = list(p.sucessores(s, op))
            reps = max(1, 60_000 // len(vizinhos))

            def cronometrar(f) -> float:
                t0 = time.perf_counter()
                for _ in range(reps):
                    for viz, start in vizinhos:
                        f(viz, start)
                return (time.perf_counter() - t0) / (reps * len(vizinhos)) * 1e6

            # completa = mesmo laço, recomeçando da posição 1 (sem reaproveitar o prefixo)
            t_comp = cronometrar(lambda v, st: p.avaliar_vizinho(v, 1, C, H))
            t_inc = cronometrar(lambda v, st: p.avaliar_vizinho(v, st, C, H))
            t_poda = cronometrar(lambda v, st: p.avaliar_vizinho(v, st, C, H, h))
            linhas.append({"instancia": nome, "operador": op, "vizinhos": len(vizinhos),
                           "us_completa": f"{t_comp:.3f}", "us_incremental": f"{t_inc:.3f}",
                           "us_incremental_poda": f"{t_poda:.3f}"})
            print(f"  {nome:<8} {op:<9} completa {t_comp:6.2f} us  incremental {t_inc:6.2f} us"
                  f"  +poda {t_poda:6.2f} us")
    return linhas


def gravar_csv(caminho, linhas: list[dict]) -> None:
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)


def tabela_a(linhas: list[dict]) -> None:
    corpo = []
    for nome in INSTANCIAS:
        for op in OPERADORES:
            r = [l for l in linhas if l["instancia"] == nome and l["operador"] == op]
            otimos = [float(l["h_medio_otimos"]) for l in r if l["otimos_locais"]]
            corpo.append(
                f"{ROTULO_INST[nome].split(' ')[0] if op == 'swap' else ''} & {ROTULO_OP[op]} & "
                f"{num(median(l['h_melhor'] for l in r))} & "
                f"{num(mean(otimos)) if otimos else '--'} & "
                f"{num(mean(l['otimos_locais'] for l in r), 1)} & "
                f"{num(mean(l['passos'] for l in r))} & "
                f"{num(mean(float(l['tempo_s']) for l in r), 2)} \\\\")
        corpo.append("\\addlinespace")
    tex = ["% gerado por tarefa_t2_vizinhancas.py -- não editar à mão",
           "\\begin{tabular}{llrrrrr}", "\\toprule",
           "Inst. & Viz. & $\\tilde h^\\star$ & $\\bar h_{\\text{ót.loc.}}$ & ót.\\,loc. "
           "& passos & $\\bar t$ (s) \\\\", "\\midrule", *corpo[:-1], "\\bottomrule",
           "\\end{tabular}"]
    (FIGURAS / "tabela_t2_vizinhancas.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def tabela_b(linhas: list[dict]) -> None:
    corpo = []
    for nome in INSTANCIAS:
        for op in OPERADORES:
            l = next(x for x in linhas if x["instancia"] == nome and x["operador"] == op)
            c, i, pz = (float(l[k]) for k in ("us_completa", "us_incremental",
                                               "us_incremental_poda"))
            corpo.append(f"{ROTULO_INST[nome].split(' ')[0] if op == 'swap' else ''} & "
                         f"{ROTULO_OP[op]} & {num(l['vizinhos'])} & {num(c, 2)} & {num(i, 2)} & "
                         f"{num(pz, 2)} & {num(c / i, 2)}$\\times$ & {num(c / pz, 2)}$\\times$ \\\\")
        corpo.append("\\addlinespace")
    tex = ["% gerado por tarefa_t2_vizinhancas.py -- não editar à mão",
           "\\begin{tabular}{llrrrrrr}", "\\toprule",
           "Inst. & Viz. & $|V|$ & compl. & incr. & +poda & \\multicolumn{2}{c}{speedup} \\\\",
           " & & & ($\\mu$s) & ($\\mu$s) & ($\\mu$s) & incr. & +poda \\\\",
           "\\midrule", *corpo[:-1], "\\bottomrule", "\\end{tabular}"]
    (FIGURAS / "tabela_t2_speedup.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Tarefa T2: comparação das vizinhanças.")
    ap.add_argument("--runs", type=int, default=10, help="runs por instância e vizinhança")
    args = ap.parse_args()
    RESULTADOS.mkdir(exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)

    print("Parte A: GD (First Improvement) por vizinhança")
    a = parte_a(args.runs)
    gravar_csv(RESULTADOS / "t2_vizinhancas.csv", a)
    tabela_a(a)

    print("Parte B: custo por avaliação num ótimo local")
    b = parte_b()
    gravar_csv(RESULTADOS / "t2_speedup.csv", b)
    tabela_b(b)


if __name__ == "__main__":
    main()
