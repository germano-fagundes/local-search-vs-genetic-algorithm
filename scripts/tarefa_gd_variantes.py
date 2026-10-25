from __future__ import annotations

import argparse
import csv
import time
from statistics import mean, median

from experimentos_t3 import INSTANCIAS, RAIZ, SEED_BASE  # põe src/ no sys.path
from plot_graficos import ROTULO_INST, num

from greedy_descent import VARIANTES, greedy_descent  # noqa: E402
from repair_problem import RepairProblem  # noqa: E402

H_OTIMO_PEQUENA = 1875
ROTULO_VAR = {"best": "Best", "first": "First", "random": "Aleatório"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Comparação das variantes do Greedy Descent.")
    ap.add_argument("--operador", default="insertion", choices=["swap", "insertion", "2opt"])
    args = ap.parse_args()

    linhas, otimos_pequena = [], {v: set() for v in VARIANTES}
    otimos_globais_pequena = {v: [0, 0] for v in VARIANTES}  # [no ótimo, total]
    for nome, (arquivo, orcamento, runs) in INSTANCIAS.items():
        inst = RepairProblem(str(RAIZ / "instancias" / arquivo))
        for var in VARIANTES:
            for run in range(runs):
                t0 = time.perf_counter()
                r = greedy_descent(inst, orcamento, SEED_BASE + run, var, args.operador)
                e = r.extras
                linhas.append({"instancia": nome, "variante": var, "operador": args.operador,
                               "run_id": run, "h_melhor": r.melhor_h, "passos": e["passos"],
                               "otimos_locais": e["otimos_locais"],
                               "otimos_distintos": e["otimos_distintos"],
                               "tempo_s": f"{time.perf_counter() - t0:.3f}"})
                if nome == "pequena":
                    otimos_pequena[var] |= e["conjunto_otimos"]
                    otimos_globais_pequena[var][0] += e["h_otimos"].get(H_OTIMO_PEQUENA, 0)
                    otimos_globais_pequena[var][1] += e["otimos_locais"]
            r_ = [l for l in linhas if l["instancia"] == nome and l["variante"] == var]
            print(f"  {nome:<8} {var:<7} mediana h* = {median(l['h_melhor'] for l in r_):>9}"
                  f"  passos/run = {mean(l['passos'] for l in r_):8.1f}"
                  f"  ót.loc./run = {mean(l['otimos_locais'] for l in r_):7.1f}")

    caminho = RAIZ / "resultados" / "gd_variantes.csv"
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)

    print("\nInstância pequena (n=12), ótimos locais ao longo dos reinícios:")
    for var in VARIANTES:
        r_ = [l for l in linhas if l["instancia"] == "pequena" and l["variante"] == var]
        no_otimo, total = otimos_globais_pequena[var]
        acertos = sum(l["h_melhor"] == H_OTIMO_PEQUENA for l in r_)
        print(f"  {var:<7} runs com h*=1875: {acertos}/{len(r_)}  "
              f"ótimos locais atingidos: {total}  distintos (união das runs): "
              f"{len(otimos_pequena[var])}  com h=1875: {no_otimo} ({100 * no_otimo / total:.1f}%)")

    corpo = []
    for nome, (_, orcamento, _) in INSTANCIAS.items():
        for var in VARIANTES:
            r_ = [l for l in linhas if l["instancia"] == nome and l["variante"] == var]
            passos = mean(l["passos"] for l in r_)
            acertos = (f"{sum(l['h_melhor'] == H_OTIMO_PEQUENA for l in r_)}/{len(r_)}"
                       if nome == "pequena" else "--")
            corpo.append(
                f"{ROTULO_INST[nome].split(' ')[0] if var == 'best' else ''} & {ROTULO_VAR[var]} & "
                f"{num(median(l['h_melhor'] for l in r_))} & {num(passos)} & "
                f"{num(orcamento / passos) if passos else '--'} & "
                f"{num(mean(l['otimos_locais'] for l in r_), 1)} & {acertos} \\\\")
        corpo.append("\\addlinespace")
    tex = ["% gerado por tarefa_gd_variantes.py -- não editar à mão",
           "\\begin{tabular}{llrrrrr}", "\\toprule",
           "Inst. & Variante & $\\tilde h^\\star$ & passos & aval./passo & ót.\\,loc. "
           "& $h^\\star{=}1875$ \\\\", "\\midrule", *corpo[:-1], "\\bottomrule", "\\end{tabular}"]
    (RAIZ / "relatorio" / "figuras" / "tabela_gd_variantes.tex").write_text(
        "\n".join(tex) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
