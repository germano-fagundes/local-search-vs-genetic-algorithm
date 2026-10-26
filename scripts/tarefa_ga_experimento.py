from __future__ import annotations

import argparse
import csv
import os
import time
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experimentos_t3 import INSTANCIAS, PASSO_HISTORICO, RAIZ, SEED_BASE  # noqa: E402
from plot_graficos import (  # noqa: E402
    COR_INST,
    ROTULO_INST,
    TINTA,
    TINTA_SEC,
    formatar_p,
    mann_whitney_u,
    num,
)

from genetic_algorithm import genetic_algorithm  # noqa: E402
from repair_problem import RepairProblem  # noqa: E402

FIGURAS = RAIZ / "relatorio" / "figuras"
RESULTADOS = RAIZ / "resultados"


def executar_experimento(runs_override: int | None = None) -> tuple[list[dict], dict[str, list[list[float]]]]:
    linhas = []
    diversidade_por_inst: dict[str, list[list[float]]] = {inst: [] for inst in INSTANCIAS}

    for nome, (arquivo, orcamento, runs_padrao) in INSTANCIAS.items():
        runs = runs_override if runs_override is not None else runs_padrao
        inst = RepairProblem(str(RAIZ / "instancias" / arquivo))
        for variante, pc in (("com_ox", 0.85), ("sem_ox", 0.0)):
            for run in range(runs):
                t0 = time.perf_counter()
                r = genetic_algorithm(inst, orcamento, SEED_BASE + run, pc=pc)
                tempo = time.perf_counter() - t0
                e = r.extras
                cruz = e["cruzamentos"]
                taxa_ambos = e["filhos_melhores_ambos"] / cruz if cruz else 0.0
                taxa_um = e["filhos_melhores_um"] / cruz if cruz else 0.0
                linhas.append({
                    "instancia": nome,
                    "variante": variante,
                    "pc": pc,
                    "run_id": run,
                    "h_melhor": r.melhor_h,
                    "geracoes": e["geracoes"],
                    "cruzamentos": cruz,
                    "filhos_melhores_ambos": e["filhos_melhores_ambos"],
                    "filhos_melhores_um": e["filhos_melhores_um"],
                    "taxa_melhor_ambos": f"{taxa_ambos:.6f}",
                    "taxa_melhor_um": f"{taxa_um:.6f}",
                    "diversidade_final": f"{e['diversidade_final']:.4f}",
                    "tempo_s": f"{tempo:.3f}",
                })
                if variante == "com_ox":
                    diversidade_por_inst[nome].append(e["historico_diversidade"])
            sub = [l for l in linhas if l["instancia"] == nome and l["variante"] == variante]
            print(f"  {nome:<8} {variante:<7} (Pc={pc:.2f})  mediana h* = "
                  f"{median(l['h_melhor'] for l in sub):>8.0f}  "
                  f"div_final = {100 * mean(float(l['diversidade_final']) for l in sub):5.1f}%")
    return linhas, diversidade_por_inst


def figura_diversidade(diversidade_por_inst: dict[str, list[list[float]]], saida: str) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for inst, curvas in diversidade_por_inst.items():
        if not curvas:
            continue
        m = np.array(curvas, dtype=float) * 100.0
        x = np.arange(1, m.shape[1] + 1) * PASSO_HISTORICO / INSTANCIAS[inst][1]
        q1, med, q3 = np.percentile(m, [25, 50, 75], axis=0)
        ax.fill_between(x, q1, q3, color=COR_INST[inst], alpha=0.18, linewidth=0)
        ax.plot(x, med, color=COR_INST[inst], label=ROTULO_INST[inst])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Fração do orçamento $B$ consumida")
    ax.set_ylabel("Indivíduos únicos na população (%)")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(os.path.join(saida, "diversidade_ga.pdf"))
    plt.close(fig)


def tabela_ga_experimento(linhas: list[dict], saida: str) -> None:
    corpo = []
    for nome in INSTANCIAS:
        com = [l for l in linhas if l["instancia"] == nome and l["variante"] == "com_ox"]
        sem = [l for l in linhas if l["instancia"] == nome and l["variante"] == "sem_ox"]
        h_com = [int(l["h_melhor"]) for l in com]
        h_sem = [int(l["h_melhor"]) for l in sem]
        _, p = mann_whitney_u(h_com, h_sem)
        taxa_ambos = 100.0 * mean(float(l["taxa_melhor_ambos"]) for l in com)
        taxa_um = 100.0 * mean(float(l["taxa_melhor_um"]) for l in com)
        div_f = 100.0 * mean(float(l["diversidade_final"]) for l in com)
        corpo.append(
            f"{ROTULO_INST[nome].split(' ')[0]} & {num(median(h_com))} & {num(median(h_sem))} & "
            f"${formatar_p(p)}$ & {num(taxa_ambos, 2)}\\% & {num(taxa_um, 1)}\\% & "
            f"{num(div_f, 1)}\\% \\\\"
        )
    tex = [
        "% gerado por tarefa_ga_experimento.py -- não editar à mão",
        "\\begin{tabular}{lrrrrrr}", "\\toprule",
        "Inst. & $\\tilde h^\\star$ (OX) & $\\tilde h^\\star$ ($P_c{=}0$) & $p$-valor "
        "& filho $<$ pais & filho $<$ 1 pai & div.\\,final \\\\",
        "\\midrule", *corpo, "\\bottomrule", "\\end{tabular}",
    ]
    with open(os.path.join(saida, "tabela_ga_experimento.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Experimento adicional do GA (OX vs sem OX e diversidade).")
    ap.add_argument("--runs", type=int, default=None, help="sobrescreve o número de runs por instância")
    args = ap.parse_args()

    os.makedirs(RESULTADOS, exist_ok=True)
    os.makedirs(FIGURAS, exist_ok=True)

    linhas, div_por_inst = executar_experimento(args.runs)
    with open(RESULTADOS / "ga_experimento.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)

    figura_diversidade(div_por_inst, str(FIGURAS))
    tabela_ga_experimento(linhas, str(FIGURAS))
    print(f"Resultados e figuras do GA gravados em {RESULTADOS} e {FIGURAS}")


if __name__ == "__main__":
    main()

