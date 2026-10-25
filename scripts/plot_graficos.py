from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from statistics import mean, median, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import numpy as np  

from experimentos_t3 import INSTANCIAS, PASSO_HISTORICO, RAIZ  

H_OTIMO_PEQUENA = 1875  

ROTULO_INST = {"pequena": "Pequena ($n=12$)", "media": "Intermediária ($n=50$)",
               "grande": "Grande ($n=150$)"}
ROTULO_ALG = {"GD": "Greedy Descent + RR", "SA": "Simulated Annealing", "GA": "Algoritmo Genético"}
COR_ALG = {"GD": "#2a78d6", "SA": "#eb6834", "GA": "#1baf7a"}
COR_INST = {"pequena": "#2a78d6", "media": "#eb6834", "grande": "#1baf7a"}
TINTA, TINTA_SEC, GRADE = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": TINTA_SEC, "axes.labelcolor": TINTA, "xtick.color": TINTA_SEC,
    "ytick.color": TINTA_SEC, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.5, "lines.linewidth": 1.4,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def ler_csv(caminho: str) -> list[dict]:
    if not os.path.exists(caminho):
        return []
    with open(caminho, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def deduplicar(linhas: list[dict], chave: tuple[str, ...]) -> list[dict]:
    """Mantém a última ocorrência de cada chave (runs reexecutadas após interrupção)."""
    return list({tuple(l[k] for k in chave): l for l in linhas}.values())


def formatar_milhar(x: float, _=None) -> str:
    if x >= 1000:
        return f"{x / 1000:g}k"
    return f"{x:g}"


# --------------------------------------------------------------------------- convergência
def painel_convergencia(ax, resultados: list[dict], inst: str) -> None:
    orcamento = INSTANCIAS[inst][1]
    for alg in ROTULO_ALG:
        hist = [list(map(int, l["historico_melhor_h"].split(";")))
                for l in resultados if l["instancia"] == inst and l["algoritmo"] == alg]
        if not hist:
            continue
        m = np.array(hist, dtype=float)
        x = np.arange(1, m.shape[1] + 1) * PASSO_HISTORICO
        q1, med, q3 = np.percentile(m, [25, 50, 75], axis=0)
        ax.fill_between(x, q1, q3, color=COR_ALG[alg], alpha=0.18, linewidth=0)
        ax.plot(x, med, color=COR_ALG[alg], label=f"{ROTULO_ALG[alg]} ({len(hist)} runs)")
        pop = [list(map(float, l["historico_media_populacao"].split(";")))
               for l in resultados if l["instancia"] == inst and l["algoritmo"] == alg
               and l.get("historico_media_populacao")]
        if pop:
            ax.plot(x[:len(pop[0])], np.median(np.array(pop), axis=0), color=COR_ALG[alg],
                    linestyle=(0, (4, 2)), linewidth=1.0, label=f"{ROTULO_ALG[alg]} (média pop.)")
        ax.annotate(f"{med[-1]:,.0f}".replace(",", "."), (x[-1], med[-1]),
                    xytext=(-2, 4), textcoords="offset points", ha="right",
                    fontsize=6.5, color=TINTA)
    ax.set_yscale("log")
    ax.set_xlim(0, orcamento)
    ax.xaxis.set_major_formatter(formatar_milhar)
    ax.set_title(ROTULO_INST[inst], color=TINTA)
    ax.set_xlabel("Avaliações de $h$ gastas")


def figuras_convergencia(resultados: list[dict], saida: str) -> None:
    fig, eixos = plt.subplots(1, 3, figsize=(7.0, 2.2))
    for ax, inst in zip(eixos, INSTANCIAS):
        painel_convergencia(ax, resultados, inst)
    eixos[0].set_ylabel("Melhor $h$ (mediana, escala log)")
    handles, labels = eixos[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(os.path.join(saida, "convergencia.pdf"))
    plt.close(fig)

    for inst in INSTANCIAS:
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        painel_convergencia(ax, resultados, inst)
        ax.set_ylabel("Melhor $h$ (mediana, escala log)")
        ax.legend(frameon=False, loc="upper right")
        fig.savefig(os.path.join(saida, f"convergencia_{inst}.pdf"))
        plt.close(fig)


# --------------------------------------------------------------------------- aceitação SA
def figura_aceitacao(diagnostico: list[dict], saida: str) -> None:
    if not diagnostico:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for inst in INSTANCIAS:
        por_run = defaultdict(dict)
        for l in diagnostico:
            if l["instancia"] == inst and l["taxa_aceitacao_piora"] != "":
                por_run[l["run_id"]][int(l["avaliacoes"])] = float(l["taxa_aceitacao_piora"])
        if not por_run:
            continue
        janelas = sorted(set().union(*por_run.values()))
        m = np.array([[r.get(j, np.nan) for j in janelas] for r in por_run.values()])
        x = np.array(janelas) / INSTANCIAS[inst][1]
        q1, med, q3 = np.nanpercentile(m, [25, 50, 75], axis=0)
        ax.fill_between(x, q1, q3, color=COR_INST[inst], alpha=0.18, linewidth=0)
        ax.plot(x, med, color=COR_INST[inst], label=ROTULO_INST[inst])
    ax.axhline(0.80, color=TINTA_SEC, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.text(0.99, 0.82, "$p_0 = 0{,}80$", ha="right", va="bottom", fontsize=6.5, color=TINTA_SEC,
            transform=ax.get_yaxis_transform())
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Fração do orçamento $B$ consumida")
    ax.set_ylabel("Pioras aceitas / pioras propostas")
    ax.legend(frameon=False, loc="center right")
    fig.savefig(os.path.join(saida, "aceitacao_sa.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- tabelas LaTeX
def num(x: float, casas: int = 0) -> str:
    """Número no padrão brasileiro com \\, como separador de milhar."""
    s = f"{x:,.{casas}f}"
    return s.replace(",", "X").replace(".", "{,}").replace("X", "\\,")


def tabela_parametros(parametros: list[dict], resultados: list[dict], saida: str) -> None:
    linhas = []
    for inst, (_, orcamento, runs) in INSTANCIAS.items():
        p = [l for l in parametros if l["instancia"] == inst]
        if not p:
            continue
        hs = [int(l["h_melhor_final"]) for l in resultados
              if l["instancia"] == inst and l["algoritmo"] == "SA"]
        dh, t0, tf = (median(float(l[k]) for l in p) for k in ("dh_medio", "T0", "Tf"))
        alpha = median(float(l["alpha"]) for l in p)
        n = {"pequena": 12, "media": 50, "grande": 150}[inst]
        linhas.append(f"{n} & {num(orcamento)} & {len(p)} & {num(dh)} & {num(t0)} & "
                      f"{num(tf, 1)} & {num(alpha, 7)} & {num(min(hs))} & {num(median(hs))} \\\\")
    tex = [
        "% gerado por plot_graficos.py -- não editar à mão",
        "\\begin{tabular}{rrrrrrrrr}", "\\toprule",
        "$n$ & $B$ & runs & $\\overline{\\Delta h}^{+}$ & $T_0$ & $T_f$ & $\\alpha$ "
        "& $h^\\star_{\\min}$ & $\\tilde h^\\star$ \\\\",
        "\\midrule", *linhas, "\\bottomrule", "\\end{tabular}",
    ]
    with open(os.path.join(saida, "tabela_parametros_sa.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")


def tabela_resultados(resultados: list[dict], saida: str) -> None:
    linhas = []
    for inst in INSTANCIAS:
        for alg in ROTULO_ALG:
            r = [l for l in resultados if l["instancia"] == inst and l["algoritmo"] == alg]
            if not r:
                continue
            hs = [int(l["h_melhor_final"]) for l in r]
            ts = [float(l["tempo_parede_segundos"]) for l in r]
            dp = stdev(hs) if len(hs) > 1 else 0.0
            gap = (f"{num(100 * (mean(hs) - H_OTIMO_PEQUENA) / H_OTIMO_PEQUENA, 2)}\\%"
                   if inst == "pequena" else "--")
            linhas.append(f"{ROTULO_INST[inst].split(' ')[0]} & {alg} & {len(r)} & "
                          f"{num(mean(hs))} & {num(dp)} & {num(min(hs))} & {num(median(hs))} & "
                          f"{gap} & {num(INSTANCIAS[inst][1])} & {num(mean(ts), 2)} \\\\")
    tex = [
        "% gerado por plot_graficos.py -- não editar à mão",
        "\\begin{tabular}{llrrrrrrrr}", "\\toprule",
        "Inst. & Alg. & runs & média & d.p. & mín. & mediana & gap & $B$ & $\\bar t$ (s) \\\\",
        "\\midrule", *linhas, "\\bottomrule", "\\end{tabular}",
    ]
    with open(os.path.join(saida, "tabela_resultados_t3.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Gera figuras e tabelas LaTeX a partir dos CSVs do T3.")
    ap.add_argument("--entrada", default=str(RAIZ / "resultados"), help="diretório dos CSVs do T3")
    ap.add_argument("--saida", default=str(RAIZ / "relatorio" / "figuras"), help="diretório das figuras e tabelas")
    args = ap.parse_args()
    os.makedirs(args.saida, exist_ok=True)

    resultados = deduplicar(ler_csv(os.path.join(args.entrada, "resultados_t3.csv")),
                            ("algoritmo", "instancia", "run_id"))
    parametros = deduplicar(ler_csv(os.path.join(args.entrada, "parametros_sa.csv")),
                            ("instancia", "run_id"))
    diagnostico = deduplicar(ler_csv(os.path.join(args.entrada, "diagnostico_sa.csv")),
                             ("instancia", "run_id", "avaliacoes"))
    if not resultados:
        raise SystemExit("resultados_t3.csv vazio ou ausente -- rode scripts/experimentos_t3.py antes.")

    figuras_convergencia(resultados, args.saida)
    figura_aceitacao(diagnostico, args.saida)
    tabela_parametros(parametros, resultados, args.saida)
    tabela_resultados(resultados, args.saida)
    print(f"Figuras e tabelas gravadas em {args.saida}/")


if __name__ == "__main__":
    main()
