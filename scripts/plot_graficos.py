from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter, defaultdict
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
    finais = []  # (h final mediano, x final) de cada algoritmo, para rotular sem colisão
    for alg in ROTULO_ALG:
        hist = [list(map(int, l["historico_melhor_h"].split(";")))
                for l in resultados if l["instancia"] == inst and l["algoritmo"] == alg]
        if not hist:
            continue
        m = np.array(hist, dtype=float)
        x = np.arange(1, m.shape[1] + 1) * PASSO_HISTORICO
        q1, med, q3 = np.percentile(m, [25, 50, 75], axis=0)
        ax.fill_between(x, q1, q3, color=COR_ALG[alg], alpha=0.18, linewidth=0)
        ax.plot(x, med, color=COR_ALG[alg], label=ROTULO_ALG[alg])
        pop = [list(map(float, l["historico_media_populacao"].split(";")))
               for l in resultados if l["instancia"] == inst and l["algoritmo"] == alg
               and l.get("historico_media_populacao")]
        if pop:
            ax.plot(x[:len(pop[0])], np.median(np.array(pop), axis=0), color=COR_ALG[alg],
                    linestyle=(0, (4, 2)), linewidth=1.0, label=f"{ROTULO_ALG[alg]} (média pop.)")
        finais.append((med[-1], x[-1]))
    # o menor valor final é rotulado abaixo da curva; os maiores, acima (sem sobreposição)
    finais = sorted({round(h): (h, x) for h, x in finais}.values(), reverse=True)
    for k, (h_final, x_final) in enumerate(finais):
        abaixo = (k == len(finais) - 1) and len(finais) > 1
        ax.annotate(f"{h_final:,.0f}".replace(",", "."), (x_final, h_final),
                    xytext=(-2, -4 if abaixo else 4), textcoords="offset points", ha="right",
                    va="top" if abaixo else "bottom", fontsize=6.5, color=TINTA)
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
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
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


# --------------------------------------------------------------------------- boxplots e Mann-Whitney
def figura_boxplots(resultados: list[dict], saida: str) -> None:
    fig, eixos = plt.subplots(1, 3, figsize=(7.0, 2.15))
    algs = list(ROTULO_ALG)
    for ax, inst in zip(eixos, INSTANCIAS):
        dados = [[int(l["h_melhor_final"]) for l in resultados
                  if l["instancia"] == inst and l["algoritmo"] == a] for a in algs]
        pos = np.arange(1, len(algs) + 1)
        bp = ax.boxplot(dados, positions=pos, widths=0.46, patch_artist=True,
                        showfliers=False, medianprops={"color": TINTA, "linewidth": 1.3},
                        boxprops={"linewidth": 0.9}, whiskerprops={"linewidth": 0.9, "color": TINTA_SEC},
                        capprops={"linewidth": 0.9, "color": TINTA_SEC})
        for patch, a in zip(bp["boxes"], algs):
            patch.set_facecolor(COR_ALG[a])
            patch.set_alpha(0.32)
            patch.set_edgecolor(COR_ALG[a])
        rng = np.random.default_rng(42)
        for i, (vals, a) in enumerate(zip(dados, algs), start=1):
            if vals:
                jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(np.full(len(vals), i) + jitter, vals, s=9, color=COR_ALG[a],
                           alpha=0.75, zorder=3, linewidths=0)
        ax.set_xticks(pos)
        ax.set_xticklabels(algs)
        ax.set_title(ROTULO_INST[inst], color=TINTA)
        ax.yaxis.set_major_formatter(formatar_milhar)
    eixos[0].set_ylabel("Melhor $h$ final")
    fig.tight_layout()
    fig.savefig(os.path.join(saida, "boxplots.pdf"))
    plt.close(fig)


def mann_whitney_u(x: list[float], y: list[float]) -> tuple[float, float]:
    """Teste U de Mann-Whitney bicaudal com correção de empates e de continuidade."""
    n1, n2 = len(x), len(y)
    n = n1 + n2
    combinado = sorted((v, i) for i, v in enumerate(list(x) + list(y)))
    ranks = [0.0] * n
    pos = 0
    while pos < n:
        fim = pos
        while fim < n and combinado[fim][0] == combinado[pos][0]:
            fim += 1
        r_medio = (pos + 1 + fim) / 2.0
        for k in range(pos, fim):
            ranks[combinado[k][1]] = r_medio
        pos = fim
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    mu = n1 * n2 / 2.0
    contagens = Counter(v for v, _ in combinado)
    soma_empates = sum(t ** 3 - t for t in contagens.values())
    var = (n1 * n2 / 12.0) * ((n + 1) - soma_empates / (n * (n - 1)))
    if var <= 0:
        return u, 1.0
    z = max(0.0, abs(u - mu) - 0.5) / math.sqrt(var)
    p = math.erfc(z / math.sqrt(2.0))
    return u1, min(1.0, p)


def formatar_p(p: float) -> str:
    if p >= 0.01:
        return f"{p:.2f}".replace(".", "{,}")
    exp = int(math.floor(math.log10(p))) if p > 0 else -16
    mant = p / (10 ** exp)
    return f"{mant:.1f}".replace(".", "{,}") + f"\\times 10^{{{exp}}}"


def tabela_mannwhitney(resultados: list[dict], saida: str, entrada: str) -> None:
    pares = [("GD", "SA"), ("SA", "GA"), ("GD", "GA")]
    linhas_csv = []
    linhas_tex = []
    for inst in INSTANCIAS:
        amostras = {a: [int(l["h_melhor_final"]) for l in resultados
                        if l["instancia"] == inst and l["algoritmo"] == a]
                    for a in ROTULO_ALG}
        if not all(amostras.values()):
            continue
        cels = []
        for a1, a2 in pares:
            u, p = mann_whitney_u(amostras[a1], amostras[a2])
            vencedor = "empate" if p >= 0.05 else (a1 if median(amostras[a1]) < median(amostras[a2]) else a2)
            linhas_csv.append({"instancia": inst, "par": f"{a1}_vs_{a2}",
                               "U": f"{u:.1f}", "p_valor": f"{p:.6e}", "vencedor_5pct": vencedor})
            cels.append(f"${formatar_p(p)}$ ({vencedor})")
        linhas_tex.append(f"{ROTULO_INST[inst].split(' ')[0]} & " + " & ".join(cels) + " \\\\")

    if linhas_csv:
        with open(os.path.join(entrada, "testes_hipotese.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas_csv[0]))
            w.writeheader()
            w.writerows(linhas_csv)
        tex = [
            "% gerado por plot_graficos.py -- não editar à mão",
            "\\begin{tabular}{lccc}", "\\toprule",
            "Instância & GD $\\times$ SA & SA $\\times$ GA & GD $\\times$ GA \\\\",
            "\\midrule", *linhas_tex, "\\bottomrule", "\\end{tabular}",
        ]
        with open(os.path.join(saida, "tabela_mannwhitney.tex"), "w", encoding="utf-8") as f:
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
    figura_boxplots(resultados, args.saida)
    tabela_parametros(parametros, resultados, args.saida)
    tabela_resultados(resultados, args.saida)
    tabela_mannwhitney(resultados, args.saida, args.entrada)
    print(f"Figuras e tabelas gravadas em {args.saida}/")


if __name__ == "__main__":
    main()
