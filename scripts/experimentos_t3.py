"""Orquestrador experimental da Tarefa T3: 3 algoritmos x 3 instâncias.

Uso (a partir da raiz do repositório):
    python scripts/experimentos_t3.py                  # roda tudo o que estiver disponível
    python scripts/experimentos_t3.py --algoritmos SA  # só o Simulated Annealing
    python scripts/experimentos_t3.py -j 6             # 6 processos em paralelo

Cada algoritmo é um módulo com a função
    executar(instancia: RepairProblem, orcamento: int, seed: int) -> avaliador.ResultadoBusca
em src/. Módulos ainda inexistentes (ex.: GD/GA em desenvolvimento) são pulados com aviso.
O GA pode devolver também extras["historico_media_populacao"] (h médio da população a
cada 500 avaliações), gravado na coluna historico_media_populacao.

Saídas em resultados/ (gravadas linha a linha, logo após cada run -- uma interrupção não perde
o que já rodou, e uma nova chamada retoma do ponto onde parou):
    resultados_t3.csv   algoritmo, instancia, n, run_id, seed, orcamento, avaliacoes,
                        h_melhor_final, tempo_parede_segundos, historico_melhor_h
                        (melhor h a cada 500 avaliações, separado por ';')
    parametros_sa.csv   T0, Tf, alpha e dh_medio calibrados em cada run do SA
    diagnostico_sa.csv  taxa de aceitação de pioras e T por janela de 1000 avaliações

Sementes: seed = SEED_BASE + run_id, iguais para todos os algoritmos de uma mesma
instância. Isso garante reprodutibilidade, mas não pareamento: cada algoritmo
consome o gerador de forma diferente, e as amostras devem ser tratadas como
independentes (Mann-Whitney).
"""
from __future__ import annotations

import argparse
import csv
import importlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from repair_problem import RepairProblem  # noqa: E402

SEED_BASE = 20260000
PASSO_HISTORICO = 500

# instância -> (arquivo, orçamento B, número de runs)
INSTANCIAS = {
    "pequena": ("manutencao_pequena.txt", 50_000, 30),
    "media": ("manutencao_media.txt", 150_000, 30),
    "grande": ("manutencao_grande.txt", 300_000, 10),
}

# sigla -> módulo que expõe executar(instancia, orcamento, seed)
ALGORITMOS = {
    "GD": "greedy_descent",
    "SA": "simulated_annealing",
    "GA": "genetic_algorithm",
}

CAMPOS_RESULTADOS = ["algoritmo", "instancia", "n", "run_id", "seed", "orcamento", "avaliacoes",
                     "h_melhor_final", "tempo_parede_segundos", "historico_melhor_h",
                     "historico_media_populacao"]
CAMPOS_PARAMETROS = ["instancia", "run_id", "seed", "vizinhanca", "dh_medio", "T0", "Tf", "alpha"]
CAMPOS_DIAGNOSTICO = ["instancia", "run_id", "seed", "avaliacoes", "temperatura",
                      "taxa_aceitacao_piora"]


def _executar_run(sigla: str, modulo: str, nome_inst: str, run_id: int) -> dict:
    """Executa uma única corrida (função de topo para ser serializável pelo pool)."""
    arquivo, orcamento, _ = INSTANCIAS[nome_inst]
    instancia = RepairProblem(str(RAIZ / "instancias" / arquivo))
    seed = SEED_BASE + run_id
    executar = importlib.import_module(modulo).executar

    inicio = time.perf_counter()
    r = executar(instancia, orcamento, seed)
    tempo = time.perf_counter() - inicio

    return {
        "algoritmo": sigla, "instancia": nome_inst, "n": instancia.n_actives,
        "run_id": run_id, "seed": seed, "orcamento": orcamento, "avaliacoes": r.avaliacoes,
        "h_melhor_final": r.melhor_h, "tempo_parede_segundos": f"{tempo:.4f}",
        "historico_melhor_h": ";".join(map(str, r.historico)),
        "historico_media_populacao": ";".join(
            f"{v:.1f}" for v in r.extras.get("historico_media_populacao", [])),
        "_extras": r.extras,
    }


class GravadorCSV:
    """Abre os CSVs em modo append, escrevendo o cabeçalho só em arquivos novos.

    Se o arquivo existente tiver um cabeçalho antigo (colunas adicionadas depois),
    ele é reescrito com as colunas atuais, deixando as novas em branco.
    """

    def __init__(self, caminho: str, campos: list[str]) -> None:
        novo = not os.path.exists(caminho) or os.path.getsize(caminho) == 0
        if not novo:
            with open(caminho, newline="", encoding="utf-8") as f:
                leitor = csv.DictReader(f)
                if leitor.fieldnames != campos:
                    linhas = list(leitor)
                    novo = True
            if novo:
                with open(caminho, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=campos, restval="")
                    w.writeheader()
                    w.writerows(linhas)
                novo = False
        self.arquivo = open(caminho, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.arquivo, fieldnames=campos)
        if novo:
            self.writer.writeheader()

    def gravar(self, linhas: list[dict]) -> None:
        self.writer.writerows(linhas)
        self.arquivo.flush()

    def fechar(self) -> None:
        self.arquivo.close()


def runs_concluidas(caminho: str) -> set[tuple[str, str, int]]:
    if not os.path.exists(caminho):
        return set()
    with open(caminho, newline="", encoding="utf-8") as f:
        return {(l["algoritmo"], l["instancia"], int(l["run_id"])) for l in csv.DictReader(f)}


def algoritmos_disponiveis(siglas: list[str]) -> dict[str, str]:
    disponiveis = {}
    for sigla in siglas:
        modulo = ALGORITMOS[sigla]
        try:
            if not hasattr(importlib.import_module(modulo), "executar"):
                raise ImportError(f"{modulo}.py não define executar()")
            disponiveis[sigla] = modulo
        except ImportError as e:
            print(f"[aviso] {sigla} pulado: {e}")
    return disponiveis


def main() -> None:
    ap = argparse.ArgumentParser(description="Orquestrador experimental da Tarefa T3.")
    ap.add_argument("--algoritmos", nargs="+", default=list(ALGORITMOS), choices=list(ALGORITMOS))
    ap.add_argument("--instancias", nargs="+", default=list(INSTANCIAS), choices=list(INSTANCIAS))
    ap.add_argument("-j", "--processos", type=int, default=1,
                    help="corridas em paralelo (1 = tempos de parede sem contenção de CPU)")
    ap.add_argument("--saida", default=str(RAIZ / "resultados"), help="diretório dos CSVs")
    args = ap.parse_args()

    os.makedirs(args.saida, exist_ok=True)
    caminho_res = os.path.join(args.saida, "resultados_t3.csv")
    feitas = runs_concluidas(caminho_res)

    tarefas = [(sigla, modulo, inst, run_id)
               for sigla, modulo in algoritmos_disponiveis(args.algoritmos).items()
               for inst in args.instancias
               for run_id in range(INSTANCIAS[inst][2])
               if (sigla, inst, run_id) not in feitas]
    print(f"{len(tarefas)} corridas pendentes ({len(feitas)} já em {caminho_res}).")

    res = GravadorCSV(caminho_res, CAMPOS_RESULTADOS)
    par = GravadorCSV(os.path.join(args.saida, "parametros_sa.csv"), CAMPOS_PARAMETROS)
    diag = GravadorCSV(os.path.join(args.saida, "diagnostico_sa.csv"), CAMPOS_DIAGNOSTICO)

    def registrar(linha: dict) -> None:
        extras = linha.pop("_extras")
        if linha["algoritmo"] == "SA":
            base = {k: linha[k] for k in ("instancia", "run_id", "seed")}
            par.gravar([{**base, **{k: extras[k] for k in CAMPOS_PARAMETROS[3:]}}])
            diag.gravar([{**base, "avaliacoes": (i + 1) * extras["janela"],
                          "temperatura": f"{T:.6g}",
                          "taxa_aceitacao_piora": "" if taxa is None else f"{taxa:.6f}"}
                         for i, (T, taxa) in enumerate(zip(extras["temperatura"],
                                                           extras["taxa_aceitacao"]))])
        # resultados_t3.csv por último: é ele que marca a run como concluída
        res.gravar([linha])
        print(f"  {linha['algoritmo']:>2} {linha['instancia']:<8} run {linha['run_id']:>2}  "
              f"h* = {linha['h_melhor_final']:>7}  ({linha['tempo_parede_segundos']} s)")

    try:
        if args.processos == 1:
            for t in tarefas:
                registrar(_executar_run(*t))
        else:
            with ProcessPoolExecutor(args.processos) as pool:
                for linha in pool.map(_executar_run, *zip(*tarefas)) if tarefas else []:
                    registrar(linha)
    finally:
        for g in (res, par, diag):
            g.fechar()


if __name__ == "__main__":
    main()
