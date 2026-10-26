# TP2 — Busca Local e Algoritmos Genéticos no Sequenciamento de Manutenção de Ativos

Trabalho Prático 2 da disciplina **Fundamentos de Inteligência Artificial (UFMG)**.
Comparação experimental entre três metaheurísticas sob orçamento fixo de avaliações da função objetivo ($B$):
- **(A) Greedy Descent com Random Restart** (`src/greedy_descent.py`)
- **(B) Simulated Annealing com Calibração Adaptativa** (`src/simulated_annealing.py`)
- **(C) Algoritmo Genético para Permutações** (`src/genetic_algorithm.py`)

---

## Estrutura do Repositório

```text
local-search-vs-genetic-algorithm/
├── instancias/                  # Arquivos de instância (n = 4, 12, 50, 150)
├── src/
│   ├── repair_problem.py        # Parser das instâncias e pré-cálculo da matriz t_ij
│   ├── solution.py              # Implementação direta de referência de C_k e h(pi)
│   ├── neighborhood_operators.py# Operadores Swap, Insertion e 2-opt
│   ├── avaliador.py             # Controle estrito do orçamento B, avaliação incremental e poda
│   ├── problema.py              # Interface comum ProblemaManutencao (algoritmo != problema)
│   ├── greedy_descent.py        # (A) Greedy Descent (Best, First e Random Step) + Random Restart
│   ├── simulated_annealing.py   # (B) Simulated Annealing (Gibbs + calibração térmica de T_0)
│   └── genetic_algorithm.py     # (C) Algoritmo Genético (Torneio, OX, Mutação, Elitismo)
├── scripts/
│   ├── tarefa_t1_validacao_h.py # Tarefa T1: validação de h(pi) nos casos de referência
│   ├── tarefa_t2_vizinhancas.py # Tarefa T2: comparação das 3 vizinhanças e speedup incremental
│   ├── tarefa_gd_variantes.py   # Comparação Best vs First vs Random Step e ótimos locais (n=12)
│   ├── experimentos_t3.py       # Tarefa T3: orquestrador das 9 combinações (GD, SA, GA)
│   ├── tarefa_ga_experimento.py # Experimento adicional do GA (OX ligado vs desligado) e diversidade
│   └── plot_graficos.py         # Gera todas as figuras PDF, boxplots, tabelas LaTeX e Mann-Whitney
├── tests/
│   ├── test_problema.py         # Testes unitários (T1, avaliação incremental, poda, orçamento, OX, GD, GA)
│   └── test_incremental.py      # Teste de 1.000 movimentos aleatórios (completa vs incremental)
├── resultados/                  # Arquivos CSV gerados pelos experimentos
└── relatorio/
    ├── relatorio.tex            # Código-fonte LaTeX do relatório final
    ├── relatorio.pdf            # Relatório compilado
    └── figuras/                 # Figuras (.pdf) e tabelas (.tex) geradas automaticamente
```

---

## Requisitos

- **Python 3.10+** com `numpy` e `matplotlib` instalados:
  ```bash
  pip install numpy matplotlib
  ```

---

## Passo a Passo para Reproduzir Testes, Tabelas e Figuras

Todos os comandos abaixo devem ser executados a partir da pasta `local-search-vs-genetic-algorithm/` (ou passando o caminho correspondente a partir da raiz):

### 1. Executar os Testes Automatizados e a Validação T1
Valida os valores de verificação do enunciado ($h(1,4,2,3)=39$, $h(1,4,3,2)=22$ e $h(1,\dots,12)=4366$), a igualdade exata entre avaliação incremental e completa em $9\,000$ movimentos e o respeito estrito ao orçamento $B$:
```bash
python3 scripts/tarefa_t1_validacao_h.py
python3 tests/test_problema.py
python3 tests/test_incremental.py
```

### 2. Reproduzir a Tarefa T2 (Vizinhanças e Speedup da Avaliação Incremental)
Gera `resultados/t2_vizinhancas.csv`, `resultados/t2_speedup.csv`, `relatorio/figuras/tabela_t2_vizinhancas.tex` (Tabela 2) e `relatorio/figuras/tabela_t2_speedup.tex` (Tabela 1):
```bash
python3 scripts/tarefa_t2_vizinhancas.py
```

### 3. Reproduzir o Experimento de Variantes do Greedy Descent e Ótimos Locais ($n=12$)
Gera `resultados/gd_variantes.csv` e `relatorio/figuras/tabela_gd_variantes.tex` (Tabela 3):
```bash
python3 scripts/tarefa_gd_variantes.py
```

### 4. Reproduzir o Protocolo Experimental Oficial (Tarefa T3 — 9 Combinações)
Executa `GD`, `SA` e `GA` nas instâncias pequena ($B=50\,000$, 30 runs), intermediária ($B=150\,000$, 30 runs) e grande ($B=300\,000$, 10 runs) com sementes controladas (`20260000 + run_id`), gravando em `resultados/resultados_t3.csv`, `resultados/parametros_sa.csv` e `resultados/diagnostico_sa.csv`:
```bash
python3 scripts/experimentos_t3.py
```
*(Observação: o script salva cada corrida imediatamente em disco e pula corridas já registradas em `resultados/resultados_t3.csv`. Para reexecutar do zero, remova `resultados/resultados_t3.csv` antes.)*

### 5. Reproduzir o Experimento Adicional do GA (Cruzamento OX Ligado vs. Desligado e Diversidade)
Gera `resultados/ga_experimento.csv`, `relatorio/figuras/tabela_ga_experimento.tex` (Tabela 6) e `relatorio/figuras/diversidade_ga.pdf`:
```bash
python3 scripts/tarefa_ga_experimento.py --runs 10
```

### 6. Gerar Todas as Figuras e Tabelas do Relatório + Testes de Mann–Whitney
Lê os CSVs em `resultados/` e gera em `relatorio/figuras/`:
- `convergencia.pdf` (Figura 2 — curvas de convergência com mediana e IQR + média da população do GA)
- `aceitacao_sa.pdf` (Figura 1 — taxa de aceitação de pioras do SA)
- `boxplots.pdf` (Figura 3 — distribuição do melhor $h$ final nas 3 instâncias)
- `tabela_parametros_sa.tex` (Tabela 4)
- `tabela_resultados_t3.tex` (Tabela 5)
- `tabela_mannwhitney.tex` e `resultados/testes_hipotese.csv` ($p$-valores do teste U de Mann–Whitney par a par)

```bash
python3 scripts/plot_graficos.py
```

### 7. Compilar o Relatório em PDF
Dentro de `relatorio/`:
```bash
cd relatorio && pdflatex relatorio.tex && pdflatex relatorio.tex
```

