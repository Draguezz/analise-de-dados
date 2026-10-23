"""
analisar_tabela.py
------------------
Lê uma tabela (CSV ou Excel), faz uma análise exploratória automática
e gera um painel com vários gráficos complexos em uma única imagem.

Uso:
    python analisar_tabela.py minha_tabela.csv
    python analisar_tabela.py minha_tabela.xlsx
    python analisar_tabela.py            # sem argumento: usa dados de exemplo

Dependências:
    pip install pandas numpy matplotlib seaborn openpyxl
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")


# ---------------------------------------------------------------------------
# 1. Carregar dados
# ---------------------------------------------------------------------------
def carregar_tabela(caminho=None):
    """Carrega CSV/Excel. Sem caminho, gera uma tabela de exemplo."""
    if caminho is None:
        print("Nenhum arquivo informado: usando dados de exemplo.\n")
        return gerar_exemplo()

    if caminho.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(caminho)

    # CSV: tenta detectar o separador (vírgula ou ponto e vírgula)
    try:
        return pd.read_csv(caminho, sep=None, engine="python", encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(caminho, sep=None, engine="python", encoding="latin-1")


def gerar_exemplo(n=500, seed=42):
    rng = np.random.default_rng(seed)
    datas = pd.date_range("2024-01-01", periods=n, freq="D")
    regiao = rng.choice(["Norte", "Nordeste", "Sul", "Sudeste"], n)
    produto = rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2])
    preco = rng.normal(100, 20, n).round(2)
    qtd = rng.poisson(20, n)
    desconto = rng.uniform(0, 0.3, n).round(3)
    receita = (preco * qtd * (1 - desconto) + rng.normal(0, 100, n)).round(2)
    return pd.DataFrame({
        "data": datas, "regiao": regiao, "produto": produto,
        "preco": preco, "quantidade": qtd, "desconto": desconto,
        "receita": receita,
    })


# ---------------------------------------------------------------------------
# 2. Limpeza e detecção de tipos
# ---------------------------------------------------------------------------
def preparar(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Tenta converter colunas de texto que parecem datas
    for col in df.select_dtypes(include="object"):
        amostra = df[col].dropna().astype(str).head(20)
        if len(amostra) and amostra.str.contains(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}").mean() > 0.8:
            convertida = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if convertida.notna().mean() > 0.8:
                df[col] = convertida

    numericas = df.select_dtypes(include="number").columns.tolist()
    datas = df.select_dtypes(include="datetime").columns.tolist()
    categoricas = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if df[c].nunique() <= 15
    ]
    return df, numericas, categoricas, datas


# ---------------------------------------------------------------------------
# 3. Análise em texto
# ---------------------------------------------------------------------------
def resumo_texto(df, numericas, categoricas):
    print("=" * 70)
    print(f"Linhas: {len(df)} | Colunas: {df.shape[1]}")
    print("=" * 70)

    nulos = df.isna().sum()
    nulos = nulos[nulos > 0]
    print("\nValores ausentes:" if len(nulos) else "\nSem valores ausentes.")
    if len(nulos):
        print(nulos.to_string())

    if numericas:
        print("\nEstatísticas das colunas numéricas:")
        print(df[numericas].describe().round(2).to_string())

        # Outliers pelo método IQR
        print("\nOutliers (método IQR):")
        for c in numericas:
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            n_out = ((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum()
            print(f"  {c}: {n_out}")

        if len(numericas) >= 2:
            corr = df[numericas].corr()
            pares = (
                corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
                .stack().sort_values(key=abs, ascending=False)
            )
            print("\nMaiores correlações:")
            print(pares.head(5).round(3).to_string())

    for c in categoricas:
        print(f"\nContagem de '{c}':")
        print(df[c].value_counts().to_string())


# ---------------------------------------------------------------------------
# 4. Painel de gráficos
# ---------------------------------------------------------------------------
def painel(df, numericas, categoricas, datas, saida="painel_analise.png"):
    fig = plt.figure(figsize=(20, 14), constrained_layout=True)
    gs = gridspec.GridSpec(3, 3, figure=fig)
    fig.suptitle("Painel de Análise da Tabela", fontsize=22, fontweight="bold")

    # (0,0) Mapa de calor de correlação
    ax = fig.add_subplot(gs[0, 0])
    if len(numericas) >= 2:
        corr = df[numericas].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, square=True, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Correlação entre variáveis")

    # (0,1) Distribuição com KDE da 1ª coluna numérica
    ax = fig.add_subplot(gs[0, 1])
    if numericas:
        col = numericas[0]
        sns.histplot(df[col].dropna(), kde=True, bins=30, ax=ax, color="steelblue")
        ax.axvline(df[col].mean(), color="red", ls="--", label=f"Média: {df[col].mean():.1f}")
        ax.axvline(df[col].median(), color="green", ls="--", label=f"Mediana: {df[col].median():.1f}")
        ax.legend()
        ax.set_title(f"Distribuição de {col}")

    # (0,2) Boxplot das numéricas normalizadas (comparar escalas diferentes)
    ax = fig.add_subplot(gs[0, 2])
    if numericas:
        norm = (df[numericas] - df[numericas].mean()) / df[numericas].std()
        sns.boxplot(data=norm, orient="h", ax=ax)
        ax.set_title("Boxplots (valores padronizados)")

    # (1,0) Dispersão com regressão e cor por categoria
    ax = fig.add_subplot(gs[1, 0])
    if len(numericas) >= 2:
        x, y = numericas[0], numericas[-1]
        hue = categoricas[0] if categoricas else None
        sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.6, ax=ax)
        sns.regplot(data=df, x=x, y=y, scatter=False, color="black",
                    line_kws={"lw": 2}, ax=ax)
        ax.set_title(f"{y} vs {x} (com tendência)")

    # (1,1) Série temporal com média móvel
    ax = fig.add_subplot(gs[1, 1:])
    if datas and numericas:
        d, v = datas[0], numericas[-1]
        serie = df.set_index(d)[v].sort_index().resample("W").sum()
        ax.plot(serie.index, serie.values, alpha=0.5, label="Soma semanal")
        ax.plot(serie.index, serie.rolling(4).mean(), lw=2.5, color="crimson",
                label="Média móvel (4 semanas)")
        ax.fill_between(serie.index, serie.values, alpha=0.15)
        ax.set_title(f"Evolução de {v} ao longo do tempo")
        ax.legend()
    elif numericas:
        df[numericas[-1]].reset_index(drop=True).plot(ax=ax)
        ax.set_title(f"{numericas[-1]} por índice")

    # (2,0) Barras agrupadas: média por categoria
    ax = fig.add_subplot(gs[2, 0])
    if categoricas and numericas:
        cat, v = categoricas[0], numericas[-1]
        ordem = df.groupby(cat)[v].mean().sort_values(ascending=False).index
        sns.barplot(data=df, x=cat, y=v, order=ordem, errorbar="sd",
                    capsize=0.1, ax=ax)
        ax.set_title(f"Média de {v} por {cat} (± desvio padrão)")

    # (2,1) Violin por segunda categoria (ou a primeira)
    ax = fig.add_subplot(gs[2, 1])
    if categoricas and numericas:
        cat = categoricas[1] if len(categoricas) > 1 else categoricas[0]
        sns.violinplot(data=df, x=cat, y=numericas[-1], inner="quartile", ax=ax)
        ax.set_title(f"Distribuição de {numericas[-1]} por {cat}")

    # (2,2) Heatmap cruzando duas categorias com média da numérica
    ax = fig.add_subplot(gs[2, 2])
    if len(categoricas) >= 2 and numericas:
        pivo = df.pivot_table(index=categoricas[0], columns=categoricas[1],
                              values=numericas[-1], aggfunc="mean")
        sns.heatmap(pivo, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
        ax.set_title(f"Média de {numericas[-1]}: {categoricas[0]} × {categoricas[1]}")
    elif categoricas:
        df[categoricas[0]].value_counts().plot.pie(autopct="%1.0f%%", ax=ax)
        ax.set_ylabel("")
        ax.set_title(f"Proporção de {categoricas[0]}")
    else:
        ax.axis("off")

    fig.savefig(saida, dpi=150, bbox_inches="tight")
    print(f"\nPainel salvo em: {saida}")
    plt.show()


# ---------------------------------------------------------------------------
# 5. Execução
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    caminho = sys.argv[1] if len(sys.argv) > 1 else None
    df = carregar_tabela(caminho)
    df, numericas, categoricas, datas = preparar(df)
    resumo_texto(df, numericas, categoricas)
    painel(df, numericas, categoricas, datas)
