import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Caminho do dataset
#caminho = "dataset_dengue_saneamento.csv"; titulo = "padrao"
caminho = "dataset_normalizado.csv"; titulo = "normalizado"

# Ler dataset
df = pd.read_csv(caminho)

# Nome da variável alvo
alvo = "Taxa_de_incidência"

# Garantir que a coluna alvo é numérica
df[alvo] = pd.to_numeric(df[alvo], errors="coerce")

# Selecionar variáveis que começam com V
variaveis_v = [col for col in df.columns if col.startswith("V")]

# Calcular correlação com Taxa de Incidência
resultados = {}

for col in variaveis_v:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    corr = df[[alvo, col]].corr().iloc[0, 1]
    resultados[col] = corr

# Transformar em DataFrame
corr_df = pd.DataFrame.from_dict(resultados, orient="index", columns=["Correlação"])
corr_df = corr_df.sort_values(by="Correlação", ascending=False)

print("\nCorrelação linear entre Taxa de Incidência e variáveis Vxxxx:")
print(corr_df)

# Salvar CSV
corr_df.to_csv("dataset_correlacao.csv")

# --------------------------------------------------------
# OPÇÃO 1 — GRÁFICO DE BARRAS
# --------------------------------------------------------
plt.figure(figsize=(10, 8))
corr_df["Correlação"].plot(kind="bar", color="steelblue")
plt.title("Correlação entre Variáveis Vxxxx e Taxa de Incidência")
plt.xlabel("Variáveis")
plt.ylabel("Correlação de Pearson")
plt.xticks(rotation=90)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("img_grafico_barras_correlacao.png", dpi=300)
plt.close()

# --------------------------------------------------------
# OPÇÃO 2 — HEATMAP DAS CORRELAÇÕES
# --------------------------------------------------------
plt.figure(figsize=(6, 10))
sns.heatmap(corr_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1, linewidths=.5)
plt.title("Heatmap da Correlação com Taxa de Incidência")
plt.tight_layout()
plt.savefig("img_heatmap_correlacao.png", dpi=300)
plt.close()

#---------------------------------------------------------


