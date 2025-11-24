import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity

# =============================
# 1. Carregar dataset
# =============================
df = pd.read_csv("dados_m/dataset_m_colunas_para_correlacoes.csv")
saida = pd.read_csv("dados_m/dataset_m_dengue_saneamento.csv")

# Selecionar somente variáveis numéricas (deve ser só isso no arquivo)
df_num = df.select_dtypes(include=['float64', 'int64'])

# =============================
# 2. Matriz de correlação
# =============================
corr_matrix = df_num.corr()

print("\n=== Matriz de Correlação ===")
print(corr_matrix)

print("\nDeterminante da matriz de correlação:", np.linalg.det(corr_matrix))

# =============================
# 3. Teste KMO (adequação amostral)
# =============================
kmo_all, kmo_model = calculate_kmo(df_num)

print("\n=== Teste KMO (Kaiser-Meyer-Olkin) ===")
print("KMO geral:", round(kmo_model, 4))
print("KMO por variável:", np.round(kmo_all, 4))

# =============================
# 4. Teste de Esfericidade de Bartlett
# =============================
chi_square, p_value = calculate_bartlett_sphericity(df_num)

print("\n=== Teste de Bartlett ===")
print(f"Chi²: {chi_square:.4f}")
print("p-valor:", p_value)

# =============================
# 6. Análise Fatorial (todas variáveis para estudo)
# =============================
fa_analyzer = FactorAnalyzer(rotation=None)
fa_analyzer.fit(df_num)

# Autovalores
ev, v = fa_analyzer.get_eigenvalues()

print("\n=== Autovalores (Eigenvalues) ===")
for i, val in enumerate(ev):
    print(f"Componente {i+1}: {val:.4f}")

# =============================
# 7. Extração do fator (1 fator para ranking)
# =============================
fa = FactorAnalysis(n_components=1, random_state=42)
fator = fa.fit_transform(df_num)

# Cargas fatoriais
cargas = fa.components_.T

print("\n=== Cargas Fatoriais (Factor Loadings) ===")
for col, carga in zip(df_num.columns, cargas):
    print(f"{col}: {carga[0]:.4f}")

# =============================
# 8. Comunalidades (variância explicada por cada variável)
# =============================
comunalidades = cargas**2

print("\n=== Comunalidades ===")
for col, com in zip(df_num.columns, comunalidades):
    print(f"{col}: {com[0]:.4f}")

# =============================
# 9. Variância explicada pelo fator
# =============================
variancia_explicada = np.sum(comunalidades)

print("\n=== Variância Total Explicada pelo Factor 1 ===")
print(float(variancia_explicada))

# =============================
# 10. Adicionar o fator ao dataset
# =============================
df['Fator_Ranking'] = fator[:, 0]
saida['Fator_Ranking'] = df['Fator_Ranking']

# =============================
# 11. Salvar resultado final
# =============================
saida.to_csv("dados_m/dataset_m_dengue_saneamento.csv", index=False)

# =============================
# 12. Classificação em quartis do Fator_Ranking
# =============================

# Cálculo dos quartis
q1 = df["Fator_Ranking"].quantile(0.25)
q2 = df["Fator_Ranking"].quantile(0.50)
q3 = df["Fator_Ranking"].quantile(0.75)

def classificar_fator(valor):
    if valor <= q1:
        return "Q4 - Muito Alto"
    elif valor <= q2:
        return "Q3 - Alto"
    elif valor <= q3:
        return "Q2 - Moderado"
    else:
        return "Q1 - Baixo"

# Criar nova coluna
df["Classificação_Fatorial"] = df["Fator_Ranking"].apply(classificar_fator)
saida['Classificação_ranking'] = df['Classificação_Fatorial']
saida.to_csv("dados_m/dataset_m_dengue_saneamento.csv", index=False)

# =============================
# 4. Gráfico: Distribuição dos Quartis por Cluster
# =============================

df = pd.read_csv("dados_m/dataset_m_dengue_saneamento.csv")

# Garantir ordenação dos clusters
df["Cluster"] = df["Cluster"].astype(int)

# =============================
# 2. Tabela de distribuição
# =============================
tabela = pd.crosstab(df["Cluster"], df["Classificação_ranking"])

# Garantir ordem dos quartis exibidos
ordem_quartis = ["Q1 - Baixo", "Q2 - Moderado", "Q3 - Alto", "Q4 - Muito Alto"]
tabela = tabela.reindex(columns=ordem_quartis)

# =============================
# 3. Gráfico no estilo solicitado
# =============================
plt.figure(figsize=(14, 7))

x = np.arange(len(tabela.index))  # posição dos clusters no eixo X
largura = 0.2                     # largura de cada barra

# Plotar barras lado a lado
for i, quartil in enumerate(ordem_quartis):
    plt.bar(x + i*largura, tabela[quartil], width=largura, label=quartil)

# Configurações visuais
plt.title("Distribuição de Classificações de Ranking por Cluster", fontsize=14)
plt.xlabel("Cluster", fontsize=12)
plt.ylabel("Número de Municípios", fontsize=12)

plt.xticks(x + largura*1.5, tabela.index)  # centralizar grupos

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(title="Classificação")

plt.tight_layout()
plt.savefig("imagens/img_distribuicao_saneamento_por_cluster_padrao.png", dpi=300)
plt.close()