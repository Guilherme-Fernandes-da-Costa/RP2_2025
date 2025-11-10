import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Caminho do arquivo original
caminho = "dataset_dengue_saneamento.csv"

#--------------------------------

# Leitura do dataset
df = pd.read_csv(caminho)

# Cálculo da taxa de incidência por 100.000 habitantes
df["Taxa_incidência_mil"] = ((df["Total_confirmados_anual"] / df["População_total_2021"]) * 10000).round()

# Salvando o novo dataset com a nova coluna
df.to_csv("dataset_dengue_saneamento.csv", index=False)

#--------------------------------

# Função para classificar a taxa de incidência
def classificar_incidencia(valor):
    if valor < 100:
        return "Baixa"
    elif valor <= 300:
        return "Média"
    else:
        return "Alta"

# Aplicar a função à coluna de taxa
df["Classificação_incidência"] = df["Taxa_incidência_mil"].apply(classificar_incidencia)

# Salvar o novo dataset com a coluna de classificação
df.to_csv("dataset_dengue_saneamento.csv", index=False)

#--------------------------------

# Ignorando as duas primeiras colunas
df_dados = df.iloc[:, 7:]

# Escolha do número de clusters (você pode ajustar)
k = 12

# Aplicando o K-Means
kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(df_dados)


print("\nDistribuição de municípios por cluster:")
print(df["Cluster"].value_counts().sort_index())

# Salvando o dataset com os clusters
df.to_csv("dataset_dengue_saneamento.csv", index=False)

#--------------------------------

# Criar a tabela cruzada com totais
tabela = pd.crosstab(
    df["Cluster"],
    df["Classificação_incidência"],
    margins=True,
    margins_name="Total"
)

# Ordenar as linhas (exceto "Total") pelo total em ordem crescente
tabela_ordenada = (
    tabela[tabela.index != "Total"]       
    .sort_values(by="Total", ascending=True)
)

# Exibir o resultado
print("\nDistribuição de classificações de incidência por cluster (ordenada por total crescente):")
print(tabela_ordenada)

#--------------------------------

