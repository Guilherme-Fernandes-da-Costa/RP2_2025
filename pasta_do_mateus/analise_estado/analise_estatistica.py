import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Caminho do arquivo original
caminho = "dataset_dengue_incidencia.csv"

# Leitura do dataset
df = pd.read_csv(caminho)

# Cálculo da taxa de incidência por 100.000 habitantes
df["Taxa_incidência_mil"] = ((df["Total_confirmados_anual"] / df["População_total_2021"]) * 10000).round()

# Salvando o novo dataset com a nova coluna
df.to_csv("dataset_dengue_incidencia.csv", index=False)

#--------------------------------

caminho = "dataset_dengue_incidencia.csv"

# Leitura do dataset
df = pd.read_csv(caminho)

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
df.to_csv("dataset_dengue_incidencia.csv", index=False)

#--------------------------------
# Caminho do arquivo
caminho1 = "dataset_saneamento_estado.csv"
caminho2 = "normalizado_saneamento.csv"

# Leitura do dataset
df1 = pd.read_csv(caminho1)
df2 = pd.read_csv(caminho2)

# Ignorando as duas primeiras colunas
df1_dados = df1.iloc[:, 2:]
df2_dados = df2.iloc[:, 2:]

# Escolha do número de clusters (você pode ajustar)
k = 12

# Aplicando o K-Means
kmeans = KMeans(n_clusters=k, random_state=42)
df1["Cluster"] = kmeans.fit_predict(df1_dados)
df2["Cluster"] = kmeans.fit_predict(df2_dados)


print("\nDistribuição de municípios por cluster:")
print(df1["Cluster"].value_counts())

print("\nDistribuição de municípios por cluster:")
print(df2["Cluster"].value_counts())

# Salvando o dataset com os clusters
df1.to_csv("dataset_saneamento_estado.csv", index=False)
df2.to_csv("normalizado_saneamento.csv", index=False)
