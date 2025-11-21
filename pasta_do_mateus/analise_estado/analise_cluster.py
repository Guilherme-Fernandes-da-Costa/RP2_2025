import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

os.makedirs("resultados", exist_ok=True)

POR_X_HABITANTES = 100000
NUM_CLUSTER = 10

def processar_dataset(caminho, titulo):

    # Leitura do dataset
    df = pd.read_csv(caminho)

    #--------------------------------

    # Ignorando as duas primeiras colunas
    df_dados = df.iloc[:, 7:]

    # Escolha do número de clusters (você pode ajustar)
    k = NUM_CLUSTER

    # Aplicando o K-Means
    kmeans = KMeans(n_clusters=k, random_state=42)
    df["Cluster"] = kmeans.fit_predict(df_dados)


    print("\nDistribuição de municípios por cluster:")
    print(df["Cluster"].value_counts().sort_index())

    # Salvando o dataset com os clusters
    df.to_csv(caminho, index=False)

    #--------------------------------

    # Cálculo da taxa de incidência por 100.000 habitantes
    df["Taxa_de_incidência"] = ((df["Total_confirmados_anual"] / df["População_total_2021"]) * POR_X_HABITANTES).round()

    # Salvando o novo dataset com a nova coluna
    df.to_csv(caminho, index=False)

    #--------------------------------

    q1 = df["Taxa_de_incidência"].quantile(0.25)
    q2 = df["Taxa_de_incidência"].quantile(0.50)
    q3 = df["Taxa_de_incidência"].quantile(0.75)

    # Função de classificação por quartis
    def classificar_incidencia(valor):
        if valor <= q1:
            return "Q1 - Baixa"
        elif valor <= q2:
            return "Q2 - Moderada"
        elif valor <= q3:
            return "Q3 - Alta"
        else:
            return "Q4 - Muito Alta"

    # Aplicar a função à coluna de taxa
    df["Classificação_incidência"] = df["Taxa_de_incidência"].apply(classificar_incidencia)

    # Salvar o novo dataset com a coluna de classificação
    df.to_csv(caminho, index=False)

    #--------------------------------

    # Criar a tabela cruzada com totais
    tabela = pd.crosstab(
        df["Cluster"],
        df["Classificação_incidência"],
        margins=True,
        margins_name="Total"
    )

    # Remover a linha "Total"
    tabela = tabela[tabela.index != "Total"]

    # Reordenar as colunas na ordem desejada
    ordem_colunas = ["Q1 - Baixa", "Q2 - Moderada", "Q3 - Alta", "Q4 - Muito Alta", "Total"]
    tabela = tabela[ordem_colunas]

    # Ordenar as linhas pelo total (coluna "Total")
    tabela_ordenada = tabela.sort_values(by="Total", ascending=True)

    # Exibir o resultado
    print("\nDistribuição de classificações de incidência por cluster (ordenada e com colunas organizadas):")
    print(tabela_ordenada)

    #--------------------------------

    # Visualização dos clusters com PCA
    from sklearn.decomposition import PCA

    # Reduzindo para 2 dimensões para visualização
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_dados)

    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]

    # Plotando os clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        df['PCA1'],
        df['PCA2'],
        c=df['Cluster'],
        cmap='tab20',
        s=35,
        alpha=0.8
    )

    plt.title(f"Visualização dos Clusters (PCA 2D) - {titulo}", fontsize=14)
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.grid(True)

    # Adicionando legenda de clusters
    plt.legend(*scatter.legend_elements(), title="Cluster")

    # Salvando imagem
    plt.savefig(f"resultados/img_clusters_pca_visualizacao_{titulo}.png", dpi=300)
    plt.close()

    #--------------------------------

    # Usando a mesma tabela ordenada (sem linha "Total")
    tabela_plot = tabela_ordenada[["Q1 - Baixa", "Q2 - Moderada", "Q3 - Alta", "Q4 - Muito Alta"]]

    clusters = tabela_plot.index.tolist()
    categorias = tabela_plot.columns.tolist()

    # Valores numéricos
    valores = tabela_plot.values

    # Número de clusters e categorias
    n_clusters = len(clusters)
    n_categorias = len(categorias)

    # Posições no eixo X
    x = np.arange(n_clusters) * 1.25

    # Largura das barras
    largura = 0.25

    plt.figure(figsize=(14, 7))

    # Criando uma barra para cada classificação
    for i in range(n_categorias):
        plt.bar(
            x + i * largura,
            valores[:, i],
            width=largura,
            label=categorias[i]
        )

    # Configurações do gráfico
    plt.xlabel("Cluster")
    plt.ylabel("Número de Municípios")
    plt.title(f"Distribuição de Classificações de Incidência por Cluster - {titulo}")
    plt.xticks(x + largura, clusters)
    plt.legend(title="Classificação")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # Salvar em arquivo
    plt.savefig(f"resultados/img_barras_incidencia_por_cluster_{titulo}.png", dpi=300)
    plt.close()

    #--------------------------------

    # Garantir que tudo seja numérico
    df_dados = df_dados.apply(pd.to_numeric, errors="coerce")
    df_dados = df_dados.fillna(df_dados.mean())

    # 3. Clusterização hierárquica (bottom-up)
    Z = linkage(df_dados, method='ward')  
    # métodos possíveis: ward, complete, single, average

    # 4. Plotar dendrograma
    plt.figure(figsize=(14, 7))
    dendrogram(Z, labels=df.index, leaf_rotation=90)
    plt.title("Dendrograma - Clusterização Hierárquica (Bottom-Up)")
    plt.xlabel("Municípios")
    plt.ylabel("Distância")
    plt.tight_layout()
    plt.savefig("resultados/img_dendrograma_cluster_hierarquico.png", dpi=300)
    plt.close()

    df["Cluster_Hierarquico"] = fcluster(Z, k, criterion='maxclust')

    # 6. Exibir distribuição dos clusters
    print("\nDistribuição dos clusters hierárquicos:")
    print(df["Cluster_Hierarquico"].value_counts().sort_index())

    # 7. Salvar dataset com os clusters
    df.to_csv("resultados/dataset_m_cluster_hierarquico.csv", index=False)

processar_dataset("dataset_m_dengue_saneamento.csv", "padrao")
processar_dataset("dataset_m_normalizado.csv", "normalizado")