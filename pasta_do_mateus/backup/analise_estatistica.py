import pandas as pd
import os
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.graphics.gofplots import qqplot

TOP_N = 10

os.makedirs("imagens", exist_ok=True)
os.makedirs("resultados", exist_ok=True)

# Carregar datasets
saneamento = pd.read_csv("dados_n/dataset_n_saneamento_estado.csv")
dengue = pd.read_csv("dados_n/dataset_n_dengue.csv")
populacao = pd.read_csv("dados_n/dataset_n_sp_mun_população.csv")

# Fazer o merge (união) dos dados
dados = pd.merge(dengue, saneamento, on="CD_MUN", how="inner")
dados.head()

# Merge (união) com o conjunto de dados que possui a população de cada município
dados_pop = pd.merge(dados, populacao, on="CD_MUN", how="inner")
dados_pop.head()

# Remover colunas constantes
dados_pop = dados_pop.loc[:, dados_pop.nunique() > 1]

# Remover colunas constantes
dados_pop = dados_pop.loc[:, dados_pop.nunique() > 1]

# Criação do coeficiente de incidência de dengue (dividindo o total de casos autóctones de dengue no município
# por sua população)
dados_pop['Total_autoctone_ponderado'] = dados_pop['Total_confirmados_autóctones'] / dados_pop['População']
alvo = "Total_autoctone_ponderado"

# Eliminação de algumas colunas redundantes ou não relevantes
dados
dados_pop.drop(['Total_setores'], axis=1)

# Variáveis de saneamento - Domicílios com determinadas características com relação a água, lixo e esgoto
cols_saneamento = [col for col in dados_pop.columns if col.startswith('V00')]

# Criar todas as novas colunas de uma vez (evita fragmentação)
novas_cols = {
    f"{col}_idx": dados_pop[col] / dados_pop["V00004"]
    for col in cols_saneamento
}

# Converter o dicionário em DataFrame
novas_df = pd.DataFrame(novas_cols)

# Concatenar ao dataset original
dados_pop = pd.concat([dados_pop, novas_df], axis=1)

# (Opcional, mas recomendado) remover fragmentação
dados_pop = dados_pop.copy()

# Selecionar colunas indexadas para análise
cols_saneamento_idx = list(novas_cols.keys())
cols_s = cols_saneamento + cols_saneamento_idx

# Salvar arquivo dentro da pasta
dados_pop.to_csv("resultados/dataset_n_dados_unidos.csv", index=False)

# Selecionar apenas as colunas numéricas do conjunto de dados
dados_num = dados_pop.select_dtypes(include=['number'])

# Criar um StandardScaler para as variáveis independentes
scaler_X = StandardScaler()

# Normalizar as variáveis de saneamento (índices)
dados_num[cols_saneamento_idx] = scaler_X.fit_transform(dados_num[cols_saneamento_idx])

# Salvar dataset apenas com as variáveis saneamento_idx normalizadas
dados_num[cols_saneamento_idx].to_csv(
    "resultados/dataset_n_normalizado_saneamento_idx.csv",
    index=False
)

# Normalizar as variáveis de saneamento (todas)
dados_num[cols_s] = scaler_X.fit_transform(dados_num[cols_s])

# Salvar dataset com todas as variáveis saneamento normalizadas
dados_num[cols_s].to_csv(
    "resultados/dataset_n_normalizado_saneamento_completo.csv",
    index=False
)


# 1. Correlação=================================================

# Calcular a matriz de correlação
correlacao = dados_num[cols_s + ['Total_autoctone_ponderado']].corr()
correlacao_dengue = correlacao['Total_autoctone_ponderado'].dropna().sort_values(ascending=False)

# Salvar em CSV
correlacao_dengue.to_csv("resultados/dataset_n_correlacao.csv", header=["Correlacao"])

# Visualizar as correlações mais fortes (excluindo a própria variável alvo)
plt.figure(figsize=(10, 6))
correlacao_dengue.drop('Total_autoctone_ponderado').plot(kind='bar')
plt.title('Correlação das Variáveis de Saneamento com Incidência de Dengue')
plt.ylabel('Correlação de Pearson')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("imagens/img_correlacao_saneamento_dengue.png", dpi=300)
plt.close()


# 2. Regressão Simples=================================================

resultados_regressao = []   # lista para armazenar resultados

for var in cols_s:
    # Preparar dados para regressão simples
    X_simples = dados_num[[var]]
    y = dados_num['Total_autoctone_ponderado']

    # Ajustar modelo de regressão simples
    modelo_simples = LinearRegression()
    modelo_simples.fit(X_simples, y)
    y_pred_simples = modelo_simples.predict(X_simples)

    # Calcular R²
    r2_simples = r2_score(y, y_pred_simples)

    # Adicionar à lista para o CSV
    resultados_regressao.append({
        "Variavel": var,
        "R2": r2_simples
    })

# Converter para DataFrame e salvar em CSV
df_reg = pd.DataFrame(resultados_regressao)
df_reg.to_csv("resultados/dataset_n_regressao_simples.csv", index=False)

# Verificar se a coluna-alvo está presente
if alvo not in dados_num.columns:
    raise ValueError(f"A coluna alvo '{alvo}' não foi encontrada nas colunas numéricas.")


# 3. Regressão Múltipla=================================================

X_multi = dados_pop[cols_saneamento_idx]  # Usar todas as variáveis de saneamento
modelo_multi = LinearRegression()
modelo_multi.fit(X_multi, y)
y_pred_multi = modelo_multi.predict(X_multi)

# Calcular R²
r2_multi = r2_score(y, y_pred_multi)
mae = mean_absolute_error(y, y_pred_multi)
rmse = np.sqrt(mean_squared_error(y, y_pred_multi))

print("\nResultados da Regressão Múltipla:")
print(f"R²: {r2_multi:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Intercepto: {modelo_multi.intercept_:.4f}")

# Verificar overfitting comparando R²
if r2_multi >= 0.99:
    print("\nAviso: R² muito alto (próximo de 1.0). Possível overfitting. Considere reduzir o número de variáveis ou coletar mais dados.")
    
# Salvar o conjunto de dados processado
dados_pop.to_csv('resultados/dataset_n_dados_unidos_estado.csv', index=False)  


# 3. imagens resultantes=================================================

print(f"\nTop {TOP_N} variáveis mais correlacionadas positivamente:")
print(correlacao_dengue.head(TOP_N))

print(f"\nTop {TOP_N} variáveis mais correlacionadas negativamente:")
print(correlacao_dengue.tail(TOP_N))

# Selecionar as TOP_N variáveis mais correlacionadas positivamente e negativamente
top_positivas = correlacao_dengue.head(TOP_N).index.tolist()
top_negativas = correlacao_dengue.tail(TOP_N).index.tolist()

# Unir listas (garantindo que não repitam)
variaveis_selecionadas = list(dict.fromkeys(top_positivas + top_negativas))

# Criar novo dataframe apenas com essas colunas
df_colunas_correlacao = dados_num[variaveis_selecionadas].copy()

# Salvar em CSV
df_colunas_correlacao.to_csv("dados_m/dataset_m_colunas_para_correlacoes.csv", index=False)

# === Gráfico de barras para as mais e menos correlacionadas ===
corr_a = correlacao_dengue.drop(alvo)
top_pos = corr_a.head(TOP_N)
top_neg = corr_a.tail(TOP_N)

# Concatenar para exibir juntas
corr_top = pd.concat([top_pos, top_neg])

plt.figure(figsize=(10, 6))
sns.barplot(
    x=corr_top.values,
    y=corr_top.index,
    hue=corr_top.index,       
    palette="coolwarm",
    dodge=False,              
    legend=False              
)
plt.title(f"Top {TOP_N} correlações positivas e negativas com a incidência de dengue")
plt.xlabel("Correlação de Pearson")
plt.ylabel("Variável")
plt.tight_layout()
plt.savefig("imagens/img_top_10_correlacoes.png", dpi=300)
plt.close()

# === Gráfico Fits Plot (observado vs predito) ===
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y, y=y_pred_multi, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # Linha 45° ajustada
plt.xlabel("Incidência de Dengue (real)")
plt.ylabel("Incidência de Dengue (predita)")
plt.title("Fits Plot - Observado vs Predito (Regressão Múltipla)")
plt.savefig("imagens/img_regressao_multipla.png", dpi=300)
plt.close()

# === Q-Q Plot dos resíduos ===
residuos = y - y_pred_multi

sm.qqplot(residuos, line='45', fit=True)
plt.title("Q-Q Plot dos Resíduos")
plt.tight_layout()
plt.savefig("imagens/img_Q-Q_plot.png", dpi=300)
plt.close()

# === Histograma dos resíduos ===
plt.figure(figsize=(6, 4))
sns.histplot(residuos, kde=True)
plt.title("Distribuição dos Resíduos")
plt.xlabel("Resíduo (erro)")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig("imagens/img_histograma_de_residuos.png", dpi=300)
plt.close()
