# Importação de bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Leitura do arquivo
df = pd.read_excel('dengue2022_mes.xlsx', sheet_name='dengue2022_mes', header=[3, 4, 5], skiprows=0)

# Visualizar as primeiras linhas e shape
print(df.head())
print(df.shape)



# Remover linhas com todos NaNs e últimas linhas (totais e notas)
df = df.dropna(how='all')
df = df[:-3]  # Ajuste se necessário com base na análise dos dados

# Preencher NaNs em colunas numéricas com 0
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

# Flatten multi-header
df.columns = ['_'.join(col).strip().replace('Unnamed:', '').replace('_level_2', '') for col in df.columns.values]
df.columns = df.columns.str.replace(r'\s+', '_', regex=True)

# Converter tipos para inteiro
case_cols = [col for col in df.columns if 'notificados' in col or 'confirmados' in col]
df[case_cols] = df[case_cols].astype(int)

# Remover colunas desnecessárias
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

print(df.head())

# Calcular totais confirmados por mês
months = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']

for month in months:
    df[f'{month}_confirmados_total'] = df[f'{month}_confirmados_autóctones'] + df[f'{month}_confirmados_importados']

# Total anual por município
df['Total_confirmados_anual'] = df[[f'{m}_confirmados_total' for m in months]].sum(axis=1)

# Filtrar municípios com >100 casos totais
df_filtered = df[df['Total_confirmados_anual'] > 100]

# Transformar em formato long
id_vars = ['DRS__0_level_1__0', 'DRS__1_level_1__1', 'GVE__2_level_1__2', 'GVE__3_level_1__3', 'Região_de_Saúde__4_level_1__4', 'Região_de_Saúde__5_level_1__5', 'município__6_level_1__6']

value_vars = [col for col in df.columns if 'confirmados_total' in col and 'anual' not in col]
df_long = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name='Mês', value_name='Casos_Confirmados')
df_long['Mês'] = df_long['Mês'].str.replace('_confirmados_total', '')

# Estatísticas descritivas por mês (ex.: Março)
print(df['Março_confirmados_total'].describe())

# Histograma de casos totais anuais
plt.figure(figsize=(10,6))
df['Total_confirmados_anual'].hist(bins=20)
plt.title('Distribuição de Casos Confirmados de Dengue por Município (2022)')
plt.xlabel('Casos Totais')
plt.ylabel('Frequência')
plt.show()

# Exportar para CSV
df.to_csv('dataset_dengue.csv', index=False)
print('Arquivo salvo como dataset_dengue.csv')