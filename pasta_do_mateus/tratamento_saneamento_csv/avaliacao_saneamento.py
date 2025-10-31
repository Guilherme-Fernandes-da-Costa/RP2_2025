# Importar a biblioteca pandas
import pandas as pd

# Carregar os três datasets
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')
df3 = pd.read_csv('dataset3.csv')

# Verificar as primeiras linhas para garantir que a coluna ID está correta
print("Dataset 1:")
print(df1.head())
print("\nDataset 2:")
print(df2.head())
print("\nDataset 3:")
print(df3.head())

# Supondo que a primeira coluna se chama 'ID' em todos os datasets
# Unir df1 e df2 com base na coluna 'ID'
merged_df = pd.merge(df1, df2, on='setor', how='inner')

# Unir o resultado com df3
final_df = pd.merge(merged_df, df3, on='setor', how='inner')

# Verificar o dataset final
print("\nDataset final:")
print(final_df.head())
print("\nInformações do dataset final:")
print(final_df.info())

# (Opcional) Salvar o dataset combinado
final_df.to_csv('dataset_combinado.csv', index=False)
# ================================
# Eliminação de algumas variáveis: as que tratam do gênero dos ocupantes do domicílio, algo que não será investigado
# Definindo os intervalos de início e fim
intervalos = [
   (11, 17),
   (72, 74),
   (105, 111),
   (183, 199),
   (226, 232),
   (295, 309),
   (381, 397),
   (451, 463),
   (481, 485),
   (502, 508),
   (524, 540),
   (546, 552),
   (566, 580),
   (596, 612),
   (624, 636),
   (640, 644),
]

# Gerar todos os nomes de colunas
cols_to_drop = [ f'V{i:05d}' for (start, end) in intervalos for i in range(start, end) ]

# Dropar as colunas
final_df = final_df.drop(columns=cols_to_drop)

# (Opcional) Salvar o dataset processado e verificar novamente as informações do dataset
final_df.to_csv('dataset_combinado.csv', index=False)

print("\nInformações do dataset final:")
print(final_df.info())

#CODE FOR FILTERS