import pandas as pd
import os

# Caminhos dos arquivos
dictionary_file = 'dicionario_filtrado.csv'
saneamento_file = 'dataset_saneamento_estado.csv'

# Passo 1: Ler o dicionário filtrado
df_dictionary = pd.read_csv(dictionary_file)

# Criar um mapeamento de Code para multiplicador (1 para 'positivo', -1 para 'negativo')
multiplier_map = {}
for _, row in df_dictionary.iterrows():
    code = row['Code']
    influence = row['influence'].strip().lower()
    if influence == 'positivo':
        multiplier_map[code] = 1
    elif influence == 'negativo':
        multiplier_map[code] = -1

# Passo 2: Ler o dataset de saneamento
df_saneamento = pd.read_csv(saneamento_file)

# Passo 3: Aplicar multiplicadores às colunas correspondentes
for code, multiplier in multiplier_map.items():
    if code in df_saneamento.columns:
        df_saneamento[code] = pd.to_numeric(df_saneamento[code], errors='coerce') * multiplier

# Passo 4: Criar nova coluna 'pontuacao' com a soma das colunas modificadas
# Somente as colunas que estavam no dicionário (ou seja, que foram multiplicadas)
score_columns = [col for col in multiplier_map.keys() if col in df_saneamento.columns]
df_saneamento['pontuacao'] = df_saneamento[score_columns].sum(axis=1)

# Passo 6: Salvar o dataset modificado com a nova coluna
output_file = os.path.join('dataset_saneamento_modificado.csv')
df_saneamento.to_csv(output_file, index=False)
