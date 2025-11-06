import pandas as pd
import os

input_file = 'dataset_dengue.csv'

df = pd.read_csv(input_file)

columns_to_drop = df.columns[df.columns.get_loc('DRS__0_level_1__0'):df.columns.get_loc('Total_notificados__43') + 1]

df_cleaned = df.drop(columns=columns_to_drop)

output_folder = 'dataset_usado'
os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(output_folder, 'dataset_dengue_limpo.csv')

df_cleaned.to_csv(output_file, index=False)

#---------------------------------

dictionary_file = 'dicionário_saneamento.csv'
saneamento_file = 'dataset_saneamento_estado.csv'

df_saneamento = pd.read_csv(saneamento_file)

used_codes = [col for col in df_saneamento.columns if col.startswith('V')]

df_dictionary = pd.read_csv(dictionary_file, header=None, names=['Category', 'Subcategory', 'Code', 'Description'])

df_dictionary_filtered = df_dictionary[df_dictionary['Code'].isin(used_codes)]

df_dictionary_final = df_dictionary_filtered.drop(columns=['Category', 'Subcategory'])

output_folder = 'dataset_usado'
os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(output_folder, 'dicionario_filtrado.csv')
df_dictionary_final.to_csv(output_file, index=False)

print(f"Dicionário filtrado e processado salvo em: {output_file}")

#---------------------------------
