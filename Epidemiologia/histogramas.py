import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar bruto
df_bruto = pd.read_excel("dengue2022_mes.xlsx", skiprows=6)
saneam_bruto = pd.read_csv(".../datasets_brutos/saneamento_2022.csv")
df_bruto = df_bruto.merge(saneam_bruto, on='cod_ibge', how='left')

# Tratamento
df_tratado = df_bruto.copy()
df_tratado['CI_10k'] = (df_tratado['Total confirmados autóctones'] / df_tratado['pop']) * 100000
df_tratado = df_tratado.dropna(subset=['CI_10k', 'SANIT', 'ABAST'])
df_tratado['SANIT'] = df_tratado['SANIT'].clip(0, 100)
df_tratado['log_renda'] = np.log(df_tratado['renda_media'])

# Histogramas
vars = ['CI_10k', 'SANIT', 'ABAST', 'LIXO', 'renda_media']
for var in vars:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df_bruto[var].dropna(), bins=20, ax=ax1, color='red', alpha=0.7)
    ax1.set_title(f'{var} - BRUTO')
    treated_var = 'log_renda' if var == 'renda_media' else var
    sns.histplot(df_tratado[treated_var].dropna(), bins=20, ax=ax2, color='green', alpha=0.7)
    ax2.set_title(f'{var} - TRATADO')
    plt.savefig(f'comparacao_{var}.png')
    plt.close()
