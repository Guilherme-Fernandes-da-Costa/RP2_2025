import pandas as pd
from sklearn.preprocessing import StandardScaler

dengue = pd.read_csv("dataset_dengue_incidencia.csv")
saneamento = pd.read_csv("dataset_saneamento_estado.csv")

dengue_num = dengue.iloc[:, 3:]
saneamento_num = saneamento.iloc[:, 2:]

dengue_ranges = dengue_num.describe().loc[['min', 'max']].T
saneamento_ranges = saneamento_num.describe().loc[['min', 'max']].T

scaler = StandardScaler()

dengue_z = pd.DataFrame(
    scaler.fit_transform(dengue_num),
    columns=dengue_num.columns
)

saneamento_z = pd.DataFrame(
    scaler.fit_transform(saneamento_num),
    columns=saneamento_num.columns
)

dengue_normalizado = pd.concat([dengue.iloc[:, :2], dengue_z], axis=1)
saneamento_normalizado = pd.concat([saneamento.iloc[:, :2], saneamento_z], axis=1)

dengue_normalizado.to_csv("normalizado_dengue.csv", index=False)
saneamento_normalizado.to_csv("normalizado_saneamento.csv", index=False)
