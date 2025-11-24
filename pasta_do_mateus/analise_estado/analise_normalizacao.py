import pandas as pd
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("dados_m/dataset_m_dengue_saneamento.csv")

dataset_num = dataset.iloc[:, 12:]

dengue_ranges = dataset_num.describe().loc[['min', 'max']].T

scaler = StandardScaler()

dataset_z = pd.DataFrame(
    scaler.fit_transform(dataset_num),
    columns=dataset_num.columns
)

dataset_normalizado = pd.concat([dataset.iloc[:, :12], dataset_z], axis=1)

dataset_normalizado.to_csv("dados_m/dataset_m_normalizado.csv", index=False)
