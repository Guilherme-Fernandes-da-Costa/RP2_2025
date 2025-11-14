import pandas as pd
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("dataset_dengue_saneamento.csv")

dataset_num = dataset.iloc[:, 7:]

dengue_ranges = dataset_num.describe().loc[['min', 'max']].T

scaler = StandardScaler()

dataset_z = pd.DataFrame(
    scaler.fit_transform(dataset_num),
    columns=dataset_num.columns
)

dataset_normalizado = pd.concat([dataset.iloc[:, :7], dataset_z], axis=1)

dataset_normalizado.to_csv("dataset_normalizado.csv", index=False)
