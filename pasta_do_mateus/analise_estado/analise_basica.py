import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import stats  # For mode, if needed

def analyze_column(csv_file, column_name):
    # carrefa o nosso dataset
    df = pd.read_csv(csv_file)
    
    # Valida os valores das colunas
    if column_name not in df.columns:
        print(f"Coluna '{column_name}' não encontrada no Dataset.")
        return
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        print(f"Coluna '{column_name}' não é numérica. Essa análise requer valores numéricos.")
        return
    
    # Drop NaN values for analysis
    data = df[column_name].dropna()
    
    if len(data) == 0:
        print("Coluna vazia após remoção de NaNs.")
        return
    
    # Calculate statistics
    mean = data.mean()
    median = data.median()
    mode = stats.mode(data)[0]  # Using scipy.stats.mode for handling multimodal
    std_dev = data.std()
    variance = data.var()
    
    # Print the results
    print(f"Análise para a coluna: {column_name}")
    print(f"Média: {mean}")
    print(f"Mediana: {median}")
    print(f"Moda: {mode}")
    print(f"Devio Padrão: {std_dev}")
    print(f"Variância: {variance}")
    
    # Generate Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins='auto', color='blue', alpha=0.7)
    plt.title(f'Histograma de {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequencia')
    plt.grid(True)
    hist_filename = f'histograma_{column_name}.png'
    plt.savefig(hist_filename)
    print(f"Histograma salvo como: {hist_filename}")
    
    # Generate Boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, vert=False)
    plt.title(f'Boxplot de {column_name}')
    plt.xlabel(column_name)
    plt.grid(True)
    box_filename = f'boxplot_{column_name}.png'
    plt.savefig(box_filename)
    print(f"Boxplot salvo como: {box_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Como uso? R.: Digite no terminal `python script.py <nome da coluna que vc quer analisar>`")
        sys.exit(1)
    
    csv_file = 'dataset_m_dengue_saneamento.csv'  # Assumindo esse arquivo aqui
    column_name = sys.argv[1]
    analyze_column(csv_file, column_name)