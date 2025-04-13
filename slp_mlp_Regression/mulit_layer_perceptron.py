import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sklearn.model_selection
from pandas.core.algorithms import nunique_ints
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(soybean_data_full):
    soybean_data_processed = soybean_data_full.copy()

    encoder = OneHotEncoder()

    #Extract the genotype and one-hot encode it
    genotypes = encoder.fit_transform(soybean_data_full['Parameters'].str.extract(r'G(\d)')).toarray()
    soybean_data_processed = pd.concat([soybean_data_processed, pd.DataFrame(genotypes, columns=[f'G{i}' for i in range(1, 7)])], axis=1)

    #Extract the salicylic acid treatment and encode it as 0, 250 mg, or 450 mg
    #1 = 250 mg, 2 = 450 mg, 3 = control
    salicylic_acid = soybean_data_full['Parameters'].str.extract(r'C(\d+)').astype(float)
    salicylic_acid = salicylic_acid.replace({1: 250, 2: 450, 3: 0})
    soybean_data_processed['Salicylic acid (mg)'] = salicylic_acid

    #Extract the water stress treatment and encode it as .05 or .7 of field capacity
    water_stress = soybean_data_full['Parameters'].str.extract(r'S(\d)').astype(float)
    water_stress = water_stress.replace({1: .05, 2: .7})
    soybean_data_processed['Water Stress (pct field capacity)'] = water_stress

    #Drop the original 'Parameters' column as well as 'Random' column
    soybean_data_processed.drop(columns=['Parameters', 'Random '], inplace=True)

    return soybean_data_processed

def main():
    soybean_data_full = pd.read_csv("soybean_data.csv")
    soybean_data_preprocessed = preprocess_data(soybean_data_full)

main()
