import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sklearn.model_selection


def main():
    notebook_start_time = time.time()
    loan_data_full = pd.read_csv("loan_data.csv")
    print(loan_data_full.describe())

main()