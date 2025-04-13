import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sklearn.model_selection
from pandas.core.algorithms import nunique_ints
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

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

def feature_importance_score_cal(train_X, train_y, test_X, test_y, feature_names):
    #### Train the GradientBoostingRegressor
    params = {
        'n_estimators': 400,
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.7,
    }
    gbr = GradientBoostingRegressor(**params, random_state=42)
    # fit on training data, apply to test data
    gbr.fit(train_X, train_y)
    test_mse = mean_squared_error(test_y, gbr.predict(test_X))
    test_r2 = gbr.score(test_X, test_y)
    print(f"Gradient boosting regressor on full test set gives MSE: {test_mse:.4f} and R^2 score: {test_r2:.4f}")
    feature_importance_score = gbr.feature_importances_

    sorted_idx = np.argsort(feature_importance_score)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance_score[sorted_idx], align="center")
    plt.yticks(pos, np.array(feature_names)[sorted_idx])
    plt.title("Feature Importance (MDI)")

    ## Box plot showing the variance of feature importance scores
    result = permutation_importance(
        gbr, test_X, test_y, n_repeats=10, random_state=42, n_jobs=2
    )
    sorted_idx = result.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(feature_names)[sorted_idx], # newer versions of matplotlib may use tick_labels as kwarg instead
    )
    plt.title("Permutation Importance (test set)")
    fig.tight_layout()
    plt.show()
    return feature_importance_score

def select_model(train_X_regr, test_X_regr, train_y_regr, test_y_regr):
    #Define MLP model
    param_grid = [
        {
            'hidden_layer_sizes': [(3,), (100,), (3, 3), (100, 100)],
            'activation': ['relu', 'logistic'],
            'max_iter': [50, 200, 500]
        },
    ]

    #Initialise GridSearchCV and fit
    grid_search = GridSearchCV(
        estimator=MLPRegressor(random_state=42),
        param_grid=param_grid,
        cv=4,
        scoring='neg_mean_squared_error',
        verbose=0,
        n_jobs=-1
    )
    grid_search.fit(train_X_regr, train_y_regr)

    #Report the best parameters and CV results
    best_model = grid_search.best_estimator_
    cv_results = grid_search.cv_results_
    best_index = grid_search.best_index_

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best CV MSE: {-grid_search.best_score_:.4f}")
    print(f"Std. Dev. of CV MSE: {cv_results['std_test_score'][best_index]:.4f}")

    #Report model performance with best parameters
    best_model.fit(train_X_regr, train_y_regr)
    test_pred = best_model.predict(test_X_regr)
    test_mse = mean_squared_error(test_y_regr, test_pred)
    test_r2 = r2_score(test_y_regr, test_pred)

    print("\nTest set performance:")
    print(f"MSE: {test_mse:.4f}")
    print(f"R² score: {test_r2:.4f}")

    train_pred = best_model.predict(train_X_regr)
    train_mse = mean_squared_error(train_y_regr, train_pred)
    train_r2 = r2_score(train_y_regr, train_pred)

    print("\nTraining set performance (for reference):")
    print(f"MSE: {train_mse:.4f}")
    print(f"R² score: {train_r2:.4f}")

def feature_importance_experiment():
    feature_names = soybean_data_processed.drop(columns=[target_col]).columns.values
    feature_importance_scores = feature_importance_score_cal(train_X_regr, train_y_regr, test_X_regr, test_y_regr, feature_names)
    pass


def main():
    #select_model(train_X_regr, test_X_regr, train_y_regr, test_y_regr)

    # best_model = MLPRegressor(
    #     hidden_layer_sizes=(100,),
    #     activation='relu',
    #     max_iter=200,
    #     early_stopping=True,  # Stop if no improvement
    #     random_state=42
    # )
    # best_model.fit(train_X_regr, train_y_regr)
    # train_pred = best_model.predict(train_X_regr)
    # test_pred = best_model.predict(test_X_regr)
    #
    # print("Train MSE:", mean_squared_error(train_y_regr, train_pred))
    # print("Test MSE:", mean_squared_error(test_y_regr, test_pred))

    feature_importance_experiment()

soybean_data_full = pd.read_csv("soybean_data.csv")
soybean_data_processed = preprocess_data(soybean_data_full)
target_col = 'Seed Yield per Unit Area (SYUA)'

#Separate the data from the labels
regression_targets = soybean_data_processed[target_col].to_numpy()
soybean_data = soybean_data_processed.drop(columns=[target_col])
regression_data = soybean_data.to_numpy()

# Perform data split
train_X_regr, test_X_regr, train_y_regr, test_y_regr = sklearn.model_selection.train_test_split(regression_data,
                                                                                                regression_targets,
                                                                                                test_size=0.15)
#Standardise
scaler = StandardScaler()
train_X_regr = scaler.fit_transform(train_X_regr)
test_X_regr = scaler.transform(test_X_regr)

main()
