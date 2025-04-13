import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sklearn.model_selection
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import model_eval_utils
from slp_mlp_Regression.model_eval_utils import run_model


def preprocess_data(soybean_data_full1):
    soybean_data_full1 = soybean_data_full.copy()

    encoder = OneHotEncoder()

    #Extract the genotype and one-hot encode it
    genotypes = encoder.fit_transform(soybean_data_full['Parameters'].str.extract(r'G(\d)')).toarray()
    soybean_data_full1 = pd.concat([soybean_data_full1, pd.DataFrame(genotypes, columns=[f'G{i}' for i in range(1, 7)])], axis=1)

    #Extract the salicylic acid treatment and encode it as 0, 250 mg, or 450 mg
    #1 = 250 mg, 2 = 450 mg, 3 = control
    salicylic_acid = soybean_data_full['Parameters'].str.extract(r'C(\d+)').astype(float)
    salicylic_acid = salicylic_acid.replace({1: 250, 2: 450, 3: 0})
    soybean_data_full1['Salicylic acid (mg)'] = salicylic_acid

    #Extract the water stress treatment and encode it as .05 or .7 of field capacity
    water_stress = soybean_data_full['Parameters'].str.extract(r'S(\d)').astype(float)
    water_stress = water_stress.replace({1: .05, 2: .7})
    soybean_data_full1['Water Stress (pct field capacity)'] = water_stress

    #Drop the original 'Parameters' column as well as 'Random' column
    soybean_data_full1.drop(columns=['Parameters', 'Random '], inplace=True)

    return soybean_data_full1

def feature_importance_score_cal(train_X, train_y, test_X, test_y, feature_names):
    #### Train the GradientBoostingRegressor
    params = {
        'n_estimators': 400,
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.7,
    }
    gbr = GradientBoostingRegressor(**params, random_state=445)
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
        gbr, test_X, test_y, n_repeats=10, random_state=445, n_jobs=2
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

def select_model():
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
        estimator=MLPRegressor(random_state=445),
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

def feature_importance_experiment(top_ns=None):
    # Calculate feature importance scores using feature_importance_score_cal()
    if top_ns is None:
        top_ns = [3, 6, 9, 12, 15]

    feature_names = soybean_data_processed.drop(columns=[target_col]).columns.values
    importance_scores = feature_importance_score_cal(train_X_regr, train_y_regr, test_X_regr, test_y_regr, feature_names)
    accs = []
    r2s = []

    #Best model determined from earlier
    best_mlp_params = {'activation': 'relu', 'hidden_layer_sizes': (100,), 'max_iter': 200, 'verbose': 0}

    #List the features in order of importance
    print(f"\nFeatures in order of importance:")
    top_features_idx = np.argsort(importance_scores)
    for i, idx in enumerate(top_features_idx[::-1]):
        print(f"{i + 1}. {feature_names[idx]} (score: {importance_scores[idx]:.4f})")


    # First train and evaluate the MLP determined from earlier model selection on all features
    mlp_full = MLPRegressor(**best_mlp_params, random_state=445)
    mlp_full.fit(train_X_regr, train_y_regr)
    full_pred = mlp_full.predict(test_X_regr)
    full_model_acc = mean_squared_error(test_y_regr, full_pred)
    full_r2 = r2_score(test_y_regr, full_pred)
    print("Testing model performance on different feature sets...")
    print(f"\nMLP (ALL features) MSE:{full_model_acc:.4f}")
    print(f"MLP (ALL features) F1: {full_r2:.4f}")

    for top_n in top_ns:

        print(f"\n==================Top {top_n} features importance scores==================")
        # Select important features

        top_features_idx = np.argsort(importance_scores)[-top_n:]
        # Validate the features
        train_X_reduced = train_X_regr[:, top_features_idx]
        test_X_reduced = test_X_regr[:, top_features_idx]

        #Train and evaluate MLP on reduced feature set
        mlp_reduced = MLPRegressor(**best_mlp_params, random_state=445)
        mlp_reduced.fit(train_X_reduced, train_y_regr)
        reduced_pred = mlp_reduced.predict(test_X_reduced)
        acc = mean_squared_error(test_y_regr, reduced_pred)
        r2 = r2_score(test_y_regr, reduced_pred)
        print(f"\nMLP (Top {top_n} features) MSE:{acc:.4f}")
        print(f"MLP (Top {top_n} features) F1: {r2:.4f}")
        accs.append(acc)
        r2s.append(r2)

    plot_feature_selection_results(top_ns, accs, r2s, full_model_acc, full_r2)

def plot_feature_selection_results(top_ns, accs, r2s, full_model_mse, full_model_r2):
    plt.figure(figsize=(12, 5))

    #MSE plot
    plt.subplot(1, 2, 1)
    plt.plot(top_ns, accs, 'bo-', label='Reduced Model')
    plt.axhline(y=full_model_mse, color='r', linestyle='--', label='Full Model')
    plt.xlabel('Number of Features')
    plt.ylabel('MSE')
    plt.title('MSE vs Feature Subset Size')
    plt.legend()

    # R² Plot (right)
    plt.subplot(1, 2, 2)
    plt.plot(top_ns, r2s, 'go-', label='Reduced Model')
    plt.axhline(y=full_model_r2, color='r', linestyle='--', label='Full Model')
    plt.xlabel('Number of Features')
    plt.ylabel('R² Score')
    plt.title('R² vs Feature Subset Size')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig("feature_selection_results.png")

def get_model_pipeline():
    #first deal with missing data
    imputer = SimpleImputer(strategy='median')
    numeric_transformer = Pipeline([
        ('imputer', imputer), #this line will fill any missing data with the median in that column
    ])
    #apply transformations (imputations) to each column
    cols_to_transform = np.arange(soybean_data_processed.shape[1] - 1)
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, cols_to_transform)
    ])

    #now define model (selected hyperparameters)
    mlp = MLPRegressor(
        activation='relu',
        hidden_layer_sizes=(100,),
        max_iter=200,
        early_stopping=True,
        random_state=42
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('mlp', mlp)
    ])

    #train
    pipeline.fit(train_X_regr, train_y_regr)

    return pipeline

def save_model(run=False):
    final_model = get_model_pipeline()
    student_username = "s89990lo"
    model_eval_utils.save_model(student_username, final_model)
    if run:
        run_model(student_username, test_X_regr, test_y_regr)

def main():

    save_model(True)


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
