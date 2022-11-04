import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from sklearn.decomposition import KernelPCA

from training import ms_loss, load_features, get_folds
from preprocessing import DATA_PATH, load_metadata


def objective_xgboost(trial, cpd, df_train, val_idcs_all, train_features):

    """
    This function is used to optimize a XGBoostClassifier using Optuna. 

    Parameters
    ----------

    trial: Optuna trial object used to track progress and store results
    df_train: pandas DataFrame containing the metadata and the labels
    val_idcs_all: list of indices for the validation set
    train_features: features DataFrame

    Returns
    -------

    loss: the loss of the CatBoostClassifier on the validation set
    
    """

    # Parameter grid to explore
    param_grid = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }

    if param_grid["booster"] == "gbtree" or param_grid["booster"] == "dart":
        param_grid["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param_grid["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param_grid["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param_grid["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param_grid["booster"] == "dart":
        param_grid["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param_grid["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param_grid["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param_grid["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    # Add a callback for pruning.
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-logloss")

    # Select a fold and prepare data
    fold = np.random.randint(0,5)
    trn_idcs = [i for i in df_train.index if i not in val_idcs_all[fold]]
    val_idcs =  val_idcs_all[fold]

    krn = KernelPCA(n_components=180, kernel='rbf')
    train_x, train_y = train_features.iloc[trn_idcs], df_train.loc[df_train.index[trn_idcs], cpd].values
    train_x = krn.fit_transform(train_x)
    val_x, val_y = train_features.iloc[val_idcs], df_train.loc[df_train.index[val_idcs], cpd].values
    val_x = krn.transform(val_x)

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(val_x, label=val_y)

    # Fit the classifier on training set with guidance from val set
    bst = xgb.train(param_grid, dtrain, evals=[(dvalid, "validation")], callbacks=[pruning_callback], num_boost_round=1000, early_stopping_rounds=100)
    
    # Extract probabilities on the validation set
    preds = bst.predict(dvalid)

    # Output the loss to minimize
    return ms_loss(val_y, preds)

def tune_xgboost(df_train, val_idcs_all, train_features):

    """
    This function tunes the xgboost classifier by running an optuna study. 
    The study is run on the pre-defined folds, and the best parameters are 
    extracted and saved for later use.

    Parameters
    ----------
    df_train : pandas DataFrame
        Dataframe containing the training labels 
    val_idcs_all : list
        List of indices for the validation set
    train_features : list
        Dataframe containing the features for the training set 

    Returns
    -------
    best_params : dict
    """

    # Initialize output
    best_params = {}

    # Loop through the compound and find the best params for each classifier
    for cpd in compounds:
      print('\n============',cpd,'============\n')

      # Create optuna study with pruner
      study = optuna.create_study(direction="minimize", study_name="XGBoost Classifier", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))

      # Define a function of trial to optimize and run optimizer
      func = lambda trial: objective_xgboost(trial, cpd, df_train, val_idcs_all, train_features)
      study.optimize(func, n_trials=100, timeout=36000, n_jobs=-1)

      # Extract the best parameters
      trial = study.best_trial
      best_params[cpd] = trial.params

      # Save for later use
    with open(DATA_PATH + 'models/best_xgb_params_kpca.pkl', 'wb') as f:
      pickle.dump(best_params, f)

    return best_params

if __name__ == '__main__':

    df_all, df_train, compounds = load_metadata()
    val_idcs_all = get_folds()
    full_train_features, _ = load_features()

    if not os.path.exists(DATA_PATH + 'models'):
        os.mkdir(DATA_PATH + 'models')

    # Tune models
    if not os.path.exists(DATA_PATH + 'models/best_xgb_params_kpca.pkl'):
        tune_xgboost(df_train, val_idcs_all, train_features)