import pandas as pd
import numpy as np
import optuna
from optuna.integration import CatBoostPruningCallback
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
from lightgbm import early_stopping
from lightgbm import log_evaluation
from sklearn.decomposition import KernelPCA

from training import ms_loss, load_features, get_folds
from preprocessing import DATA_PATH, load_metadata

def objective_catboost(trial, df_train, val_idcs_all, train_features):
    '''
    Define the parameter grid and the CatBoost classifier for optuna to explore
    '''

    # Parameter grid to explore
    param_grid = {
        'loss_function': 'MultiCrossEntropy',
        'eval_metric': 'MultiLogLoss',
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', .01, .5, log=True),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 10),
        "random_strength" : trial.suggest_int("random_strength", 1, 10),
    }

    if param_grid["bootstrap_type"] == "Bayesian":
        param_grid["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param_grid["bootstrap_type"] == "Bernoulli":
        param_grid["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

    # Initialize classifier and pruning
    gbm = CatBoostClassifier(**param_grid)
    pruning_callback = CatBoostPruningCallback(trial, "MultiLogloss")

    # Select a fold and prepare data
    fold = np.random.randint(0,5)
    trn_idcs = [i for i in df_train.index if i not in val_idcs_all[fold]]
    val_idcs =  val_idcs_all[fold]

    krn = KernelPCA(n_components=180, kernel='rbf')
    train_x, train_y = train_features.iloc[trn_idcs], df_train.loc[df_train.index[trn_idcs], cpd].values
    train_x = krn.fit_transform(train_x)
    val_x, val_y = train_features.iloc[val_idcs], df_train.loc[df_train.index[val_idcs], cpd].values
    val_x = krn.transform(val_x)

    # Fit the classifier on training set with guidance from val set
    gbm.fit(
        train_x, 
        train_y, 
        eval_set=(val_x, val_y),
        verbose=0,
        early_stopping_rounds=100,
        callbacks=[pruning_callback],
    )

    # Extract probabilities on the validation set
    preds = gbm.predict_proba(val_x)

    # Output the loss to minimize
    return ms_loss(val_y, preds)

def tune_catboost(df_train, val_idcs_all, train_features):
    '''Get the best paramaters on the pre-defined folds'''

    # Initialize output
    best_params = {}

    # Create optuna study with pruner
    study = optuna.create_study(direction="minimize", study_name="CatBoost Classifier", 
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))

    # Define a function of trial to optimize and run optimizer
    def func(trial): return objective_catboost(trial, df_train, val_idcs_all, train_features)
    study.optimize(func, n_trials=100, timeout=36000, n_jobs=-1)

    # Extract the best parameters
    trial = study.best_trial
    best_params['catboost'] = trial.params

    # Save for later use
    with open(DATA_PATH + 'models/best_catb_params_kpca.pkl', 'wb') as f:
      pickle.dump(best_params, f)
    return

def objective_xgboost(trial, cpd, df_train, val_idcs_all, train_features):
    '''
    Define the parameter grid and the XGBoost classifier for optuna to explore
    '''

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
    '''Get the best paramaters on the pre-defined folds'''

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

    return

def objective_lightgbm(trial, cpd, df_train, val_idcs_all, train_features):
    '''
    Define the parameter grid and the XGBoost classifier for optuna to explore
    '''

    # Parameter grid to explore
    param_grid = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbose': -1,
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    # Add a callback for pruning.
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "validation-binary_logloss")

    # Select a fold and prepare data
    fold = np.random.randint(0,5)
    trn_idcs = [i for i in df_train.index if i not in val_idcs_all[fold]]
    val_idcs =  val_idcs_all[fold]

    krn = KernelPCA(n_components=180, kernel='rbf')
    train_x, train_y = train_features.iloc[trn_idcs], df_train.loc[df_train.index[trn_idcs], cpd].values
    train_x = krn.fit_transform(train_x)
    val_x, val_y = train_features.iloc[val_idcs], df_train.loc[df_train.index[val_idcs], cpd].values
    val_x = krn.transform(val_x)

    dtrain = lgb.Dataset(train_x, label=train_y)
    dval = lgb.Dataset(val_x, label=val_y)

    # Fit the classifier on training set with guidance from val set
    model = lgb.train(param_grid, 
                      dtrain,
                      valid_sets=[dtrain, dval],
                      callbacks=[early_stopping(100),pruning_callback])

    
    # Extract probabilities on the validation set
    preds = model.predict(val_x, num_iteration=model.best_iteration)

    # Output the loss to minimize
    return ms_loss(val_y, preds)

def tune_lightgbm(df_train, val_idcs_all, train_features):
    '''Get the best paramaters on the pre-defined folds'''

    # Initialize output
    best_params = {}

    # Loop through the compound and find the best params for each classifier
    for cpd in compounds:
      print('\n============',cpd,'============\n')

      # Create optuna study with pruner
      study = optuna.create_study(direction="minimize", study_name="LightGBM Classifier", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))

      # Define a function of trial to optimize and run optimizer
      func = lambda trial: objective_lightgbm(trial, cpd, df_train, val_idcs_all, train_features)
      study.optimize(func, n_trials=100, timeout=36000, n_jobs=-1)

      # Extract the best parameters
      trial = study.best_trial
      best_params[cpd] = trial.params

      # Save for later use
    with open(DATA_PATH + 'models/best_lgb_params_kpca.pkl', 'wb') as f:
      pickle.dump(best_params, f)

    return

if __name__ == '__main__':

    df_all, df_train, compounds = load_metadata()
    val_idcs_all = get_folds()
    full_train_features, _ = load_features()

    # Tune models
    if not os.path.exists(DATA_PATH + 'models/best_catb_params_kpca.pkl'):
        tune_catboost(df_train, val_idcs_all, train_features)
    if not os.path.exists(DATA_PATH + 'models/best_xgb_params_kpca.pkl'):
        tune_xgboost(df_train, val_idcs_all, train_features)
    if not os.path.exists(DATA_PATH + 'models/best_lgb_params_kpca.pkl'):
        tune_lightgbm(df_train, val_idcs_all, train_features)