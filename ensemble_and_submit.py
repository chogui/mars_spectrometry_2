import pandas as pd
import numpy as np
import pickle
from training import get_folds, ms_loss
from preprocessing import DATA_PATH, load_metadata

def ensemble_selector(loss_function, y_hats, y_true, init_size=1,
                      replacement=True, max_iter=100):
    
    """
    Implementation of the algorithm of Caruana et al. (2004) 'Ensemble
    Selection from Libraries of Models'. Given a loss function mapping
    predicted and ground truth values to a scalar along with a dictionary of
    models with predicted and ground truth values, constructs an optimal
    ensemble minimizing ensemble loss, by default allowing models to appear
    several times in the ensemble.

    Parameters
    ----------
    loss_function: function
        accepting two arguments - numpy arrays of predictions and true values - 
        and returning a scalar
    y_hats: dict
        with keys being model names and values being numpy arrays of predicted
        values
    y_true: np.array
        numpy array of true values, same for each model
    init_size: int
        number of models in the initial ensemble, picked by the best loss.
        Default is 1
    replacement: bool
        whether the models should be returned back to the pool of models once
        added to the ensemble. Default is True
    max_iter: int
        number of iterations for selection with replacement to perform. Only
        relevant if 'replacement' is True, otherwise iterations continue until
        the dataset is exhausted i.e.
        min(len(y_hats.keys())-init_size, max_iter). Default is 100

    Returns
    -------
    ensemble_loss: pd.Series
        with loss of the ensemble over iterations
    model_weights: pd.DataFrame
        with model names across columns and ensemble selection iterations
        across rows. Each value is the weight of a model in the ensemble

    """
    # Step 1: compute losses
    losses = dict()
    for model, y_hat in y_hats.items():
        losses[model] = loss_function(y_true, y_hat)

    # Get the initial ensemble comprised of the best models
    losses = pd.Series(losses).sort_values()
    init_ensemble = losses.iloc[:init_size].index.tolist()

    # Compute its loss
    if init_size == 1:
        # Take the best loss
        init_loss = losses.loc[init_ensemble].values[0]
        y_hat_avg = y_hats[init_ensemble[0]].copy()
    else:
        # Average the predictions over several models
        y_hat_avg = np.array(
            [y_hats[mod] for mod in init_ensemble]).mean(axis=0)
        init_loss = loss_function(y_true, y_hat_avg)

    # Define the set of available models
    if replacement:
        available_models = list(y_hats.keys())
    else:
        available_models = losses.index.difference(init_ensemble).tolist()
        # Redefine maximum number of iterations
        max_iter = min(len(available_models), max_iter)

    # Sift through the available models keeping track of the ensemble loss
    # Redefine variables for the clarity of exposition
    current_loss = init_loss
    current_size = init_size

    loss_progress = [current_loss]
    ensemble_members = [init_ensemble]
    for i in tqdm(range(max_iter)):
        # Compute weights for predictions
        w_current = current_size / (current_size + 1)
        w_new = 1 / (current_size + 1)

        # Try all models one by one
        tmp_losses = dict()
        tmp_y_avg = dict()
        for mod in available_models:
            tmp_y_avg[mod] = w_current * y_hat_avg + w_new * y_hats[mod]
            tmp_losses[mod] = loss_function(y_true, tmp_y_avg[mod])

        # Locate the best trial
        best_model = pd.Series(tmp_losses).sort_values().index[0]

        # Update the loop variables and record progress
        current_loss = tmp_losses[best_model]
        loss_progress.append(current_loss)
        y_hat_avg = tmp_y_avg[best_model]
        current_size += 1
        ensemble_members.append(ensemble_members[-1] + [best_model])

        if not replacement:
            available_models.remove(best_model)

    # Organize the output
    ensemble_loss = pd.Series(loss_progress, name="loss")
    model_weights = pd.DataFrame(index=ensemble_loss.index,
                                 columns=y_hats.keys())
    for ix, row in model_weights.iterrows():
        weights = pd.Series(ensemble_members[ix]).value_counts()
        weights = weights / weights.sum()
        model_weights.loc[ix, weights.index] = weights

    return ensemble_loss, model_weights.fillna(0).astype(float)

def get_ensemble_preds(fold):
    """
    Select the best model for fold and return weighted predictions.

    Parameters
    ----------
    fold: int

    Returns
    -------
    ens_preds: pd.DataFrame
        DataFrame with sample in test set as index and compounds as columns

    """
	# Load the validation indices and the predictions
	val_idcs = val_idcs_all[fold]
	y_true = df_train.loc[df_train.index[val_idcs], compounds].values
	y_hats_val = y_hats_val_all[fold]
	y_hats_pred = y_hats_pred_all[fold]

	# Consruct the ensemble for this fold and record predictions on test set
	ens_loss, mw = ensemble_selector(ms_loss, y_hats_val, y_true, replacement=True)
	mw = mw.iloc[-1].sort_values() # Weight on the last optimization round
	mw = mw.loc[mw > 0].copy() 

	# Get predictions on the test set
	ens_preds = pd.DataFrame(0, columns=np.sort(compounds), index=range(n_test))

	for model in mw.index:
		ens_preds += mw.loc[model] * y_hats_pred[model]

	return ens_preds


if __name__ == '__main__':

	# Get the metadata and the indices of the folds
    df_all, df_train, compounds = load_metadata()
    val_idcs_all = get_folds()
    n_test = len(df_all.loc[df_all['split'] != 'train'])

    # Load the predictions on the validation and test sets
    with open(DATA_PATH + 'predictions/all_predictions.pkl', 'rb') as f:
        dic = pickle.load(f)

    y_hats_val_all = dic['val_preds'] # Nested dictionary {fold: {model: preds}, ...}
    y_hats_pred_all = dic['tst_preds']

    # Get predictions
    all_preds = {fold: get_ensemble_preds(fold) for fold in val_idcs_all.keys()}

    # Output predictions
    sub = pd.DataFrame(np.mean(list(all_preds.values()), axis=0), columns=compounds)
	sub['sample_id'] = df_all.loc[df_all['split']!='train', 'sample_id'].values

	sub_template = pd.read_csv(DATA_PATH + 'submission_format.csv')

	sub[sub_template.columns].to_csv(DATA_PATH + 'all_preds_sub.csv', index=False)

