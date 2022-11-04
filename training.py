import pandas as pd
import numpy as np
from skmultilearn.model_selection import IterativeStratification
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, KernelPCA
import pickle
import os
from joblib import dump, load
from preprocessing import DATA_PATH, load_metadata
from fastai.vision.all import *

def get_folds(N_folds=5):

	'''
	Get fold indices for cross validation

	Parameters
	----------
	N_folds : int, optional
	    Number of folds. The default is 5.

	Returns
	-------
	val_idcs_all : dict
	    Keys are the fold number and values are the indices
	    of the fold.
	'''

	stratifier_valid = IterativeStratification(n_splits=N_folds, order=10)
	# Have to do this bc random_state does not work
	if not os.path.exists(DATA_PATH + f'vmodels/al_indices_{N_folds}_fold.pkl'):
	    val_idcs_all= {}
	    for fold, (trn_idcs, val_idcs) in tqdm(enumerate(stratifier_valid.split(df_train, df_train[compounds + ['derivatized']]))):
	        val_idcs_all[fold] = val_idcs
	    with open(DATA_PATH + f'models/val_indices_{N_folds}_folds.pkl', 'wb') as f:
	        pickle.dump(val_idcs_all, f, protocol=4)
	else:
	    with open(DATA_PATH + f'models/val_indices_{N_folds}_folds.pkl', 'rb') as f:
	        val_idcs_all = pickle.load(f)
	return val_idcs_all

def load_features():

	'''
    Load features for a trained model.

    Returns
    -------

    full_train_features : pandas.DataFrame
        Dataframe containing all features for training.
    full_test_features : pandas.DataFrame
        Dataframe containing all features for testing.

    '''

	if not os.path.exists(DATA_PATH + 'features/train_features.parquet'):
		print('Train features not found.')
		return
	if not os.path.exists(DATA_PATH + 'features/test_features.parquet'):
		print('Test features not found.')
		return

	df_all, df_train, compounds = load_metadata()

	# Load data for RF
	full_train_features = pd.read_parquet(DATA_PATH + 'features/train_features.parquet')
	full_train_features = full_train_features.set_index(['mass', 'time_bin']).T
	full_train_features.columns = [str(l) for l in range(full_train_features.shape[1])]
	full_train_features['derivatized'] = df_train['derivatized'].astype(int).values

	full_test_features = pd.read_parquet(DATA_PATH + 'features/test_features.parquet')
	full_test_features = full_test_features.set_index(['mass', 'time_bin']).T
	full_test_features.columns = [str(l) for l in range(full_test_features.shape[1])]
	full_test_features['derivatized'] = df_all.loc[df_all['split']=='test', 'derivatized'].astype(int).values

	return full_train_features, full_test_features


def ms_loss(y_true, y_hat, eps=1e-15):

	"""
	Compute the competition loss

	Parameters
    ----------
	y_true: np.array
		True values
	y_hat: np.array
		Predictions
	eps: float
		Used to clip predictions to avoid log(0)

    Returns
    -------
    loss: float

	"""
	y_hat = y_hat.clip(eps, 1-eps)
	return - np.mean(y_true*np.log(y_hat) + (1-y_true)*np.log(1-y_hat))

def get_dfs(trn_idcs, val_idcs, kernel=None, cpd=None):
  
	"""
	This function applies dimensionalty reduction to the features dataframe 
	and splits the data into training, validation, and testing sets according to the indices that are passed in.

	Parameters
    ----------
	trn_idcs: np.array
		List of training indices
	val_idcs: np.array
		List of validation indices
	kernel: string
		Kernel to use for dimensionality reduction if any
	cpd: string
		Compound to treat as target if training compounds separately

    Returns
    -------
    train_x: np.array
    	Array with training features
    train_y: np.array
	    Array with training labels
    val_x: np.array
    	Array with validation features
    val_y: np.array
    	Array with validation labels
    tst_x: np.array
    	Array with test features

	"""

  if kernel == 'kpca':
    krn = KernelPCA(n_components=180, kernel='rbf') # n_components chosen to preserve ~95% of the variance (empirically)
  elif kernel == 'pca':
    krn = PCA(n_components=100) # max for TabPFN

  if cpd:
    # target is only one compound
    train_x, train_y = train_features.iloc[trn_idcs], df_train.loc[df_train.index[trn_idcs], cpd].values
    val_x, val_y = train_features.iloc[val_idcs], df_train.loc[df_train.index[val_idcs], cpd].values
    # Competition predictions require original validation set and test set
    tst_x = pd.concat([train_features.iloc[metadata.loc[metadata['split'] == 'val'].index], test_features])

    if kernel:
    	train_x = krn.fit_transform(train_x)
    	val_x = krn.transform(val_x)
	    tst_x = krn.transform(tst_x)

  else:
    # target is for multi-label classification
    train_x, train_y = train_features.iloc[trn_idcs], df_train.loc[df_train.index[trn_idcs], compounds].values
    val_x, val_y = train_features.iloc[val_idcs], df_train.loc[df_train.index[val_idcs], compounds].values
    # Competition predictions require original validation set and test set
    tst_x = pd.concat([train_features.iloc[metadata.loc[metadata['split'] == 'val'].index], test_features])

    if kernel:
    	train_x = krn.fit_transform(train_x)
    	val_x = krn.transform(val_x)
	    tst_x = krn.transform(tst_x)

  return train_x, train_y, val_x, val_y, tst_x


def update_predictions(fold, model, val_preds, tst_preds):
  
	"""
	This function saved the prediction of a given model on the val set for a given fold and on the test set.

	Parameters
    ----------
	fold: int
		Use to index the prediction dictionaries
	model: str
		Name of the model
	val_preds: np.array
		Predictions on the validation set for a given fold
	tst_preds: np.array
		Predictions on the test set

    Returns
    -------

  """
  if os.path.exists(DATA_PATH + 'predictions/all_predictions.pkl'):
    # Load
    with open(DATA_PATH + 'predictions/all_predictions.pkl', 'rb') as f:
        dic = pickle.load(f)

    y_hats_val_all = dic['val_preds']
    y_hats_pred_all = dic['tst_preds']
    # Update
    y_hats_val_all[fold][model] = val_preds
    y_hats_pred_all[fold][model] = tst_preds

  else:
    # Create
    y_hats_val_all = {i: {} for i in range(N_folds)}
    y_hats_pred_all = {i: {} for i in range(N_folds)}
    # Update
    y_hats_val_all[fold][model] = val_preds
    y_hats_pred_all[fold][model] = tst_preds

  # Save 
  with open(DATA_PATH + 'predictions/all_predictions.pkl', 'wb') as f:
    pickle.dump({'val_preds': y_hats_val_all, 'tst_preds': y_hats_pred_all}, f)
  
  return


####### Vision models #######

# Implement label smoothing
class MultiLabelSmoothingCallback(Callback):
    def __init__(self, ε:float=0.05):
        super().__init__()
        self.ε = ε
    
    def before_backward(self, **kwargs):
        labels = self.yb[0].detach().clone()
        for i, x in enumerate(labels):
            labels[i] = torch.where(x==1.0, 1.0-self.ε, self.ε)
        smoothed_criterion = nn.BCEWithLogitsLoss()(self.pred, labels)
        return {'last_loss': smoothed_criterion}

def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):

    """Compute accuracy when `inp` and `targ` are the same size."""

    if sigmoid: inp = inp.sigmoid()
    return ((inp>thresh)==targ.bool()).float().mean()

def prep_dls(arch, df_train, val_idcs, size=None, bs=16):
	
	"""
    Returns a fastai dataloader and learner object for a given architecture. 
    
    Parameters:
    ----------
    arch: fastai architecture object
        The architecture to build a learner for.
    df_train: pd.DataFrame
         A DataFrame of training data.
    val_idcs: array
        A numpy array of indices to use as validation.
    size: tuple
        A tuple with the dimensions of an image input.
    bs: int
        Batch size.

    Returns
    -------
    learn: fastai vision learner
    dls: dataloaders
    df_tst: pd.DataFrame
    	Dataframe with the path to the validation and test images
    """

    if size:
        dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                      get_x = lambda r: DATA_PATH + r['features_path'].split('.')[0] + '_224.jpg', 
                      get_y = lambda r: r['labels'].split(' '),
                      splitter=ColSplitter())
    else:
        dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                      get_x = lambda r: DATA_PATH + r['features_path'].split('.')[0] + '.jpg', 
                      get_y = lambda r: r['labels'].split(' '),
                      splitter=ColSplitter(),
                      )

    # train + valid
    df = df_train.copy()
    df['is_valid'] = 0
    df.loc[val_idcs, 'is_valid'] = 1

    # test (original val + test -- submission requirement)
    if size:
        df_tst = metadata.loc[metadata['split'] != 'train', 'features_path'].apply(lambda x: DATA_PATH + x.split('.')[0] + '_224.jpg').copy()
    else:
        df_tst = metadata.loc[metadata['split'] != 'train', 'features_path'].apply(lambda x: DATA_PATH + x.split('.')[0] + '.jpg').copy()

    dls = dblock.dataloaders(df, bs=bs)
    learn = vision_learner(
      dls, arch,
      metrics=[partial(accuracy_multi, thresh=0.3), f1_macro, f1_samples],
      cbs=[MultiLabelSmoothingCallback(), MixUp()])
    return learn, dls, df_tst

class VisionModel(object):
	def __init__(self, arch, size, df_train, val_idcs_all, save_path, bs=16):
		self.arch = arch
		self.size = size
		self.df_train = df_train
		self.bs = bs
		self.val_idcs_all = val_idcs_all
		self.val_preds = {}
		self.tst_preds = {}

	def train_fold(self, n_epochs=20, fold=0):

		"""
	    Trains the vision model on the fold.
	    
	    Returns
	    -------
	    val_preds: pd.DataFrame
	    	DataFrame with predictions on the fold's validation set
	    tst_preds: pd.DataFrame
	    	DataFrame with predictions on the competition's test set
	    """

		val_idcs = self.val_idcs_all[fold]
		learn, dls, df_tst = prep_dls(self.arch, self.df_train, val_idcs, size=self.size, bs=self.bs)

	    # Initial training
	    lr = learn.lr_find(suggest_funcs=(slide), show_plot=False)[0]
	    learn.fit_one_cycle(n_epochs, lr)

	    # Save the model
	    learn.save(DATA_PATH + f'models/{self.arch}_{fold}')
	    
	    # Make predictions
	    val_preds, _ = learn.tta(dl=dls.valid)
	    val_preds = pd.DataFrame(val_preds[:, 1:], columns=learn.dls.vocab[1:]) # 0 is no label

	    tst_dl = dls.test_dl(df_tst)
	    tst_preds, _ = learn.tta(dl=tst_dl)
	    tst_preds = pd.DataFrame(tst_preds[:, 1:], columns=learn.dls.vocab[1:])
	    return val_preds, tst_preds

	def predict_all(self):

		"""
	    Trains the vision model on all fold and saves predictions on val and test sets.
	    
	    """

		for fold in tqdm(self.val_idcs_all.keys()):
    		val_idcs = self.val_idcs_all[fold]

    		if not os.path.exists(DATA_PATH + f'models/{self.arch}_{fold}.pth'):
    			val_preds, tst_preds = self.train_fold(fold=fold)

    		else:
    			# Load model
    			learn, dls, df_tst = prep_dls(self.arch, self.df_train, val_idcs, size=self.size, bs=self.bs)
    			learn.load(DATA_PATH + f'models/{self.arch}_{fold}')

			    val_preds, _ = learn.tta(dl=dls.valid)
			    val_preds = pd.DataFrame(val_preds[:, 1:], columns=learn.dls.vocab[1:]) # 0 is no label

			    tst_dl = dls.test_dl(df_tst)
			    tst_preds, _ = learn.tta(dl=tst_dl)
			    tst_preds = pd.DataFrame(tst_preds[:, 1:], columns=learn.dls.vocab[1:])

			self.val_preds[fold] = val_preds
			self.tst_preds[fold] = tst_preds

			# Save
			update_predictions(fold, self.arch, val_preds.values, tst_preds.values)
		return 

####### Other models #######

class Model(object):

	def __init__(self, name, df_train, val_idcs_all, train_features, test_features, save_path):

		self.name = name
		self.df_train = df_train

		if self.name == 'RF':
			self.params = dict(max_features='sqrt', bootstrap=True, criterion='entropy', n_estimators=700) 

		elif self.name == 'XGBoost':
			with open(DATA_PATH + 'models/best_xgb_params_kpca.pkl', 'rb') as f:
				self.params = pickle.load(f)

		elif self.name == 'TabPFN':
			self.params = {'device': 'cuda'}

		self.val_idcs_all = val_idcs_all
		self.train_features = train_features
		self.test_features = test_features
		self.val_preds = {}
		self.tst_preds = {}
		self.save_path = save_path


	def train_fold(self, cpd, fold=0):

		"""
	    Trains the vision model on the fold.
	    
	    Returns
	    -------
	    val_preds: pd.DataFrame
	    	DataFrame with predictions on the fold's validation set
	    tst_preds: pd.DataFrame
	    """

		# Define indices for this fold
		val_idcs = self.val_idcs_all[fold]
		trn_idcs = [l for l in self.df_train.index if l not in val_idcs]

	    if self.name == 'XGBoost':
	    	train_x, train_y, val_x, val_y, tst_x = get_dfs(trn_idcs, val_idcs, kernel='kpca', cpd=cpd)

		    dtrain = xgb.DMatrix(train_x, label=train_y)
		    dvalid = xgb.DMatrix(val_x, label=val_y)
		    dtest = xgb.DMatrix(tst_x)

		    # Fit the classifier on training set with guidance from val set
		    model = xgb.train({**self.params[cpd], **{"verbosity": 0,"objective": "binary:logistic","eval_metric": "logloss"}}, dtrain, evals=[(dvalid, "validation")], num_boost_round=1000, early_stopping_rounds=100)

		    # Extract probabilities on the validation set
		    val_pred = model.predict(dvalid)
		    tst_pred = model.predict(dtest)

		elif self.name == 'RF':
			train_x, train_y, val_x, val_y, tst_x = get_dfs(trn_idcs, val_idcs, cpd=cpd)

			model = RandomForestClassifier(**self.params)
			model.fit(train_x, train_y)

			val_pred = model.predict(val_x)
			tst_pred = model.predict(tst_x)

		elif self.name == 'TabPFN':
			train_x, train_y, val_x, val_y, tst_x = get_dfs(trn_idcs, val_idcs, kernel='pca', cpd=cpd)

			model = TabPFNClassifier(**self.params)
		    model.fit(train_x, train_y)

	        clas, y_pred = model.predict(val_x, return_winning_probability=True)
		    y_pred[clas == 0] = 1-y_pred[clas==0]
		    val_pred = y_pred # Probability of class 1 (cpd is present)
		    
		    # The predictions to submit are the original val set + the test set
		    clas, y_hat_test = model.predict(tst_x, return_winning_probability=True)
		    y_hat_test[clas == 0] = 1-y_hat_test[clas==0]
		    tst_pred = y_hat_test

		# Save trained model
		dump(model, DATA_PATH + f'models/{self.name}_{cpd}_{fold}.joblib')

		return val_pred, tst_pred

	def predict_all(self):

		"""
	    Trains the model on all fold and saves predictions on val and test sets.
	    
	    """

		for fold in tqdm(self.val_idcs_all.keys()):
    		val_idcs = self.val_idcs_all[fold]

    		tmp_val_preds = {}
    		tmp_tst_preds = {}

    		for cpd in compounds:

	    		if not os.path.exists(DATA_PATH + f'models/{self.name}_{cpd}_{fold}.joblib'):
	    			val_pred, tst_pred = self.train_fold(fold=fold)

	    		else:
	    			# Load model
	    			model = load(DATA_PATH + f'models/{self.name}_{fold}.joblib')

	    			if self.name == 'RF':
	    				train_x, train_y, val_x, val_y, tst_x = get_dfs(trn_idcs, val_idcs, cpd=cpd)
						
						tmp_val_preds[cpd] = model.predict(val_x)
						tmp_tst_preds[cpd] = model.predict(tst_x)

					elif self.name == 'XGBoost':
				    	train_x, train_y, val_x, val_y, tst_x = get_dfs(trn_idcs, val_idcs, kernel='kpca', cpd=cpd)

					    dvalid = xgb.DMatrix(val_x, label=val_y)
					    dtest = xgb.DMatrix(tst_x)

					    # Extract probabilities on the validation set
					    tmp_val_preds[cpd] = model.predict(dvalid)
					    tmp_tst_preds[cpd] = model.predict(dtest)

					elif self.name == 'TabPFN':
						train_x, train_y, val_x, val_y, tst_x = get_dfs(trn_idcs, val_idcs, kernel='pca', cpd=cpd)

				        clas, y_pred = model.predict(val_x, return_winning_probability=True)
					    y_pred[clas == 0] = 1-y_pred[clas==0]
					    tmp_val_preds[cpd] = y_pred # Probability of class 1 (cpd is present)
					    
					    # The predictions to submit are the original val set + the test set
					    clas, y_hat_test = model.predict(tst_x, return_winning_probability=True)
					    y_hat_test[clas == 0] = 1-y_hat_test[clas==0]
					    tmp_tst_preds[cpd] = y_hat_test

			tmp_val_preds = pd.DataFrame(tmp_val_preds)
			tmp_tst_preds = pd.DataFrame(tmp_tst_preds)

			self.val_preds[fold] = tmp_val_preds
			self.tst_preds[fold] = tmp_tst_preds

			# Save
			update_predictions(fold, self.name, tmp_val_preds.values, tmp_tst_preds.values)

		return


if __name__ == '__main__':
	
    df_all, df_train, compounds = load_metadata()
    val_idcs_all = get_folds()
    full_train_features, full_test_features = load_features()

	########### RF, XGBoost, and TabPFN training ###########
	# Initialize a model and call predict_all, this will train it and save the predictions on the val and tst sets
	rf = Model('RF', df_train, val_idcs_all, full_train_features, full_test_features, SAVE_PATH)
	xgb = Model('XGBoost', df_train, val_idcs_all, full_train_features, full_test_features, SAVE_PATH)
	tabpfn = Model('TapPFN', df_train, val_idcs_all, full_train_features, full_test_features, SAVE_PATH)

	for model in [rf, xgb, tabpfn]:
		model.predict_all()

	########### Vision training ###########
	# List of models to use
	vision_models = {
	    'vit_small_patch16_224': 224,
	    'swin_base_patch4_window7_224_in22k': 224,
	    'vit_small_patch32_224': 224,
	    'convnext_tiny_in22k': None,
	    'vit_base_patch32_224': 224,
	    'swinv2_cr_tiny_ns_224': 224,
	    'beit_base_patch16_224': 224,
	    'vit_base_patch16_224_miil': 224,
	    'vit_base_patch16_224': 224,
	    'convnext_small_in22k': None,
	    'efficientnet_b0': 224
	}
	for arch, size in vision_models.items():
		model = VisionModel(arch, size, df_train, val_idcs_all, SAVE_PATH)
		mode.predict_all()
