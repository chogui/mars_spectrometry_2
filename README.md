# Mars Spectrometry 2: Gas Chromatography challenge on DrivenData

This presents the implementation of my solution to the challenge (18th place).

## Requirements

The packages necessary to run the code are listed below.

`pip install -r requirements.txt`

## Data structure

The project is organized as follows:

```
gcms_data
+-- raw data
+-- features
+-- models
+-- predictions
```

## Generate submissions

There are two ways to run the code for this project: 
- sandalone notebook `gcms.ipynb`, or
- `.py` scripts in order to do the pre-processing, hyperparameters tuning, training, and predictions tasks:

```
python preprocessing.py
python hyperparameter_tuning.py
python training.py
python ensemble_and_submit.py
```

