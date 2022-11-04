import pandas as pd
import numpy as np
import itertools
import pybaselines
from PIL import Image
from tqdm import tqdm

# Set data path
DATA_PATH = 'gcms_data/'

def load_metadata():
    
    """
    Load labels and prepare the dataframes with file paths and labels
    
    Parameters
    ----------
    None
        
    Returns
    -------
    df_all : pandas.DataFrame
        DataFrame of file paths and labels for training, validation, and test sets.
    df_train : pandas.DataFrame
        DataFrame of file paths and labels for training, and validation sets.
    compounds : list
        List of coumpound names, sorted alphabetically.
    """

    metadata = pd.read_csv(DATA_PATH + 'metadata.csv')
    metadata['derivatized'] = metadata['derivatized'].fillna(0)

    train_files = metadata[metadata["split"] != "test"]["features_path"].to_dict()
    test_files = metadata[metadata["split"] == "test"]["features_path"].to_dict()

    print("Number of training samples: ", len(train_files))
    print("Number of testing samples: ", len(test_files))

    train_labels = pd.read_csv(DATA_PATH + 'train_labels.csv')
    val_labels = pd.read_csv(DATA_PATH + 'val_labels.csv')

    # Sort the columns alphabetically
    train_labels = pd.concat([train_labels[['sample_id']], train_labels.drop(['sample_id'],axis=1).pipe(lambda df: df.reindex(sorted(df.columns), axis=1))], axis=1)
    val_labels = pd.concat([val_labels[['sample_id']], val_labels.drop(['sample_id'],axis=1).pipe(lambda df: df.reindex(sorted(df.columns), axis=1))], axis=1)

    # Merge labels with metadata
    df_all = (metadata
                  .drop(['features_md5_hash'], axis=1)
                  .merge(pd.concat([train_labels, val_labels]), on='sample_id', how='left')
                )
    df_train = df_all.loc[df_all['split'] != 'test'].copy()

    # Extract list of compounds
    compounds = list(train_labels.drop(['sample_id'], axis=1).columns)

    # Create text-based labels to input into the datablock for vision models
    df_labels = df_train.copy()
    for comp in df_labels.columns:
      df_labels.loc[df_labels[comp] == 1, comp] = comp
      df_labels.loc[df_labels[comp] == 0, comp] = ''
    a = df_labels[compounds].agg(lambda x: re.sub(' +', ' ', ' '.join(x)).strip(), axis=1)
    df_train['labels'] = a

    return df_all, df_train, compounds

def get_bins():

    """
    This function generates a series of time bins, which are used to create a dataframe of all possible combinations 
    of time and mass values.

    Returns
    -------

    timerange (pd.interval_range): A series of time bins with a frequency of 0.5
    timerange_224 (pd.interval_range): A series of time bins with a frequency of 0.5 and 224 periods
    massrange_224 (pd.interval_range): A series of mass bins with a frequency of 1 and 224 periods
    allcombs_df (pd.DataFrame): A dataframe with all possible combinations of time and mass values
    allcombs_224_df (pd.DataFrame): A dataframe with all possible combinations of time and mass values
    
    """

    # Create a series of time bins
    timerange = pd.interval_range(start=0, end=45, freq=0.5)
    timerange_224 = pd.interval_range(start=0, end=45, periods=224)
    massrange_224 = pd.interval_range(start=12, end=350, periods=224)

    # Make dataframe with rows that are combinations of all temperature bins and all m/z values
    allcombs = list(itertools.product(timerange, [*range(12, 350)]))
    allcombs_df = pd.DataFrame(allcombs, columns=["time_bin", "mass_bin"])

    allcombs_224 = list(itertools.product(timerange_224, massrange_224))
    allcombs_224_df = pd.DataFrame(allcombs, columns=["time_bin", "mass_bin"])

    return timerange, timerange_224, massrange_224, allcombs_df, allcombs_224_df

def generate_features(flpth, image=False, size_224=False):

    """
    Generate features from a raw data file (path in the metadata df)
    
    Parameters
    ----------
    flpth : str
        Path to raw data file.
    image : bool, default=False
        Whether to generate an image of the features.
    size_224 : bool, default=False
        Whether to generate a 224x224 image.
        
    Returns
    -------
    df_out : pandas.DataFrame
        DataFrame of features.
    """
    
    df_src = pd.read_csv(DATA_PATH + 'raw/' + flpth)
    df = df_src.copy()
    
    # Round mass
    df['mass'] = np.round(df['mass']).astype(int)

    # Clip the masses to a reasonable range (also gets rid of He at 4.0)
    df = df.loc[df['mass'].between(12, 350)].reset_index(drop=True)
    df = df.groupby(['time', 'mass'])['intensity'].mean().reset_index()
    
    # Scale intensity among mass range of interest
    df['intensity'] /= df['intensity'].max()

    # Load binning info
    timerange, timerange_224, massrange_224, allcombs_df, allcombs_224_df = get_bins()
    
    # Bin time and take max intensity in bin
    if size_224:
        df["time_bin"] = pd.cut(df["time"], bins=timerange_224)
        df['mass_bin'] = pd.cut(df['mass'], bins=massrange_224)
    else:
        # if we don't want a 224x224 image, we bin the masses by 1-unit increment
        df["time_bin"] = pd.cut(df["time"], bins=timerange)
        df['mass_bin'] = df['mass']


    # Combine with a list of all time and m/z combinations
    if size_224:
        df = pd.merge(allcombs_df, df, on=["time_bin", "mass_bin"], how="left")
    else:
        df = pd.merge(allcombs_224_df, df, on=["time_bin", "mass_bin"], how="left")

    # Aggregate by time bin to find max
    df = df.groupby(["time_bin", "mass_bin"]).max("intensity").reset_index()
    df['intensity'].fillna(0, inplace=True)

    # Pivot and remove baseline by mass channel
    df_piv = df.pivot(index='time_bin', columns='mass_bin', values='intensity')

    # Filter out the baseline
    df_out = df_piv.copy()
    bkgs = {}
    # Process one mass channel at a time
    for col in df_out.columns:
        # Need to be non-zero
        if df_out[col].abs().sum() > 0:
            bkg_2 = pybaselines.whittaker.aspls(df_out[col], lam=2e6)[0]
            if (~np.isfinite(bkg_2)).sum() > 0:
                bkg_2 = pybaselines.polynomial.modpoly(df_out[col], range(len(df_out.index)))[0]

            if (~np.isfinite(bkg_2)).sum() > 0:
                  bkg_2 = np.zeros_like(bkg_2) + df_out[col].min()

            bkgs[col] = bkg_2
            df_out[col] = (df_out[col] - bkg_2).clip(0, np.inf)
        
    if image:
        im = Image.fromarray((df_out.values*255).astype(np.uint8))
        if size_224:
            im.save(DATA_PATH + 'features/' + flpth.split('.')[0] + '_224.jpg')
        else:
            im.save(DATA_PATH + 'features/' + flpth.split('.')[0] + '.jpg')
    else:
        df_out = (df_out.reset_index()
                        .melt(id_vars=['time_bin'], var_name='mass_bin', value_name='intensity')
                 )

        # Reshape so that each row is a single sample (to build feature matrix)
        df_out = df_out.pivot_table(
            columns=["mass_bin", "time_bin"], values=["intensity"]
        )

    return df_out

if __name__ == '__main__':
    df_all, df_train, compounds = load_metadata()

    # Keep track of processed features
    train_features_dict = {}
    test_features_dict = {}

    if not os.path.exists(DATA_PATH + 'features'):
        os.mkdir(DATA_PATH + 'features')

    for filepath in tqdm(df_all['features_path'].values):
        # Generate image with rounded mass and binned time
        _ = generate_features(filepath, image=True)

        # Generate 224x224 image
        _ = generate_features(filepath, image=True, size_224=True)

        # Generate features
        sample_id = filepath.split('/')[1].split('.')[0]
        df_out = generate_features(filepath)
        
        # Add to the corresponding dictionary
        if df_all.loc[df_all['features_path'] == filepath, 'split'].values[0] != 'test':
            train_features_dict[sample_id] = df_out
        else:
            test_features_dict[sample_id] = df_out

    # Post-process and export table of features

    train_features = pd.concat(train_features_dict, names=["sample_id", "dummy_index"]).reset_index(level="dummy_index", drop=True)
    train_features.T.reset_index().rename(columns=lambda c: str(c)).to_parquet(DATA_PATH + 'features/train_features.parquet')

    test_features = pd.concat(test_features_dict, names=["sample_id", "dummy_index"]).reset_index(level="dummy_index", drop=True)
    test_features.T.reset_index().rename(columns=lambda c: str(c)).to_parquet(DATA_PATH + 'features/test_features.parquet')