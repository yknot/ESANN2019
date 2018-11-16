"""Convert file into/out of SDV format and a couple other tasks"""
import argparse
import pickle
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression


def fix_ages(df):
    """fix the negative ages by making them the max 90"""
    age_mask = df.AGE < 0
    df.loc[age_mask, 'AGE'] = 90
    return df


def one_hot_encode(df):
    """convert all categorical variables into one hot encodings"""
    # drop id/date cols
    category_cols = [c for c in df.columns if df[c].dtype.name == 'object']
    df = pd.get_dummies(df, columns=category_cols, prefix=category_cols)
    return df


def impute_column(df, c):
    """impute the column c in dataframe df"""
    # get x and y
    y = df[c]
    x = df.drop(c, axis=1)

    # remove columns with in the values to impute
    x = x.loc[:, ~(x[y.isna()].isna().any())]

    # remove rows with na values in the training data
    na_mask = ~(x.isna().any(axis=1))
    y = y[na_mask]
    x = x[na_mask]

    # one hot encode the data
    x = one_hot_encode(x)

    # get mask for data to impute
    impute_mask = y.isna()
    # if y is continuous then use linear regression
    if y.dtype.name == 'float64':
        clf = LinearRegression()
    elif y.dtype.name == 'object':
        # Train KNN learner
        clf = KNeighborsClassifier(3, weights='distance')
        # le = LabelEncoder()
        # le.fit(df[col])
    else:
        raise ValueError
    trained_model = clf.fit(x[~impute_mask], y[~impute_mask])
    imputed_values = trained_model.predict(x[impute_mask])

    return imputed_values


def fix_na_values(df):
    """run of imputing columns with missing values"""
    ignored = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME']
    df_core = df.drop(ignored, axis=1)

    while df_core.isna().sum().sum():
        # get column with least amount of missing values
        cols_with_na = df_core.isna().sum()
        col = cols_with_na[cols_with_na > 0].idxmin()
        # impute that column
        df_core.loc[df_core[col].isna(), col] = impute_column(df_core, col)

    return pd.concat([df_core, df[ignored]], axis=1)


def categorical(col, limits=None):
    """convert a categorical column to continuous"""
    if limits:
        # reconstruct the distributions
        distributions = {}
        a = 0
        for b, cat in limits.items():
            mu, sigma = (a + b) / 2, (b - a) / 6
            distributions[cat] = truncnorm((a - mu) / sigma,
                                           (b - mu) / sigma,
                                           mu, sigma)
            a = b

        # convert values that don't exist in orig col to most common
        col = col.copy() # to lose copy warnings
        common = col.value_counts().index[0]
        for cat in col.unique():
            if cat not in distributions:
                col.loc[col == cat] = common

        return col.apply(lambda x: distributions[x].rvs()), None
    # get categories
    categories = (col.value_counts() / len(col)).sort_values(ascending=False)
    # get distributions to pull from
    distributions = {}
    limits = {}
    a = 0
    # for each category
    for cat, val in categories.items():
        # figure out the cutoff value
        b = a + val
        # create the distribution to sample from
        mu, sigma = (a + b) / 2, (b - a) / 6
        distributions[cat] = truncnorm((a - mu) / sigma,
                                       (b - mu) / sigma,
                                       mu, sigma)
        limits[b] = cat
        a = b

    # sample from the distributions and return that value
    return col.apply(lambda x: distributions[x].rvs()), limits


def numeric(col, min_max=None):
    """normalize a numeric column"""
    if min_max:
        return ((col - min_max[0]) / (min_max[1] - min_max[0])), None, None
    return ((col - min(col)) / (max(col) - min(col))), min(col), max(col)


def undo_categorical(col, lim):
    """convert a categorical column to continuous"""

    def cat_decode(x, limits):
        """decoder for categorical data"""
        for k, v in limits.items():
            if x < k:
                return v

    return col.apply(lambda x: cat_decode(x, lim))


def undo_numeric(col, min_col, max_col):
    """normalize a numeric column"""
    return ((max_col - min_col) * col) + min_col


def read_data(filename):
    """read in the file"""
    data = None
    if filename.endswith('.csv'):
        data = pd.read_csv(filename)
    elif filename.endswith('.npy'):
        data = pd.DataFrame(np.load(filename))

    # check if file can be read
    if data is None:
        raise ValueError

    return data


def encode(df, limits=None, min_max=None):
    """encode the data into SDV format"""
    # loop through every column
    if limits and min_max:
        already_exists = True
    else:
        limits = {}
        min_max = {}
        already_exists = False
    for c in df.columns:
        # if object or int
        if df[c].dtype.char == 'O' or df[c].dtype.char == 'l':
            if already_exists:
                df[c], _ = categorical(df[c], limits[c])
            else:
                df[c], lim = categorical(df[c])
                limits[c] = lim
        # if decimal
        elif df[c].dtype.char == 'd':
            if already_exists:
                df[c], _, _ = numeric(df[c], min_max[c])
            else:
                df[c], min_res, max_res = numeric(df[c])
                min_max[c] = (min_res, max_res)

    return df, limits, min_max


def decode(df_new, df_orig, limits, min_max):
    """decode the data from SDV format"""
    df_new = pd.DataFrame(df_new, columns=df_orig.columns)
    for c in df_new.columns:
        if c in limits:
            df_new[c] = undo_categorical(df_new[c], limits[c])
        else:
            df_new[c] = undo_numeric(df_new[c], *min_max[c])

    return df_new


def save_files(df, limits, min_max, prefix):
    """save the sdv file and decoders"""
    df.to_csv(f'{prefix}_sdv.csv', index=False)
    if limits:
        pickle.dump(limits, open(f'{prefix}.limits', 'wb'))
    if min_max:
        pickle.dump(min_max, open(f'{prefix}.min_max', 'wb'))


def read_decoders(prefix, npy_file):
    """read the decoder files"""
    limits = pickle.load(open(f'{prefix}.limits', 'rb'))
    min_max = pickle.load(open(f'{prefix}.min_max', 'rb'))
    if npy_file.endswith('.csv'):
        npy = pd.read_csv(npy_file)
    elif npy_file.endswith('.npy'):
        npy = np.load(npy_file)
    else:
        npy = None

    return limits, min_max, npy


def parse_arguments(parser):
    """parser for arguments and options"""
    parser.add_argument('data_file', type=str, metavar='<data_file>',
                        help='The data to transform')
    subparsers = parser.add_subparsers(dest='op')
    subparsers.add_parser('encode')

    parser_decode = subparsers.add_parser('decode')
    parser_decode.add_argument('npy_file', type=str, metavar='<npy_file>',
                               help='numpy file to decode')
    parser.add_argument('--fix_ages', dest='ages', action='store_const',
                        const=True, default=False, help='fix negative ages')
    parser.add_argument('--impute', dest='impute', action='store_const',
                        const=True, default=False,
                        help='impute missing values')
    parser.add_argument('--encoder_file')#, type=str, metavar='<encoder_file>',
                        #help='use encoder file')

    return parser.parse_args()


if __name__ == '__main__':
    # read in arguments
    args = parse_arguments(argparse.ArgumentParser())
    # open and read the data file
    df_raw = read_data(args.data_file)

    if args.op == 'encode':
        if args.ages:
            # fix negative ages
            df_raw = fix_ages(df_raw)

        if args.impute:
            # fix the NA values
            df_raw = fix_na_values(df_raw)
            assert df_raw.isna().sum().sum() == 0

        if args.encoder_file:
            lims, mms, _ = read_decoders(args.encoder_file, '')
            df_converted, _, _ = encode(df_raw, lims, mms)
            save_files(df_converted, '', '', args.data_file[:-4])
        else:
            df_converted, lims, mm = encode(df_raw)
            save_files(df_converted, lims, mm, args.data_file[:-4])
    elif args.op == 'decode':
        lims, mm, npy_new = read_decoders(args.data_file[:-4], args.npy_file)
        df_converted = decode(npy_new, df_raw, lims, mm)
        # save decoded
        df_converted.to_csv(args.data_file[:-4] + '_synthetic.csv',
                            index=False)
