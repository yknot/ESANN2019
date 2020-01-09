"""Convert file into/out of SDV format and a couple other tasks"""
import argparse
import json
import numpy as np
import numpy.random as rnd
import pandas as pd
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression


def fix_ages(df):
    """fix the negative ages by making them the max 90"""
    age_mask = df.AGE < 0
    df.loc[age_mask, 'AGE'] = 90
    return df


def boost(df, lim, cutoff=0.1):
    """boost samples below the cutoff to at least the cutoff"""
    # find the columns to boost
    to_boost = {
        k: v
        for k, v in lim.items()
        if set(v.values()) == set((0, 1)) and min(v.keys()) > (1 - cutoff)
    }

    tot = round(len(df) * cutoff)

    # boost by the number needed
    for k, v in to_boost.items():
        sub = df[df[k] > min(v.keys())]
        up = tot - sub.shape[0]
        # if we have already reach the cutoff don't bother
        if up > 0:
            df = df.append(sub.sample(up, replace=True))

    # shuffle and reset index
    df = df.sample(frac=1).reset_index(drop=True)
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
            b = float(b)
            mu, sigma = (a + b) / 2, (b - a) / 6
            distributions[cat] = stats.truncnorm((a - mu) / sigma,
                                                 (b - mu) / sigma, mu, sigma)
            a = b

        # convert values that don't exist in orig col to most common
        col = col.copy()  # to lose copy warnings
        common = col.value_counts().index[0]
        for cat in col.unique():
            if cat not in distributions:
                col.loc[col == cat] = common

        return col.apply(lambda x: distributions[x].rvs()), None
    # get categories, ensures sort by value and then name to tiebreak
    series = col.value_counts(normalize=True)
    tmp = pd.DataFrame({'names': series.index, 'pcts': series.values})
    tmp = tmp.sort_values(['pcts', 'names'], ascending=[False, True])
    categories = pd.Series(tmp.pcts.values, tmp.names.values)

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
        distributions[cat] = stats.truncnorm((a - mu) / sigma,
                                             (b - mu) / sigma, mu, sigma)
        limits[b] = cat
        a = b

    # sample from the distributions and return that value
    return col.apply(lambda x: distributions[x].rvs()), limits


def ordinal(col, limits=None):
    """convert a ordinal column to continuous"""
    if limits:
        # reconstruct the distributions
        distributions = {}
        a = 0
        for b, cat in limits.items():
            b = float(b)
            mu, sigma = (a + b) / 2, (b - a) / 6
            distributions[cat] = stats.truncnorm((a - mu) / sigma,
                                                 (b - mu) / sigma, mu, sigma)
            a = b

        # # convert values that don't exist in orig col to most common
        # col = col.copy()  # to lose copy warnings
        # common = col.value_counts().index[0]
        # for cat in col.unique():
        #     if cat not in distributions:
        #         col.loc[col == cat] = common

        return col.apply(lambda x: distributions[x].rvs()), None
    # get categories, ensures sort by value and then name to tiebreak
    categories = col.value_counts()
    # find missing categories and fill with 0s
    for i in range(categories.keys().min(), categories.keys().max() + 1):
        if i not in categories.keys():
            categories[i] = 0
    # sort by index to get in order
    categories = categories.sort_index()

    # additive smoothing for 0 counts
    alpha = 1
    new_vals = (categories.values + alpha) / (len(col) +
                                              (alpha * len(categories)))

    # create new categories
    categories = pd.Series(new_vals, index=categories.index)

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
        distributions[cat] = stats.truncnorm((a - mu) / sigma,
                                             (b - mu) / sigma, mu, sigma)
        limits[b] = cat
        a = b

    # sample from the distributions and return that value
    return col.apply(lambda x: distributions[x].rvs()), limits


def truncated_beta(alpha, beta, low, high):
    """truncated beta distribution with params alpha and beta, and limits low and high"""
    nrm = stats.beta.cdf(high, alpha, beta) - stats.beta.cdf(low, alpha, beta)

    low_cdf = stats.beta.cdf(low, alpha, beta)

    while True:
        yr = rnd.random(1) * nrm + low_cdf
        xr = stats.beta.ppf(yr, alpha, beta)
        yield xr[0]


def binary(col, limits=None):
    """convert a binary column to continuous"""
    if limits:
        # reconstruct the distributions
        distributions = {}

        zeros = min(limits.keys())

        # case of all zeros and all ones
        if zeros == 1:
            return col.apply(lambda x: 0), {1.0: 0}
        if zeros == 0:
            return col.apply(lambda x: 1), {1.0: 1}

        alpha = (zeros) * 100
        beta = ((len(col) - (zeros * len(col))) / len(col)) * 100

        distributions[0] = truncated_beta(alpha, beta, 0, zeros)

        distributions[1] = truncated_beta(alpha, beta, zeros, 1)

        # convert values that don't exist in orig col to most common
        col = col.copy()  # to lose copy warnings

        return col.apply(lambda x: next(distributions[x])), None

    zeros = (col == 0).sum() / len(col)
    alpha = zeros * 100
    beta = ((len(col) - (col == 0).sum()) / len(col)) * 100

    # case of all zeros and all ones
    if zeros == 1:
        return col.apply(lambda x: 0), {1.0: 0}
    if zeros == 0:
        return col.apply(lambda x: 1), {1.0: 1}
    # get distributions to pull from
    distributions = {}
    limits = {}

    distributions[0] = truncated_beta(alpha, beta, 0, zeros)
    limits[zeros] = 0

    distributions[1] = truncated_beta(alpha, beta, zeros, 1)
    limits[1] = 1

    # sample from the distributions and return that value
    return col.apply(lambda x: next(distributions[x])), limits


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
            if x <= float(k):
                return v

    return col.apply(lambda x: cat_decode(x, lim))


def undo_numeric(col, min_col, max_col, discrete=None):
    """normalize a numeric column"""
    if discrete:
        return (((max_col - min_col) * col) + min_col).round().astype('int')
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


def encode(df, limits=None, min_max=None, beta=False):
    """encode the data into SDV format"""
    # loop through every column
    if limits and min_max:
        already_exists = True
    else:
        limits = {}
        min_max = {}
        already_exists = False
    for c in df.columns:
        # if object
        if df[c].dtype.char == 'O':
            if already_exists:
                df[c], _ = categorical(df[c], limits[c])
            else:
                df[c], lim = categorical(df[c])
                limits[c] = lim
        # if int
        elif df[c].dtype.char == 'l':
            # if binary
            if set(df[c].unique()).issubset(set((0, 1))):
                if already_exists:
                    if beta:
                        df[c], _ = binary(df[c], limits[c])
                    else:
                        df[c], _ = categorical(df[c], limits[c])
                else:
                    if beta:
                        df[c], lim = binary(df[c])
                    else:
                        df[c], lim = categorical(df[c])
                    limits[c] = lim
            # else ordinal
            else:
                if already_exists:
                    df[c], _ = ordinal(df[c], limits[c])
                else:
                    df[c], lim = ordinal(df[c])
                    limits[c] = lim
        # if boolean
        elif df[c].dtype.char == '?':
            if already_exists:
                if beta:
                    df[c], _ = binary(df[c], limits[c])
                else:
                    df[c], _ = categorical(df[c], limits[c])
            else:
                if beta:
                    df[c], lim = binary(df[c])
                else:
                    df[c], lim = categorical(df[c])
                limits[c] = lim

        # if decimal
        elif df[c].dtype.char == 'd':
            if already_exists:
                df[c], _, _ = numeric(df[c], min_max[c])
            else:
                df[c], min_res, max_res = numeric(df[c])
                min_max[c] = (min_res, max_res, 0)

    return df, limits, min_max


def decode(df_new, df_orig_cols, limits, min_max):
    """decode the data from SDV format"""
    df_new = pd.DataFrame(df_new, columns=df_orig_cols)
    for c in df_new.columns:
        if c in limits:
            df_new[c] = undo_categorical(df_new[c], limits[c])
        else:
            df_new[c] = undo_numeric(df_new[c], *min_max[c])

    return df_new


def save_files(df, prefix, limits=None, min_max=None, cols=False):
    """save the sdv file and decoders"""
    df.to_csv(f'{prefix}_sdv.csv', index=False)
    if cols:
        json.dump(df.columns.tolist(), open(f"{prefix}.cols", "w"))
    if limits:
        json.dump(limits, open(f"{prefix}.limits", "w"))
    if min_max:
        json.dump(min_max, open(f"{prefix}.min_max", "w"))


def read_decoders(prefix, npy_file):
    """read the decoder files"""
    limits = json.load(open(f"{prefix}.limits"))
    try:
        min_max = json.load(open(f"{prefix}.min_max"))
    except FileNotFoundError:
        min_max = None
    try:
        cols = json.load(open(f"{prefix}.cols"))
    except FileNotFoundError:
        cols = None
    if npy_file.endswith('.csv'):
        npy = pd.read_csv(npy_file)
    elif npy_file.endswith('.npy'):
        npy = np.load(npy_file)
    else:
        npy = None

    return limits, min_max, cols, npy


def parse_arguments(parser):
    """parser for arguments and options"""
    parser.add_argument(
        'data_file',
        type=str,
        metavar='<data_file>',
        help='The data to transform')
    subparsers = parser.add_subparsers(dest='op')
    subparsers.add_parser('encode')
    subparsers.add_parser('test')

    parser_decode = subparsers.add_parser('decode')
    parser_decode.add_argument(
        'npy_file',
        type=str,
        metavar='<npy_file>',
        help='numpy file to decode')
    parser.add_argument(
        '--fix_ages',
        dest='ages',
        action='store_const',
        const=True,
        default=False,
        help='fix negative ages')
    parser.add_argument(
        '--impute',
        dest='impute',
        action='store_const',
        const=True,
        default=False,
        help='impute missing values')
    parser.add_argument(
        '--beta',
        dest='beta',
        action='store_const',
        const=True,
        default=False,
        help='use beta distribution')
    parser.add_argument(
        '--boost',
        dest='boost',
        action='store_const',
        const=True,
        default=False,
        help='boost 1% and lower samples')
    parser.add_argument(
        '--encoder_file')  #, type=str, metavar='<encoder_file>',
    #help='use encoder file')

    return parser.parse_args()


if __name__ == '__main__':
    # read in arguments
    args = parse_arguments(argparse.ArgumentParser())

    if args.op == 'test':
        # open and read the data file
        df_raw = read_data(args.data_file)

        df_converted, lims, mm = encode(df_raw, beta=args.beta)
        df_converted = decode(df_converted, df_raw, lims, mm)
        assert (df_converted == df_raw).all().all()
        print('Test Passed')

    elif args.op == 'encode':
        # open and read the data file
        df_raw = read_data(args.data_file)

        if args.ages:
            # fix negative ages
            df_raw = fix_ages(df_raw)

        if args.impute:
            # fix the NA values
            df_raw = fix_na_values(df_raw)
            assert df_raw.isna().sum().sum() == 0

        if args.encoder_file:
            lims, mms, _, _ = read_decoders(args.encoder_file, '')
            df_converted, _, _ = encode(df_raw, lims, mms, beta=args.beta)
            save_files(df_converted, args.data_file[:-4])

        else:
            df_converted, lims, mm = encode(df_raw, beta=args.beta)
            save_files(df_converted, args.data_file[:-4], lims, mm, True)

    elif args.op == 'decode':
        lims, mm, cols, npy_new = read_decoders(args.data_file[:-4],
                                                args.npy_file)
        if not cols:
            # open and read the data file
            df_raw = read_data(args.data_file)
            cols = df_raw.columns

        df_converted = decode(np.clip(npy_new, 0, 1), cols, lims, mm)
        # save decoded
        df_converted.to_csv(
            args.data_file[:-4] + '_synthetic.csv', index=False)
