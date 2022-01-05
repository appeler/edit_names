import Levenshtein as lv
import multiprocessing as mp
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
from tqdm import tqdm


def predict_race(arg):
    # reading the tuple passed on by the calling function
    idx, row_data, test_df, corpus_df, corp_vector, k, filt = arg

    # resizing the tf-idf (1, m) & corpus vectors to be (n, m)
    #  n = number of samples
    #  m = number of dimentions
    orig_vector = np.array(row_data['tfidf']).reshape(1, -1)
    # corp_vector = np.array([x for x in corpus_df['tfidf']])

    # calculating the cosine similarity beteween the name vector
    #   and the corpus vectors.  Then filtering for only values
    #   that are greater that what was passed on
    cossim = cosine_similarity(orig_vector, corp_vector)
    filt_result = np.argwhere(cossim >= filt).reshape(-1)

    # if we don't get any matches on cosine similarity >= "value"
    #    we open up the critiria to 0.1 to get something
    if (len(filt_result) == 0):
        # this is to handle if we still are not getting anything
        #  after opening up cosine similarity.  Just return a 0
        #  which right now is "Asian"
        if (filt == 0.1):
            return 0
        else:
            filt_result = predict_race(
                (idx, row_data, test_df, corpus_df, corp_vector, k, .1))

    # filtering the corpus dataframe to only inclue the items
    #   that met the cosine similarity filter
    filtered_corpus_df = corpus_df.iloc[filt_result]

    # calculate the levenshtein distance between our vector
    #   and the filtered corpus vectors.
    # Levenshtein is an expensive operation so we don't
    #   want to calculate it for every name in the corpus
    lev_dist = calc_leven(row_data['name_last'],
                          filtered_corpus_df['name_last'])

    # The calc_leven function returns a dictionary
    #  we seperate the keys from the values into arrays
    #  that we can use which names are the most similar
    #  i.e. smallest levenstein distance
    values = np.array(list(lev_dist.values()))
    keys = np.array(list(lev_dist.keys()))

    if (k < values.shape[0]):
        # This is when k is smaller than the size of the
        #   values array, we can partition it by the smallest
        #   k values
        filt_values = np.argpartition(values, k)
        max_value = np.max(values[filt_values[:k]])
    else:
        # Otherwise whatever the filt_value are will be the
        #   k nearest neighbors to our string
        filt_values = values.shape[0] - 1
        max_value = np.max(values[filt_values])

    # Determining which indexes from teh corpus need to be considered
    #   for k nearest neighbors distance wise.
    mask = (values <= max_value) & (values > 0)
    mask_idx = np.argwhere(mask).reshape(-1)
    df_idx = keys[mask_idx]

    # Calculating the probability for each race
    total_sum = (corpus_df.iloc[df_idx]['total_n'].sum())
    pred_white = (corpus_df.iloc[df_idx]['nh_white'] *
                  corpus_df.iloc[df_idx]['total_n']).sum() / total_sum
    pred_black = (corpus_df.iloc[df_idx]['nh_black'] *
                  corpus_df.iloc[df_idx]['total_n']).sum() / total_sum
    pred_hispanic = (corpus_df.iloc[df_idx]['hispanic'] *
                     corpus_df.iloc[df_idx]['total_n']).sum() / total_sum
    pred_asian = (corpus_df.iloc[df_idx]['asian'] *
                  corpus_df.iloc[df_idx]['total_n']).sum() / total_sum

    # Creating a list of probilities so we can just get the max index
    predictions = [pred_asian, pred_hispanic, pred_black, pred_white]

    return predictions.index(max(predictions))


def calc_leven(orig_string, filt_df):
    lev_dist = {}

    # determing if levenshtein was passed a dataframe or a string
    #  if its just a string then we return a dictionary with index 0
    if not (isinstance(filt_df, str)):
        for idx, row in filt_df.iteritems():
            lev = lv.distance(orig_string, row)
            lev_dist[idx] = lev
    else:
        lev = lv.distance(orig_string, filt_df)
        lev_dist[0] = lev
    return lev_dist


def calc_prop(row):
    total = row['total_n']
    values = [(i/total) for i in row]
    return pd.Series(values)


def get_race_idx(val, races):
    race_idx = races.index(val)
    return race_idx


def find_ngrams(text, n):
    a = zip(*[text[i:] for i in range(n)])
    wi = []
    for i in a:
        w = ''.join(i)
        try:
            idx = words_list.index(w)
        except:
            idx = 0
        wi.append(idx)
    return wi


def check_k(test_df, corpus_df, k, filt):
    results = []

    num_cpu = mp.cpu_count()
    pool = mp.Pool(processes=(num_cpu))

    # creating the corpus vector of tf-idf once and then passing it along
    #   to other methods as required
    corp_vector = np.array([x for x in corpus_df['tfidf']])

    # Multi-processing
    r = pool.map(predict_race, [(idx, row, test_df, corpus_df, corp_vector, k, filt)
                                for idx, row in test_df.iterrows()])
    results.append(r)

    # Cleaning up the multi-processes
    pool.close()
    pool.join()

    return results
