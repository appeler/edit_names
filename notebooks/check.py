import Levenshtein as lv
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
from tqdm import tqdm


def check_k(test_df, corpus_df, k):
    final_pred = []
    for idx, row in tqdm(test_df.iterrows()):
        indices = cos_sim(row, corpus_df, filt=.6)
        if (len(indices) == 0):
            indices = cos_sim(row, corpus_df, filt=.1)

        filtered_corpus_df = corpus_df.iloc[indices]

        lev_dist = calc_leven(row['name_last'], filtered_corpus_df)

        values = np.array(list(lev_dist.values()))
        keys = np.array(list(lev_dist.keys()))

        if (k < values.shape[0]):
            filt_values = np.argpartition(values, k)
        elif (values.shape[0] == 0):
            indices = cos_sim(row, corpus_df, filt=.1)
            filtered_corpus_df = corpus_df.iloc[indices]

            lev_dist = calc_leven(row['name_last'], filtered_corpus_df)
            values = np.array(list(lev_dist.values()))
            keys = np.array(list(lev_dist.keys()))
            filt_values = np.argpartition(values, k)
        else:
            filt_values = values.shape[0] - 1

        if (isinstance(filt_values, np.ndarray)):
            max_value = np.max(values[filt_values[:k]])
        else:
            max_value = np.max(values[filt_values])

        mask = (values <= max_value) & (values > 0)
        mask_idx = np.argwhere(mask).reshape(-1)
        df_idx = keys[mask_idx]

        total_sum = (corpus_df.iloc[df_idx]['total_n'].sum())
        pred_white = (corpus_df.iloc[df_idx]['nh_white'] *
                      corpus_df.iloc[df_idx]['total_n']).sum() / total_sum
        pred_black = (corpus_df.iloc[df_idx]['nh_black'] *
                      corpus_df.iloc[df_idx]['total_n']).sum() / total_sum
        pred_hispanic = (corpus_df.iloc[df_idx]['hispanic'] *
                         corpus_df.iloc[df_idx]['total_n']).sum() / total_sum
        pred_asian = (corpus_df.iloc[df_idx]['asian'] *
                      corpus_df.iloc[df_idx]['total_n']).sum() / total_sum
        predictions = [pred_asian, pred_hispanic, pred_black, pred_white]

        final_pred.append(predictions.index(max(predictions)))

    test_df['pred_race'] = final_pred

    return classification_report(test_df['true_race'], test_df['pred_race'])


def cos_sim(row_data, corpus_df, filt=0.6):
    orig_vector = np.array(row_data['tfidf']).reshape(1, -1)
    corp_vector = np.array([x for x in corpus_df['tfidf']])
    cossim = cosine_similarity(orig_vector, corp_vector)
    return np.argwhere(cossim >= filt).reshape(-1)


def calc_leven(orig_string, filt_df):
    lev_dist = {}
    for idx, row in filt_df.iterrows():
        lev = lv.distance(orig_string, row['name_last'])
        lev_dist[idx] = lev
    return lev_dist
