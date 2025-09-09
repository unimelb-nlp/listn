# save to file
import json
import numpy as np
from scipy.spatial.distance import cdist
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from multiprocessing import Pool
from collections import defaultdict
from tqdm import tqdm
import re
from scipy import sparse
from sklearn import preprocessing


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def cosine_similarity(x,y):
    return 1-cdist(x, y, 'cosine')/2


def find_related(ind, W):
    sim = cosine_similarity(W[ind].reshape(1, -1),W)[0]
    return sim


def parallelize(data, func, num_pool, num_chunks=None):
    if num_chunks:
        data = chunker(data, num_chunks)
    pool = Pool(num_pool)
    res = pool.map(func, data)
    pool.close()
    pool.join()
    return res


def vectorize(texts):
    ngram_vectorizer = CountVectorizer(analyzer='word', tokenizer=word_tokenize,
                                       lowercase=True)
    mat = ngram_vectorizer.fit_transform(texts)
    return (ngram_vectorizer.get_feature_names_out(),
            list(np.asarray(mat.sum(axis=0))))

def chunker(seq, size):
    return [seq[i::size] for i in range(size)]


def combine_dicts(list_of_dicts):
    obs = defaultdict(list)

    for sub in list_of_dicts:
        for k, v in sub.items():
            obs[k].extend(v)

    return obs

def read_posts(path, fields):
    texts = []

    with open(path, 'r') as f:
        for l in tqdm(f.readlines()):
            a = json.loads(l)
            texts.append({field:a[field] for field in fields})

    return texts

def remove_URL(text):
    return re.sub('(https?:\/\/|www.)\S+', '', text)


def get_word_probs(counts):
    total = np.array(counts).sum()
    probs = np.array(counts) / total

    return probs


def get_pmi(freq_mat, p_word):
    p_word_inv = sparse.csr_matrix(1 / p_word)
    p_word_user = preprocessing.normalize(freq_mat, norm='l1', axis=1)  # N x M
    pmi = p_word_inv.multiply(p_word_user)
    pmi.data = np.log2(pmi.data)
    return pmi


def get_npmi(freq_mat, p_word):
    total_word_freq = freq_mat.sum(axis=0)  # M x 1
    total_words = total_word_freq.sum()
    # p_word = total_word_freq / total_words # M x 1

    total_user_freq = freq_mat.sum(axis=1)  # N x 1
    p_user = total_user_freq / total_words  # N x 1

    joint_word_user = freq_mat / total_words  # N x M

    npmi = np.log2(p_user * p_word) / np.log2(joint_word_user) - 1
    return npmi


def load_sparse_matrix(path):
    return sparse.load_npz(path)
