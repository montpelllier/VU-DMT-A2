import numpy as np
import pandas as pd


def dcg_at_k(rank, rel, k):
    rank += 1
    dg = rel[:k] / np.log2(rank[:k])
    dcg = np.sum(dg)
    return dcg


def iDCG_at_k(rel, k):
    rel = sorted(rel, reverse=True)
    rel = pd.DataFrame(rel)
    rank = [i for i in range(1, k + 1)]
    rank = pd.DataFrame(rank)
    return dcg_at_k(rank, rel, k)


def nDCG_at_k(rank, labels, k):
    dcg = dcg_at_k(rank, labels, k)
    idcg = iDCG_at_k(labels, k)
    return dcg / idcg
