"""Quality-control metrics of data/dev_eval_500x/qc_rule.txt: does a low predicted
response flag a sensor that will not respond at all (rel_resp_median < 0.01)?"""
import numpy as np
from scipy.stats import mannwhitneyu

NONRESP_LOG10 = -2.0          # rel_resp_median < 0.01
RECALL_TARGET = 0.80


def auc(score, nonresp):
    """AUC for flagging non-responders by a LOW score."""
    pos, neg = score[nonresp], score[~nonresp]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(mannwhitneyu(neg, pos).statistic / (len(pos) * len(neg)))


def auc_perm_p(score, nonresp, n=10000, seed=0):
    a = auc(score, nonresp)
    rng = np.random.default_rng(seed)
    null = np.array([auc(score, rng.permutation(nonresp)) for _ in range(n)])
    return a, float((1 + (null >= a).sum()) / (1 + n))


def cutoff(dev_score, dev_nonresp):
    """Highest cutoff (flag if score <= cutoff) whose recall on dev non-responders is >= RECALL_TARGET."""
    s = np.sort(dev_score[dev_nonresp])
    return float(s[int(np.ceil(RECALL_TARGET * len(s))) - 1])


def qc_report(dev_score, dev_log_y, test_score, test_log_y):
    dn, tn = dev_log_y < NONRESP_LOG10, test_log_y < NONRESP_LOG10
    c = cutoff(dev_score, dn)
    flag = test_score <= c
    a, p = auc_perm_p(test_score, tn)
    return dict(qc_dev_auc=round(auc(dev_score, dn), 4), qc_cutoff=round(c, 4),
                qc_test_nonresp=int(tn.sum()), qc_test_auc=round(a, 4), qc_test_auc_p=round(p, 4),
                qc_test_recall=round(float(flag[tn].mean()) if tn.any() else np.nan, 4),
                qc_test_specificity=round(float((~flag[~tn]).mean()) if (~tn).any() else np.nan, 4))
