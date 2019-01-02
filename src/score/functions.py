import math

def get_f1_score(prec, recall):
    if prec == 0 and recall == 0:
        return 0.0
    f1 = 2 * ((prec * recall) / (prec + recall))
    if math.isnan(f1):
        f1 = 0.0
    return f1