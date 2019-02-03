import numpy as np

from util import proj as dt




def main(rankings_fn, ppmi_fn,  fn):
    """
    discrete_labels_fn = "Rankings/films100N0.6H75L1P1.discrete"
    ppmi_fn = "Journal Paper Data/Term Frequency Vectors/class-all"
    phrases_fn = "SVMResults/films100.names"
    phrases_to_check_fn = ""#"RuleType/top1ksorted.txt"
    fn = "films 100, 75 L1"
    """
    getNDCG(rankings_fn, fn)
