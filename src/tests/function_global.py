from score.classify import MultiClassScore
import numpy as np
from util.save_load import SaveLoad

def make_score_dict(f1s=None, accs=None, aurocs=None, recalls=None, precs=None, kappas=None):
    score = MultiClassScore([], [],  None, None, None)
    if f1s != None:
        score.f1s.value = f1s
        score.recalls.value = recalls
        score.precs.value = precs
    if accs != None:
        score.accs.value = accs
    if aurocs != None:
        score.aurocs.value = aurocs
    if kappas != None:
        score.kappas.value = kappas
    score_dict = score.get()
    return score_dict


def test_average_scores():
    pred = [0,0,0,0]
    true = [0,0,0,1]
    save = SaveLoad(rewrite=False, no_save=True)
    zero_score = MultiClassScore(true, pred, pred, "", "", save)
    zero_score.process_and_save()
    zero_score.print()
    exit()
    # 100 features 50 classes
    zero_array = np.zeros(shape=(100,50))
    one_array = np.ones(shape=(100,50))
    random_array = np.random.rand(100, 50)
    save = SaveLoad(rewrite=False, no_save=True)
    zero_score = MultiClassScore(zero_array, zero_array, zero_array, "", "", save)
    zero_score.process_and_save()
    one_score = MultiClassScore(one_array, zero_array, zero_array, "", "", save)
    one_score.process_and_save()
    random_score = MultiClassScore(random_array, zero_array, zero_array, "", "", save)
    random_score.process_and_save()

if __name__ == '__main__':
    test_average_scores()