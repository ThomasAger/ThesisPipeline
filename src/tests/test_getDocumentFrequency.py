from util.classify import getDocumentFrequency
import numpy as np
from hypothesis.strategies import composite
from hypothesis.strategies import lists

from io.io import isInt
from util.classify import getDocumentFrequency


@composite
def test_getDocumentFrequency(draw, elements=lists(max_size=50)):
    x = draw(lists(elements, max_size=100))
    assert len(getDocumentFrequency(x) == len(x))
    assert len(getDocumentFrequency(x) > len(x[0]))
    assert isInt(getDocumentFrequency(x)[0])
    assert len(np.nonzero(getDocumentFrequency(x))) == 0