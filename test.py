import numpy as np
from pprint import pprint
from typing import Tuple

mat = ((1.0, 2.0),
       (3.0, 4.0),
       (1.0, 2.0),
       (3.0, 4.0))

mat = np.matrix(mat)

b = np.matrix((True, False, False, False)).T

b = np.tile(b, 2)
pprint(b)


out = np.multiply(mat, b)

pprint(out)