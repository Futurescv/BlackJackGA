import numpy as np
thresh = 0
a =np.random.randint(-1, 2, size=10)
a[ a > thresh] *= 5
print(a)