import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


X = pd.read_csv('mimic_train_sdv.csv')

gm = GaussianMixture()
gm.fit(X)

for i in range(1, 21):
    print(i)
    gen = pd.DataFrame(gm.sample(X.shape[0])[0], columns=X.columns)
    gen.to_csv('mimic_mg_synthetic_' + str(i) + '.csv', index=False)