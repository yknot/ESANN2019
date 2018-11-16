import pandas as pd

from sklearn.neighbors import KernelDensity

# start with the average distance between points
start_bandwidth = 0.28336393686529654

input_dir = 'mimic'
output_dir = 'mimic_parzen'

train = pd.read_csv(f'{input_dir}/mimic_train_sdv.csv')

for i in range(11):
    # create kernel density object and fit
    kd = KernelDensity(bandwidth=start_bandwidth)
    kd.fit(train)

    for j in range(1, 8):
        # generate data and write out
        gen = pd.DataFrame(kd.sample(13463), columns=train.columns)
        gen.to_csv(
            f'{output_dir}_iter_{i}/mimic_parzen_iter_{i}_synthetic_{j}.csv',
            index=False)

    # divide by two for the next iteration
    start_bandwidth /= 2
