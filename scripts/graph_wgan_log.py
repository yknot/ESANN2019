import sys
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

pylab.rcParams['figure.figsize'] = 8, 8
try:
    log = pkl.load(open('log.pkl', 'rb'))
except:
    if len(sys.argv) > 1:
        log = pkl.load(open(sys.argv[1], 'rb'))
    else:
        print('Usage: python graph_wgan_log.py <log_file>')

losses = ['test_loss', 'gen_loss', 'disc_loss', 'time']
titles = [
    'Test Loss', 'Generator Loss', 'Discriminator Loss', 'Time per Epoch'
]
labels = ['Epochs (in thousands)', 'Epochs', 'Epochs', 'Epochs']

for i, loss in enumerate(losses):
    if isinstance(log[loss][0], list):
        new_df = pd.DataFrame({titles[i]: [v[-1] for v in log[loss][100:]]})
    else:
        new_df = pd.DataFrame({titles[i]: log[loss]})
    sns.lineplot(data=new_df, dashes=False, palette="hls")
    plt.title(titles[i])
    plt.xlabel('Epochs (in thousands)')
    plt.savefig(loss + '.pdf')
    plt.close()
