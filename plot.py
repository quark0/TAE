from scipy.io import loadmat
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

data = loadmat('/tmp/params.mat')

cmap = 'coolwarm'

plt.figure(2)
heatmap_U = plt.pcolor(normalize(data['U'], norm='l1', axis=1), cmap=cmap)

plt.show()
