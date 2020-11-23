import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

epoch_stop=2500

plt.rcParams["font.family"] = "Times New Roman"


def plot_graph(naul, basel, fn):
    t = np.arange(epoch_stop-10)
    basetemp = basel[(-epoch_stop+10):]
    basetemp[0] = 0.5
    nautemp = naul[(-epoch_stop+10):]
    nautemp[0] = 0.5
    plt.plot(t, basetemp, 'r', label='Baseline')
    plt.plot(t, nautemp, 'b', label='NAU')
    plt.ylim(0, 0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.savefig('1024_%s.pdf' %fn, transparent=True, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.clf()

def tsne_graph(weights, fn):
    pca = PCA(n_components=40)
    pca_res = pca.fit_transform(weights)
    print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))
    tsne = TSNE(n_components=2, verbose=1)
    tsne_res = tsne.fit_transform(pca_res)
    col = np.arange(0, 256)
    plt.scatter(tsne_res[:, 0], tsne_res[:, 1], c=col, alpha=0.3, cmap='viridis')
    plt.colorbar()
    plt.savefig('tsne_%s' %fn, transparent=True, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.clf()
