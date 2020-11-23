import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

epoch_stop=2000

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)

plt.rc('axes', labelsize=24)


def plot_graph(naul, basel, fn):
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    
    t = np.arange(epoch_stop-10)
    basetemp = basel[(-epoch_stop+10):]
    basetemp[0] = 0.5
    nautemp = naul[(-epoch_stop+10):]
    nautemp[0] = 0.5
    ax1.plot(t, basetemp, 'C3', label='Baseline')
    ax1.plot(t, nautemp, 'C0', label='NAU')
    #ax1.ylim(0, 0.3)
    #ax1.xlabel('Epochs')
    #ax1.ylabel('MSE Loss')
    #ax1.gca().set_axis_off()
    #ax1.show()
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(*ax1.get_legend_handles_labels(), frameon=False, loc='center', ncol=10)
    fig = legend.figure
    fig.canvas.draw()
    #bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('legend.pdf', bbox_inches='tight', pad_inches=0, dpi=200, transparent=True)
    
    #plt.savefig('1024_%s.pdf' %fn, transparent=True, bbox_inches='tight', pad_inches=0, dpi=200)
    #plt.clf()
    """
    fig_legend = plt.figure()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.legend((ax1, ax2), sys, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fancybox=True, shadow=True, frameon=False, handlelength=4, fontsize=25)
    fig_legend.legend((ax1, ax2), sys, loc='upper center', ncol=3, frameon=False, handlelength=3, fontsize=18)
    fig_legend.savefig('legend.pdf', transparent=True, bbox_inches='tight', pad_inches=0, dpi=200)
    """
    

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


"""
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

mpl.rc('font', family='Times New Roman')

sys = ['Debin', 'StateFormer']
x = np.arange(3) + 1

Debin = [0.606, 0.661, 0.461] # 0.576
StateFormer = [0.98, 0.98, 0.99] # 0.983

fig_legend = plt.figure()
fig, ax = plt.subplots(figsize=(10, 4))

patterns = ["+", "x"]
ax1 = ax.bar(x - .15, Debin, width=0.3, hatch=patterns[0], align='center', alpha=0.7, color='C0',
edgecolor='black')
ax2 = ax.bar(x + .15, StateFormer, width=0.3, hatch=patterns[1], align='center', alpha=0.7, color='C3',
edgecolor='black')

# ax.tick_params(axis='both', which='major', labelsize=18)

plt.xlim([0.5, len(x) + 0.5])
ax.set_xticks(x)
ax.set_xticklabels(['bcf', 'cff', 'sub'], fontsize=30)

ax.legend((ax1, ax2), sys, loc='upper center', bbox_to_anchor=(0.5, 1.2),
ncol=2, fancybox=True, shadow=True, frameon=False, handlelength=4, fontsize=25)

plt.ylabel('F1 score', fontsize=20)

# plt.show()
plt.savefig('figs/debin-obf.pdf', transparent=True, bbox_inches='tight', pad_inches=0, dpi=200)

# fig_legend.legend((ax1, ax2), sys, loc='upper center', ncol=3, frameon=False, handlelength=3,
# fontsize=18)
# fig_legend.savefig(f'figs/debin-obf-legend.pdf', transparent=True, bbox_inches='tight', pad_inches=0, dpi=200)

"""
