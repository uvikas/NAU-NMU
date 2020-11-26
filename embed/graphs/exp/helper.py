import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib as mpl

epoch_stop=2500

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["legend.handlelength"] = 4.0

plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

plt.rc('axes', labelsize=14)


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
    legend = ax2.legend(*ax1.get_legend_handles_labels(), frameon=False, loc='center', ncol=10, )
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
    plt.clf()
    
    pca = PCA(n_components=40)
    pca_res = pca.fit_transform(weights)
    print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))
    tsne = TSNE(n_components=2, verbose=1)
    tsne_res = tsne.fit_transform(pca_res)
    col = np.arange(0, 256)
    
    plt.figure(figsize=(4, 3))
        
    sc = plt.scatter(tsne_res[:, 0], tsne_res[:, 1], c=col, alpha=1.0, cmap='nipy_spectral')
    plt.colorbar(pad=0.0)
    plt.savefig('tsne_%s.pdf' %fn, transparent=True, bbox_inches='tight', pad_inches=0, dpi=200)
    
    #fig1, ax1 = plt.subplots(figsize=(10, 6))
    #plt.colorbar(sc, ax=ax1, orientation='horizontal')
    #ax1.remove()
    #plt.savefig('tsne_legend.pdf', bbox_inches='tight')

def bar_graphs(i):
    models = ['Baseline', 'NAU']
    x = np.arange(3)+1
    
    Baseline = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    NAU = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    fig_legend = plt.figure()
    fig, ax = plt.subplots(figsize=(10, 4))
    
    patterns = ["+", "x"]
    ax1 = ax.bar(x - 0.15, Baseline[i], width=0.3, hatch=patterns[0], align='center', alpha=0.7, color='C3', edgecolor='black')
    ax2 = ax.bar(x + 0.15, NAU[i], width=0.3, hatch=patterns[1], align='center', alpha=0.7, color='C0', edgecolor='black')
    plt.xlim([0, 10])
    ax.set_xticks(x)
    ax.set_xticklabels(['256', '512', '1024'], fontsize=30)
    ax.legend((ax1, ax2), models, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fancybox=True, shadow=True, frameon=False, handlelength=4, fontsize=25)
    plt.ylabel('Final MSE Loss')
    plt.show()
    
    

def bar_graph_ex():

    mpl.rc('font', family='Times New Roman')

    sys = ['Baseline', 'NAU']
    x = np.arange(3) + 1
    
    """
    Layer 2
                Baseline        NAU
    1024        0.01205         0.00683
    512         0.01231         0.00826
    256         0.01257         0.01112
    
    Layer 3
                Baseline        NAU
    1024        0.01949         0.00454
    512         0.02177         0.00759
    256         0.02427         0.00824
    """
    

    #Baseline = [0.01257, 0.01231, 0.01205]    # 0.576
    #NAU = [0.01112, 0.00826, 0.00683]         # 0.983
    
    Baseline = [0.02427, 0.02177 , 0.01949]
    NAU = [0.00824, 0.00759, 0.00454]
    
    fig_legend = plt.figure()
    fig, ax = plt.subplots(figsize=(10, 6))

    patterns = ["+", "x"]
    ax1 = ax.bar(x - .15, Baseline, width=0.3, hatch=patterns[0], align='center', alpha=0.7, color='C3',
    edgecolor='black')
    ax2 = ax.bar(x + .15, NAU, width=0.3, hatch=patterns[1], align='center', alpha=0.7, color='C0',
    edgecolor='black')

    # ax.tick_params(axis='both', which='major', labelsize=18)

    plt.xlim([0.5, len(x) + 0.5])
    ax.set_xticks(x)
    ax.set_xticklabels(['256', '512', '1024'], fontsize=30)

    #ax.legend((ax1, ax2), sys, loc='upper center', bbox_to_anchor=(0.5, 1.2),
    #ncol=2, fancybox=True, shadow=True, frameon=False, handlelength=4, fontsize=25)

    plt.ylabel('MSE Loss', fontsize=20)

    #plt.show()
    plt.savefig('3Layer_Bar1.pdf', transparent=True, bbox_inches='tight', pad_inches=0, dpi=200)

    fig_legend.legend((ax1, ax2), sys, loc='upper center', ncol=3, frameon=False, handlelength=6,
    fontsize=18)
    fig_legend.savefig('bar_legend.pdf', transparent=True, bbox_inches='tight', pad_inches=0, dpi=200)

