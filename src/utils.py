import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_with_labels(low_dim_embs, labels, fname):
    plt.figure(figsize=(28, 28))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(fname)
    plt.close()
    pass
