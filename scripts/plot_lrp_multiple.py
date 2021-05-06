import pickle
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import pylab
import argparse

def avg_lrp_by_pos(data, seg='inp'):
    count = 0
    res = np.zeros(data[0][seg + '_lrp'].shape[0])
    for i in range(len(data)):
        if not any(np.isnan(data[i][seg + '_lrp'])):
            res += np.sum(data[i][seg + '_lrp'], axis=-1)
            count += 1
    res /= count
    return res


from scipy.stats import entropy

def all_inp_entropy(data, pos=None):
    res = []
    for i in range(len(data)):
        if not any(np.isnan(data[i]['inp_lrp'])):
            res_ = np.sum(data[i]['inp_lrp'], axis=-1)
            try:
                if pos is None:
                    res += [entropy(data[i]['inp_lrp'][p]/res_[p]) for p in range(data[i]['inp_lrp'].shape[0])]
                else:
                    res.append(entropy(data[i]['inp_lrp'][pos] / res_[pos]))
            except Exception:
                pass
    return res

def all_out_entropy(data, pos):
    res = []
    for i in range(len(data)):
        if not any(np.isnan(data[i]['out_lrp'])):
            res_ = np.sum(data[i]['out_lrp'], axis=-1)
            res.append(entropy(data[i]['out_lrp'][pos][:pos + 1] / res_[pos]))
    return res

def avg_lrp_by_src_pos_normed(data, ignore_eos=False):
    count = 0
    tgt_tokens = data[0]['inp_lrp'].shape[0]
    if ignore_eos:
        res = np.zeros(data[0]['inp_lrp'].shape[1] - 1)
    else:
        res = np.zeros(data[0]['inp_lrp'].shape[1])
    for i in range(len(data)):
        if not any(np.isnan(data[i]['inp_lrp'])):
            if ignore_eos:
                elem = data[i]['inp_lrp'][:, :-1] / np.sum(data[i]['inp_lrp'][:, :-1], axis=1).reshape([tgt_tokens, 1])
            else:
                elem = data[i]['inp_lrp'] / np.sum(data[i]['inp_lrp'], axis=1).reshape([tgt_tokens, 1])
            res += np.sum(elem, axis=0)
            count += 1
    res /= count
    res /= tgt_tokens
    if ignore_eos:
        res *= (data[0]['inp_lrp'].shape[1] - 1)
    else:
        res *= data[0]['inp_lrp'].shape[1]
    return res

def plot_source_influence(alldata,filename,colors,labels):
    fig = plt.figure(figsize=(7, 6), dpi=100)

    for data, color in zip(alldata,colors):
        res = avg_lrp_by_pos(data, seg='inp')[1:]
        plt.plot(range(2, len(res)+2), res, lw=2., color=color)
        plt.scatter(range(2, len(res)+2), res, lw=3.0, color=color)

    if labels is not None:
        plt.legend(labels)
    plt.xlabel("target token position", size=25)
    plt.ylabel("source contribution", size=25)
    plt.yticks(size=20)
    plt.xticks([2, 5, 10, 15, 20], size=20)
    plt.title('source ⟶ target(k)', size=25)
    plt.grid()

    pylab.savefig(filename, bbox_inches='tight')


def main():
    p = argparse.ArgumentParser('plot_lrp.py')
    p.add_argument('--data', required=True)
    p.add_argument('--labels')
    p.add_argument('--output-prefix', required=True)

    args = p.parse_args()

    data_list=[]
    for filename in args.data.split(","):
        data = pickle.load(open(filename, 'rb'))
        data_list.append(data)

    labels=None
    if args.labels:
        labels=args.labels.split(",")

    #Plot configuration
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}
    plt.rc('font', **font)
    cmap = cm.get_cmap('Spectral', 30)    # PiYG
    spectral_map = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
    colors = spectral_map[-len(data_list):]


    plot_source_influence(data_list,args.output_prefix+".sourceinfluence.png",colors,labels)


if __name__ == '__main__':
    main()
