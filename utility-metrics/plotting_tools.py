#!/usr/bin/env python

import codecs
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools

# Reference:
# https://www.kaggle.com/grfiv4/plot-a-confusion-matrix

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          prefix="",
                          cmap=None,
                          normalize=True, 
                          save_dir="."
                          ):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    save_dir      parent directory to save the images

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This function was modified slightly by the QUIPP development team.
    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=24)
    cbar = plt.colorbar(fraction=0.03)
    cbar.ax.tick_params(labelsize=24)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, size=20, rotation=90)
        plt.yticks(tick_marks, target_names, size=20)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize=22,
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize=22,
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', size=28)
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), size=28)
    plt.ylim(len(cm)-0.5, -0.5)
    figpath = f"{prefix}_{title}_confusion_matrix.png"
    save_path = os.path.join(save_dir, figpath)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_path, format="PNG", bbox_inches = "tight")
    return os.path.abspath(save_path)

def plot_util_confusion_matrix(confusion_dict_path, method_names=None, 
                               prefix="", normalize=False, save_dir="."):

    dict_r = codecs.open(confusion_dict_path, 'r', encoding='utf-8').read()
    confusion_dict = json.loads(dict_r)

    if type(method_names) == str:
        method_names = [method_names]
    if method_names == None:
        method_names = list(confusion_dict.keys())
    plt_names = []
    for method_name in method_names:
        if method_name not in confusion_dict:
            print(confusion_dict.keys())
            raise ValueError(f"Method name: {method_name} is not in the dictionary.")

        title = method_name
        cm = np.array(confusion_dict[method_name]["conf_matrix"])
        target_names = confusion_dict[method_name]["target_names"]
        plt_name = plot_confusion_matrix(cm, 
                                         target_names=target_names, 
                                         normalize=normalize, 
                                         title=title, 
                                         prefix=prefix, 
                                         save_dir=save_dir
                                         )
        plt_names.append(plt_name)
    return plt_names