import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt






def create_cm_figure(cm, class_names):        
        figure = plt.figure(figsize=(80, 80))

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        
        plt.colorbar()

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Normalize the confusion matrix.
        #cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        
        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
            
        plt.tight_layout()

        plt.title("Confusion matrix")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        return figure



def colors_from_values(values):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.diverging_palette(h_neg=240, h_pos=10, center="light", n=len(values), s=100,  l=50)
    #palette = sns.color_palette('Reds', len(values))
    return np.array(palette).take(indices, axis=0)



def create_metrics_figure(metric_array, xlabel, ylabel, title, threshold=None):
    df = pd.DataFrame(metric_array.reshape(-1, len(metric_array)), columns=[str(i) for i in range(1, len(metric_array)+1)])

    figure = plt.figure(figsize=(20, 10))

    ax = sns.barplot(data=df, palette=colors_from_values(metric_array))

    # Keeping only every 4 x label
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % 4 != 1:
            label.set_visible(False)

    if threshold is not None:
        plt.yticks([tick for tick in list(plt.yticks()[0]) if abs(tick-threshold)>30] + [threshold])
        plt.axhline(y=threshold, color='k', linestyle='-')
    
    plt.title(label=title)
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)

    plt.tight_layout()

    return figure