import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt






def create_cm_figure(cm, class_names):        
        figure = plt.figure(figsize=(30, 30)) if cm.shape[0] <= 61 else plt.figure(figsize=(80, 80)) 

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
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color, fontsize='medium')
            
        plt.tight_layout()

        plt.title("Confusion matrix")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        return figure



def colors_from_values(values):
    # normalize the values to range [0, 1]
    #normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    #indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    #palette = sns.diverging_palette(h_neg=240, h_pos=10, center="dark", n=len(values), s=100,  l=50)
    #palette = sns.color_palette('Reds', len(values))

    clrs = ['indianred' if (x == max(values)) else 'darkred' for x in values ]

    return clrs

    #rank = values.argsort().argsort()
    #return np.array(palette[::-1])[rank]

    #return np.array(palette).take(indices, axis=0)



# Setting newperson to True when creating metric figures for Siamese testing
def create_metrics_figure(metric_array, xlabel, ylabel, title, threshold=None, newperson=False):
    df = pd.DataFrame(metric_array.reshape(-1, len(metric_array)), columns=[str(i) for i in range(1, len(metric_array)+1)])

    figure = plt.figure(figsize=(20, 10))

    ax = sns.barplot(data=df, palette=colors_from_values(metric_array))

    # Modifying the last xtick label to New (NewPerson is too long)
    if newperson:
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[-1] = 'New'
        ax.set_xticklabels(labels)
    
    # Keeping only every 4 xlabel (and the NewPerson xlabel)
    num_xtick_labels = len(ax.xaxis.get_ticklabels())
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if not newperson:
            if index % 4 != 1:
                label.set_visible(False)
        else:
            if index % 4 != 1 and index != (num_xtick_labels - 1):
                label.set_visible(False)
    
    if threshold is not None:
        plt.yticks([tick for tick in list(plt.yticks()[0]) if abs(tick-threshold)>30] + [threshold])
        plt.axhline(y=threshold, color='k', linestyle='-', linewidth=4)
    
    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.title(label=title, fontsize='xx-large')
    plt.xlabel(xlabel=xlabel, fontsize='xx-large')
    plt.ylabel(ylabel=ylabel, fontsize='xx-large')

    plt.tight_layout()

    return figure