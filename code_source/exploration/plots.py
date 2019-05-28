import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from matplotlib.ticker import FuncFormatter
import pandas as pd
from functools import reduce

background_style = plt.style.use('seaborn-whitegrid')
single_color = '#008080'
second_color = '#BF7FA6'
transparency = 0.9


def format_number(number):
    return '{:,}'.format(int(number)).replace(',', ' ')


def plot_style(title, xlabel, ylabel, xticks):
    plt.tick_params(labelsize=xticks)
    plt.title(title, fontsize=24)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    # plt.xticks(fontsize=8, rotation=90)
    plt.show()


def lineplot(data, title, xlabel, ylabel, x=None, y=None, figsize=(18, 8), xticks=16, percent=False, xscale='linear'):
    background_style
    f, ax = plt.subplots(figsize=figsize)
    ax.set(xscale=xscale)
    if x is None:
        a = sns.lineplot(data=data, color=single_color, alpha=transparency)
    else:
        a = sns.lineplot(data=data, x=x, y=y, color=single_color, alpha=transparency, lw=2)
    plt.grid(axis='x')
    plt.ylim(0)
    if percent:
        a.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
    plot_style(title, xlabel, ylabel, xticks)


def barplot(data, x, y, title, xlabel, ylabel, figsize=(18, 8), labels=True, xticks=16, color=False, horizontallabels=False):
    background_style
    plt.figure(figsize=figsize)
    if color:
        sns.set_palette([single_color, second_color,single_color, second_color])
    p = sns.barplot(data=data, x=x, y=y, alpha=0.7)
    if horizontallabels:
        label_shift = data[x].max() * 0.05
        for index, row in data.iterrows():
            p.text(row.Score + label_shift, index + 0.25, round(row[x], 3), color='black',
               ha="center", fontsize=14)
    if labels:
        label_shift = int(data[y].max()) * 0.02
        for index, row in data.iterrows():
            p.text(row.name, row[y] + label_shift, format_number(row[y]), color='black',
                   ha="center", fontsize=14)
    plot_style(title, xlabel, ylabel, xticks)

def pairplot(data):
    p=sns.pairplot(data, kind='reg', hue='Default', markers=['o', 'o'], palette='husl')
    p.fig.suptitle('Paired scatter and regression plot',y=1.01,fontsize=24)
    plt.show


def stacked_barplot(data, x, y, title, xlabel, ylabel, figsize=(18, 8), xticks=16):
    background_style
    data.plot(kind='bar', x=x, y=y, stacked=True, alpha=0.7, figsize=figsize, rot=0, color=[single_color, second_color])
    plt.grid(axis='x')
    plt.legend(fontsize=16)
    plt.ylim(0)
    plot_style(title, xlabel, ylabel, xticks)


def densityplot(data, x, category, title, xlabel, ylabel, figsize=(18, 8), xticks=16):
    background_style
    f, ax = plt.subplots(figsize=figsize)
    ax.set(xscale="log")
    categories = data[category].unique()
    sns.set_palette([single_color, second_color])
    for c in categories:
        sns.kdeplot(data[data[category] == c][x], shade=True, ax=ax, label=f"{category}: {c}", gridsize=500)
    plot_style(title, xlabel, ylabel, xticks)


def boxplot(data, x, y, title, xlabel, ylabel, figsize=(18, 8), yscale='linear', xticks=16):
    background_style
    f, ax = plt.subplots(figsize=figsize)
    ax.set(yscale=yscale)
    sns.set_palette([single_color, second_color])
    sns.boxplot(x=x, y=y, data=data, notch=True, width=0.5, saturation=0.9, boxprops=dict(alpha=.5), ax=ax)
    # sns.stripplot(x=x, y=y, data=data, color=single_color, jitter=0.2, size=2.5)
    plot_style(title, xlabel, ylabel, xticks)


def roc_df(true_y, score_y, title):
    fpr, tpr, thresholds = roc_curve(true_y, score_y)
    auc = roc_auc_score(true_y, score_y)
    df = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Model": f"{title}: {round(auc, 3)}"})
    return df


def rocplot(true_y, predictors, titles):
    background_style
    plt.figure(figsize=[8, 8])
    ax = sns.lineplot([0, 1], [0, 1])
    ax.lines[0].set_linestyle("--")
    ax.lines[0].set_color("black")
    ax.set(ylim=[0, 1], xlim=[0, 1])

    df_list = [roc_df(true_y, predictors[i], titles[i]) for i in range(len(predictors))]
    df = pd.concat(df_list)
    sns.lineplot("FPR", "TPR", hue="Model", data=df, palette='husl')
    plt.grid(axis='x')

    plt.legend(fontsize=16)
    plt.tick_params(labelsize=16)
    plt.title('ROC comparison', fontsize=24)
    plt.xlabel(xlabel="FPR", fontsize=16)
    plt.ylabel(ylabel="TPR", fontsize=16)
    plt.show()


def heatmap(data, figsize=(18, 12)):
    np.random.seed(0)
    mask = np.zeros_like(data)
    mask[np.triu_indices_from(mask)] = True
    plt.figure(figsize=figsize)
    with sns.axes_style("white"):
        # sns.heatmap(data, mask=mask, square=True, cmap='PiYG', alpha=0.8, vmin=-1, vmax=1, )
        sns.heatmap(data, mask=mask, square=True, cmap=sns.diverging_palette(220, 10, s=74, l=50, sep=10, n=9, as_cmap=True), alpha=0.8, vmin=-1, vmax=1, annot=True, fmt='.3f')
    plt.title('Correlation plot', fontsize=24)
    plt.show()

def kdeplot(data, title, xlabel, ylabel, figsize=(18, 8), xticks=16):
    background_style
    plt.subplots(figsize=figsize)
    sns.kdeplot(data=data, color=single_color, alpha=transparency)
    plt.grid(axis='x')
    plt.ylim(0)
    plot_style(title, xlabel, ylabel, xticks)

def revenue_lineplot(rev_lm , rev_log, rev_svm_l, rev_rf, rev_gb):
    plt.subplots(figsize=(18,8))
    plt.style.use('seaborn-whitegrid')
    sns.set_palette('husl')
    sns.lineplot(data=rev_lm, x='Threshold', y='Revenue')
    sns.lineplot(data=rev_log, x='Threshold', y='Revenue')
    sns.lineplot(data=rev_svm_l, x='Threshold', y='Revenue')
    sns.lineplot(data=rev_rf, x='Threshold', y='Revenue',lw=3, color='#BF7FA6')
    sns.lineplot(data=rev_gb, x='Threshold', y='Revenue')
    plt.legend(['Linear Regression', 'Logistic Regression', \
             'Linear SVM','Random forest', 'Gradient Boosting'],fontsize=16)
    plt.grid(axis='x')
    plt.tick_params(labelsize=16)
    plt.title('Revenue depending on model', fontsize=24)
    plt.xlabel('Threshold',fontsize=16)
    plt.ylabel('Revenue', fontsize=16)
    plt.ylim(0)
    plt.show()