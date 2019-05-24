import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns;

background_style = plt.style.use('seaborn-whitegrid')
single_color = '#008080'
transparency = 0.9


def plot_style(title, xlabel, ylabel):

    plt.tick_params(labelsize=16)
    plt.title(title, fontsize=24)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.show()


def lineplot(data, title, xlabel, ylabel, figsize=(18, 8), x=None, y=None):
    background_style
    plt.figure(figsize=figsize)
    if x is None:
        sns.lineplot(data=data, color=single_color, alpha=transparency)
    else:
        sns.lineplot(data=data, x=x, y=y, color=single_color, alpha=transparency, lw=2)
    plt.grid(axis='x')
    plt.ylim(0)
    plot_style(title, xlabel, ylabel)



def barplot(data, x, y, title, xlabel, ylabel, figsize=(18, 8), labels=True):
    background_style
    plt.figure(figsize=figsize)
    p = sns.barplot(data=data, x=x, y=y, alpha=0.7)
    label_shift = int(data[y].max()) * 0.02
    if labels:
        for index, row in data.iterrows():
            p.text(row.name, row[y] + label_shift, '{:,}'.format(int(row[y])).replace(',', ' '), color='black', ha="center", fontsize=14)
    plot_style(title, xlabel, ylabel)

def densityplot(data, x, category, title, xlabel, ylabel, figsize=(18,8)):
    background_style
    f, ax = plt.subplots(figsize=figsize)
    ax.set(xscale="log")
    categories = data[category].unique()
    for c in categories:
        sns.kdeplot(data[data[category] == c][x], shade=True, ax = ax, label=f"{category}: {c}", gridsize=500)
    plot_style(title, xlabel, ylabel)


def boxplot(data, x, y, title, xlabel, ylabel, figsize=(18, 8), yscale='linear'):
    background_style
    f, ax = plt.subplots(figsize=figsize)
    ax.set(yscale=yscale)
    sns.boxplot(x=x, y=y, data=data, notch=True, width=0.5, saturation=0.9, boxprops=dict(alpha=.5), ax=ax)
    # sns.stripplot(x=x, y=y, data=data, color=single_color, jitter=0.2, size=2.5)
    plot_style(title, xlabel, ylabel)


