import numpy as np
import util
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

def gen_confusion_matrix(y, y_hat, set="training"):
    y_hat = np.array(y_hat)
    cm_train = confusion_matrix(y, y_hat)
    plt.subplots(figsize=(10, 6))
    sb.heatmap(cm_train, annot = True, fmt = 'g')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for the {set} set")
    plt.show()

def getAxisValues(axis,stepSize):
    return np.arange(axis['xmin'], axis['xmax'], stepSize)

def gen_membership_function(feature, i):
    cmap = get_cmap(len(feature))
    # Setting a domain field.
    xaxis = { 'xmin':0, 'xmax':25 }

    # Creating the values for the domain of all our membership functions.
    xrange = getAxisValues(xaxis, 0.1)

    for k, f in enumerate(feature):
        s, m, mf = f[0], f[1], f[2]
        if mf == 1:
            y = []
            for x in xrange:
                y.append(util.isosceles_triangular(x, s, m))
            plt.plot(xrange, y, color=cmap(k))
        elif mf == 2:
            y = []
            for x in xrange:
                y.append(util.right_angled_trapezoidal(x, s, m))
            plt.plot(xrange, y, color=cmap(k))
        elif mf == 3:
            y = []
            for x in xrange:
                y.append(util.gaussian(x, s, m))
            plt.plot(xrange, y, color=cmap(k))
        elif mf == 4:
            y = []
            for x in xrange:
                y.append(util.sigmoid(x, s, m))
            plt.plot(xrange, y, color=cmap(k))

    plt.xlabel(f"Feature number {i}, x range")
    plt.ylabel("Fuzzy membership value")
    plt.title(f"Feature number {i}, {len(feature)} Linguistic variables")
    plt.show()

def avg_fitness():
    df = pd.read_csv('histories.csv', header=None)
    df.loc['mean'] = df.mean()
    df.loc['min'] = df.min()
    df.loc['max'] = df.max()
    print(df.loc['mean'].max())
    fig, ax = plt.subplots(figsize =(20,20))
    ax.plot(df.columns,df.loc['mean'])
    ax.fill_between(df.columns, df.loc['min'], df.loc['max'], alpha=0.2)
    ax.set_xlabel('Generations')
    ax.set_ylabel('Fitness (MCC)')
    plt.xticks(np.arange(min(df.columns), max(df.columns)+25, 25.0))
    plt.show()

if __name__ == "__main__":
    avg_fitness()
