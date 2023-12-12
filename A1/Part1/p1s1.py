import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-tr", "--train", help = "train data file path")



# Read arguments from command line
args = parser.parse_args()

file_path = args.train
train_dataset = pd.read_csv(file_path)


features = train_dataset.drop('t',axis = 1)

plt.figure(figsize = (8,8))
hmap = sns.heatmap(features.corr(),annot = True,cmap = 'coolwarm')
figure = hmap.get_figure()
figure.savefig('pairwise_corr_heatmap.png')
plt.show()


pplot = sns.pairplot(features,height = 2.5)
pplot.savefig('pair_plot.png')
plt.show()

