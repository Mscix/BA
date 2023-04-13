import pandas as pd
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, df):
        self.df = df

    def plot_accuracy(self):
        self.df[['x', 'y']].plot()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.show()



