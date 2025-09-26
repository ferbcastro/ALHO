from graphs.graph import Graph

import matplotlib.pyplot as plt

class Comparison():

    def __init__(self, rows:int=1, columns:int=2):
        self.rows = rows
        self.columns = columns

    def render(self, graphs:list[Graph]):

        for index in range(len(graphs)):
            plt.subplot(self.rows, self.columns, index+1)

            graphs[index].render(True)

        plt.show()