"""Module to show a comparison between graphs"""

# External libs
import matplotlib.pyplot as plt

# Internal modules
from graphs.graph import Graph


class Comparison():

    def __init__(self, rows:int=1, columns:int=2):
        self.rows = rows
        self.columns = columns

    def render(self, graphs: list[Graph]):
        """Render a comparison between graphs"""

        for index in range(len(graphs)):
            plt.subplot(self.rows, self.columns, index+1)

            graphs[index].render(True)

        plt.show()
    
    def export(self, graphs: list[Graph]):
        plt.figure(figsize=(10, 10))
        for index in range(len(graphs)):
            plt.subplot(self.rows, self.columns, index+1)

            graphs[index].render(True)
        
        plt.savefig("top_k_ngrams.pdf", dpi=199)
