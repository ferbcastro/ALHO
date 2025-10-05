"""Module to serve as basic implementation for graphs"""

#External libs
import matplotlib.pyplot as plt

class Graph():
    """Basic implementation of a graph"""

    def __init__(
        self,
        title: str,
        x_label: str,
        y_label: str
    ):
        self.title = title
        self.x_label = x_label
        self.y_label = y_label

    def config(self, x, y):
        """Method to implement the data transformation of the graph"""

    def render(self, override_show: bool = False):
        """
        Function that renders the graph

        Automatically renders title, x and y labels
        """
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

        if override_show:
            plt.show()
