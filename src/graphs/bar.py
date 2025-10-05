"""Module to implement a Bar Graph"""

# External libs
import matplotlib.pyplot as plt
import numpy as np

# Internal modules
from graphs.graph import Graph


class BarGraph(Graph):
    """Simple implementation of a bar graph"""

    def config(self, categories: list, data: list):
        """Method to set the necessary arguments for rendering

        Args:
            categories (list): List of categories, used to render the X axis
            data (list): The data to be displayed in each categories
        """
        assert len(categories) == len(data)

        self.categories = categories
        self.data = data

    def render(self, override_show: bool = False):
        plt.bar(self.categories, self.data)
        super().render(override_show)
