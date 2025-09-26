"""Module to implement a Bar Graph"""

from graphs.graph import Graph

import matplotlib.pyplot as plt

class BarGraph(Graph):
    """Simple implementation of a bar graph"""

    def config(self, categories: list, data: list):
        """Method to set the necessary arguments for rendering

        Args:
            categories (list): List of categories, used to render the X axis
            data (list): The data to be displayed in each categories
        """
        self.categories = categories
        self.data = data


    def render(self, override_show: bool = False) -> None:
        """Render a bar graph"""
        plt.bar(self.categories, self.data)
        super().render(override_show)