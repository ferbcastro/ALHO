"""Module to implement a Bar Graph"""

# External libs
import matplotlib.pyplot as plt

# Internal modules
from graphs.graph import Graph


class ScatterGraph(Graph):
    """Simple implementation of a scatter graph"""
    categories = []
    data = []

    def config(self, categories: list, data: list):
        """Method to set the necessary arguments for rendering

        Args:
            categories (list): List of categories, used to render the X axis
            data (list): The data to be displayed in each categories
        """
        assert len(categories) == len(data)

        self.categories = categories
        self.data = data


    def render(self, override_show: bool = False) -> None:
        """Render a scatter graph"""
        plt.scatter(
            self.categories,
            self.data,
            s=20,
        )
        plt.ylim(0, 1)
        plt.xlim(0, 2)

        super().render(override_show)

