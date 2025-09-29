"""Module to implement a Histogram"""
# External libs
import matplotlib.pyplot as plt

# Internal modules
from graphs.graph import Graph


class HistogramGraph(Graph):
    """Simple implementation of a histogram"""

    def config(self, frequencies: list, bins: int =20):
        """Method to set the necessary arguments for rendering

        Args:
            frequencies (list): List of elements to be displayed
            bins (int): How many divisions the histogram should have
        """
        self.frequencies = frequencies
        self.bins = bins

    def render(self, override_show: bool = False) -> None:
        """Render a histogram"""
        plt.hist(self.frequencies, self.bins)
        super().render(override_show)
