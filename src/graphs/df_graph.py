from pandas import DataFrame

from graphs.graph import Graph

class DFGraph():

    def __init__(
        self,
        df: DataFrame,
        graph: Graph,
    ):
        self.df = df
        self.graph = graph

    def config(self, x_label, y_label):
        x_axis = self.df[x_label].to_list()
        y_axis = self.df[y_label].to_list()

        self.graph.config(x_axis, y_axis)

    def render(self, override_show: bool = False):
        self.graph.render(override_show)
