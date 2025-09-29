from pandas import DataFrame

from graphs.graph import Graph

class DFGraph():

    def __init__(
        self,
        df: DataFrame,
        graph: Graph,
        x_label: str,
        y_label: str
    ):
        self.df = df
        self.graph = graph
        self.config(x_label, y_label)

    def config(self, x_label, y_label):
        x_axis = self.df[x_label].to_list()
        y_axis = self.df[y_label].to_list()

        self.graph.config(x_axis, y_axis)

    def render(self, override_show: bool = False):
        self.graph.render(override_show)
