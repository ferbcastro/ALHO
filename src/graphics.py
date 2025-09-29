
from matplotlib.pylab import PCG64
import numpy as np
import pandas as pd

from graphs.bar import BarGraph
from graphs.comparison import Comparison
from graphs.df_graph import DFGraph
from graphs.histogram import HistogramGraph
from graphs.scatter import ScatterGraph


def main():
    # Data for bar graph
    categories = ["a", "b", "c"]
    data = [1, 2, 3]

    # Data for histogram
    x = np.random.Generator(PCG64())
    hist_data = x.standard_normal(170)

    # Graphs initialization
    graph = BarGraph("Teste Bar", "letras", "numeros")
    hist = HistogramGraph("Teste Hist", "valores", "freq")
    scat = ScatterGraph("Teste Scatter", "letras", "numeros")

    # Graphs configuration
    graph.config(categories, data)
    hist.config(hist_data)
    scat.config(categories, data)

    # Graphs rendering
    graph.render()
    hist.render()
    scat.render()

    # Comparison between graphs
    comp = Comparison()
    comp.render([graph, hist])

    df = pd.read_csv("features_info_3.csv")
    df_graph = DFGraph(
        df=df,
        graph=ScatterGraph(
            "titulo",
            "gram",
            "freq"
        ),
        x_label="gram_names", 
        y_label="frequency"
    )

    df_graph.render()

if __name__ == "__main__":
    main()
