
from matplotlib.pylab import PCG64
import numpy as np

from graphs.bar import BarGraph
from graphs.comparison import Comparison
from graphs.histogram import Histogram


def main():
    # Data for bar graph
    categories = ["a", "b", "c"]
    data = [1, 2, 3]

    # Data for histogram
    x = np.random.Generator(PCG64())
    hist_data = x.standard_normal(170)

    # Graphs initialization
    graph = BarGraph("Teste Bar", "letras", "numeros")
    hist = Histogram("Teste Hist", "valores", "freq")

    # Graphs configuration
    graph.config(categories, data)
    hist.config(hist_data)

    # Graphs rendering
    graph.render()
    hist.render()

    # Comparison between graphs
    comp = Comparison()
    comp.render([graph, hist])

if __name__ == "__main__":
    main()
