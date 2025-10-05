#External libs
import matplotlib.pyplot as plt

from matplotlib.pylab import PCG64
import numpy as np
import pandas as pd

from graphs.bar import BarGraph
from graphs.comparison import Comparison
from graphs.df_graph import DFGraph
from graphs.histogram import HistogramGraph
from graphs.scatter import ScatterGraph

CSV_SOURCE_1 = 'features_ppv_npv_freqs_4.csv'
CSV_SOURCE_2 = 'leg_PhiUSIIL.csv'
GRAPH_NAME_1 = 'Selection (n = 200)'
GRAPH_NAME_2 = 'Legitimate URLs'
X_LABEL_GRAPH = 'PPV'
Y_LABEL_GRAPH = 'NPV'
X_LABEL_CSV = 'ppv'
Y_LABEL_CSV = 'npv'

def malign_begign_top_10(file_path: str) -> tuple:
    df = pd.read_csv(file_path)
    top10malignant = df.copy().sort_values(by=["frequency", "specificity"]).head(10)
    top10benign = df.copy().sort_values(by="frequency").sort_values(by="specificity", ascending=False).head(10)

    top10_malignant_graph = DFGraph(
        df=top10malignant,
        graph=ScatterGraph(
            f"Top 10 {file_path} Maligns",
            "gram",
            "freq"
        ),
        x_label="gram_names",
        y_label="frequency"
    )
    top10_benign_graph = DFGraph(
        df=top10benign,
        graph=ScatterGraph(
            f"Top 10 {file_path} Benigns",
            "gram",
            "freq"
        ),
        x_label="gram_names",
        y_label="frequency"
    )

    return top10_benign_graph, top10_malignant_graph
    


def main():
    # Data for bar graph
    # categories = ["a", "b", "c"]
    # data = [1, 2, 3]

    # Data for histogram
    # x = np.random.Generator(PCG64())
    # hist_data = x.standard_normal(170)

    # Graphs initialization
    #bar1 = BarGraph(GRAPH_NAME, X_LABEL, Y_LABEL)
    #bar2 = BarGraph(GRAPH_NAME, X_LABEL, Y_LABEL)
    #hist = HistogramGraph(GRAPH_NAME, "valores", "freq")
    #scat = ScatterGraph(GRAPH_NAME", X_LABEL, Y_LABEL)

    # Graphs configuration
    #bars.config(categories, data)
    #hist.config(hist_data)
    #scat.config(categories, data)

    # Graphs rendering
    #bars.render()
    #hist.render()
    #scat.render()

    df = pd.read_csv(CSV_SOURCE_1)[:200]
    df_graph_1 = DFGraph(
        df=df,
        graph=ScatterGraph(
            GRAPH_NAME_1,
            X_LABEL_GRAPH,
            Y_LABEL_GRAPH
        ),
    )
    df_graph_1.config(X_LABEL_CSV, Y_LABEL_CSV)
    df_graph_1.render()
    plt.savefig('features_ppv_npv_freqs_4.png')

    # Comparison between graphs
    comp = Comparison(rows=2, columns=2)

    fourgram_top10benign, fourgram_top10malignant = malign_begign_top_10("features_info_4.csv")
    threegram_top10benign, threegram_top10malignant = malign_begign_top_10("features_info_3.csv")

    comp.export([fourgram_top10malignant, fourgram_top10benign, threegram_top10malignant, threegram_top10benign])


if __name__ == "__main__":
    main()
