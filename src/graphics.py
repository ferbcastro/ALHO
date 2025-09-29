
from matplotlib.pylab import PCG64
import numpy as np
import pandas as pd

from graphs.bar import BarGraph
from graphs.comparison import Comparison
from graphs.df_graph import DFGraph
from graphs.histogram import HistogramGraph
from graphs.scatter import ScatterGraph

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

    # Comparison between graphs
    comp = Comparison(rows=2, columns=2)

    fourgram_top10benign, fourgram_top10malignant = malign_begign_top_10("features_info_4.csv")
    threegram_top10benign, threegram_top10malignant = malign_begign_top_10("features_info_3.csv")

    comp.render([fourgram_top10malignant, fourgram_top10benign, threegram_top10malignant, threegram_top10benign])

if __name__ == "__main__":
    main()
