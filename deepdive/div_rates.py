import copy
import glob
import os.path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf  # saves pdfs
from .feature_extraction import *
from .utilities import prep_dd_input, print_update, predict
from .plots import add_geochrono_no_labels
def plot_empirical_diversification(res_wd, dd_estimate_file,
        log_cols=True,
        out_file=None, vmin=None, vmax=None,
        plot_per_lineage_rate=True):

    dd_pred = pd.read_csv(dd_estimate_file)
    time_bins = np.array(dd_pred.columns).astype(float)[::-1]
    time_bins_mid = np.array(
        [np.mean([time_bins[i], time_bins[i + 1]]) for i in range(0, len(time_bins) - 1)])


    dd_pred_np = np.flip(np.array(dd_pred), axis=1)[:,:-1]
    div_rates_raw = np.diff(dd_pred_np, axis=1)

    if plot_per_lineage_rate:
        div_rate = div_rates_raw / dd_pred_np[:,:-1]
    else:
        div_rate = div_rates_raw

    # div_rate[:, np.where(time_bins > 148)[0]] = 0

    # PLOT
    fig = plt.figure(figsize=(7, 5))


    if log_cols:
        colors = np.log(1 + np.abs(div_rate))
        colors[div_rate < 0] = -colors[div_rate < 0]
    else:
        colors = div_rate

    val = np.maximum(np.abs(np.min(colors)), np.max(colors))
    if vmax is None:
        vmax = val
    if vmin is None:
        vmin = -val

    for i in range(div_rate.shape[0]):
        plt.scatter(-time_bins_mid[1:], div_rate[i],
                    c=colors[i], cmap="coolwarm", alpha=0.99,
                    vmin=vmin, vmax=vmax,
                    s=32)

    plt.hlines(0, -time_bins[0], 0,
               colors="k", linestyles="dashed", alpha=0.4)

    if plot_per_lineage_rate:
        plt.plot(-time_bins_mid[1:],np.mean(div_rate, axis=0),
                 "k", linewidth=2)
    else:
        plt.step(-time_bins[1:],np.mean(div_rate, axis=0),
                 "w", linewidth=2)


    plt.xlim(-145, -np.min(time_bins) + 2)
    if plot_per_lineage_rate:
        add_geochrono_no_labels(-0.70, -0.62, max_ma=-(np.max(time_bins) * 1.05), min_ma=0)
        plt.ylim(bottom=-0.70, top=0.82)
        plt.ylabel("Diversification rate", fontsize=15)
    else:
        add_geochrono_no_labels(-300, -260, max_ma=-(np.max(time_bins) * 1.05), min_ma=0)
        plt.ylim(bottom=np.min(div_rate) * 1.05, top=np.max(div_rate) * 1.05)
        plt.ylabel("Diversity change", fontsize=15)
    plt.xlabel("Time (Ma)", fontsize=15)

    # plt.show()

    if out_file is None:
        out_file = os.path.basename(dd_estimate_file)
    else:
        out_file = out_file
    if plot_per_lineage_rate:
        out_file = "%s_diversification_rates.pdf" % out_file
    else:
        out_file = "%s_diversity_change.pdf" % out_file
    file_name = os.path.join(res_wd, out_file)
    div_plot = matplotlib.backends.backend_pdf.PdfPages(file_name)
    div_plot.savefig(fig)
    div_plot.close()

    res = np.vstack((-time_bins_mid[1:], div_rate))
    pd.DataFrame(res).to_csv(str(file_name).replace(".pdf", ".csv"),
                             index=False)

    print("Plot saved as", file_name)
