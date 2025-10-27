import copy
import glob
import os.path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf  # saves pdfs
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from .feature_extraction import *
from .utilities import prep_dd_input, print_update, predict


def plot_dd_predictions(pred_div, time_bins, wd, out_tag="", total_diversity=False):

    if total_diversity:
        fig = plt.figure(figsize=(12, 8))
        plt.hist(pred_div.flatten())
        plt.ylabel("Frequency", fontsize=15)
        plt.xlabel("Total diversity", fontsize=15)
    else:
        pred = np.mean(pred_div, axis=0)

        fig = plt.figure(figsize=(12, 8))

        plt.fill_between(-time_bins,
                         y1=np.max(pred_div, axis=0).T,
                         y2=np.min(pred_div, axis=0).T,
                         step="pre",
                         color="b",
                         alpha=0.2)

        plt.fill_between(-time_bins,
                         y1=np.quantile(pred_div, q=0.975, axis=0).T,
                         y2=np.quantile(pred_div, q=0.025, axis=0).T,
                         step="pre",
                         color="b",
                         alpha=0.2)

        plt.fill_between(-time_bins,
                         y1=np.quantile(pred_div, q=0.75, axis=0).T,
                         y2=np.quantile(pred_div, q=0.25, axis=0).T,
                         step="pre",
                         color="b",
                         alpha=0.2)

        plt.step(-time_bins,
                 pred.T,
                 label="Mean prediction",
                 linewidth=2,
                 c="b",
                 alpha=1)

        add_geochrono_no_labels(0, -0.1 * np.max(pred), max_ma=-(np.max(time_bins) * 1.05), min_ma=0)
        plt.ylim(bottom=-0.1 * np.max(pred), top=np.max(pred_div) * 1.05)
        plt.xlim(-(np.max(time_bins) * 1.05), -np.min(time_bins) + 2)
        plt.ylabel("Diversity", fontsize=15)
        plt.xlabel("Time (Ma)", fontsize=15)

    file_name = os.path.join(wd, "Empirical_predictions_%s.pdf" % out_tag)
    div_plot = matplotlib.backends.backend_pdf.PdfPages(file_name)
    div_plot.savefig(fig)
    div_plot.close()
    print("Plot saved as:", file_name)



def plot_ensemble_predictions(csv_files=None,
                              model_wd=None,
                              empirical_prediction_tag="Empirical_predictions_",
                              wd=None, out_tag="",
                              save_predictions=True,
                              verbose=False,
                              tot_div=False,
                              return_file_name=False):
    if model_wd is not None:
        csv_files = []
        if tot_div:
            model_folders = glob.glob(os.path.join(model_wd, "*_totdiv"))
        else:
            # model_folders = np.array(glob.glob(os.path.join(model_wd, "*")))
            model_folders = np.array([os.path.join(model_wd, i) for i in next(os.walk(model_wd))[1]])
            model_folders = model_folders[["_totdiv" not in i for i in model_folders]]
        for i in model_folders:
            # print(os.path.join(i, "*%s*.csv" %  empirical_prediction_tag))
            f = glob.glob(os.path.join(i, "*%s*.csv" %  empirical_prediction_tag))
            print_update("Found %s files" % len(csv_files))
            csv_files.append(f[0])

        if verbose:
            print(csv_files)

        print("\n")

    pred_div_list = None
    for f in csv_files:
        f_pd = pd.read_csv(f)
        if tot_div:
            time_bins = ["total_diversity"]
        else:
            time_bins = f_pd.columns.to_numpy().astype(float)
        if pred_div_list is None:
            pred_div_list = f_pd.to_numpy().astype(float)
        else:
            pred_div_list = np.vstack((pred_div_list, f_pd.to_numpy().astype(float)))

    if save_predictions:
        if tot_div:
            out_tag = out_tag + "_totdiv"
        pred_div_list_pd = pd.DataFrame(pred_div_list)
        pred_div_list_pd.columns = time_bins
        pred_div_list_pd.to_csv(os.path.join(wd,
                                             "Empirical_predictions_%s.csv" % out_tag),
                                index=False)
    if tot_div is False:
        plot_dd_predictions(pred_div_list, time_bins, wd, out_tag)

    if return_file_name:
        return os.path.join(wd, "Empirical_predictions_%s.csv" % out_tag)



def plot_trajectories(sim_obj,
                      simulation,
                      file_name="div_traj.pdf",
                      fontsize=8,
                      show=False):

    mid_time_bins = -sim_obj.mid_time_bins
    fig = plt.figure(figsize=(15, 10))

    fig.add_subplot(231)
    # plot the first line: simulated true global diversity
    plt.plot(mid_time_bins, simulation['global_true_trajectory'], '-', label="True diversity")
    # plot a second line of diversity from the fossil record
    plt.plot(mid_time_bins, simulation['global_fossil_trajectory'], '-', label="Sampled diversity")
    plt.xlabel("Time (Ma)")
    plt.ylabel("Diversity (no. species)")
    plt.gca().set_title("A) Simulated Global Diversity Trajectories", fontweight="bold", fontsize=fontsize)

    # LOCAL DIV PLOTS
    fig.add_subplot(232)
    plt.plot(mid_time_bins, simulation['local_true_trajectory'].T, '-')
    plt.xlabel("Time (Ma)")
    plt.ylabel("Diversity (No. species)")
    plt.gca().set_title("B) Simulated True Diversity Trajectories By Area", fontweight="bold", fontsize=fontsize)

    # plot second local fossil data
    fig.add_subplot(233)
    plt.plot(mid_time_bins, simulation['local_fossil_trajectory'].T, '-')
    plt.xlabel("Time (Ma)")
    plt.ylabel("Diversity (No. species)")
    plt.gca().set_title("C) Simulated Fossil Diversity Trajectories By Area", fontweight="bold", fontsize=fontsize)

    # plot number of localities
    fig.add_subplot(234)
    plt.plot(mid_time_bins, simulation['localities'].T, '-')
    plt.xlabel("Time (Ma)")
    plt.ylabel("No. localities")
    plt.gca().set_title("D) Simulated number of fossil localities per area", fontweight="bold", fontsize=fontsize)

    fig.add_subplot(235)
    plt.plot(mid_time_bins, simulation['n_localities_w_fossils'].T, '-')
    plt.xlabel("Time (Ma)")
    plt.ylabel("No. localities")
    plt.gca().set_title("E) Simulated number of sampled fossil localities per area", fontweight="bold", fontsize=fontsize)

    fig.add_subplot(236)
    plt.plot(mid_time_bins, n_occs_per_area_time(simulation).T, '-')
    plt.xlabel("Time (Ma)")
    plt.ylabel("No. occurrences")
    plt.gca().set_title("F) Number of fossil occurrences per area", fontweight="bold", fontsize=fontsize)

    # fig.tight_layout()
    if show:
        plt.show()
    else:
        plot_div = matplotlib.backends.backend_pdf.PdfPages(file_name)
        plot_div.savefig(fig)
        plot_div.close()
        print("Plot saved as:", file_name)
        return plot_trajectories



def plot_training_history(history, criterion='val_loss', b=0, show=True, wd='', filename=""):
    stopping_point = np.argmin(history.history[criterion])
    fig = plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'][b:],
             label='Training loss (%s)' % np.round(np.min(history.history['loss']), 3))
    plt.plot(history.history['val_loss'][b:],
             label='Validation loss (%s)' % np.round(np.min(history.history['val_loss']), 3))
    plt.axvline(stopping_point, linestyle='--', color='red', label='Early stopping point')
    plt.grid(axis='y', linestyle='dashed', which='major', zorder=0)
    plt.xlabel('Training epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, 'history_plot' + filename + '.pdf')
        plot_div = matplotlib.backends.backend_pdf.PdfPages(file_name)
        fig.subplots_adjust(top=0.92)
        plot_div.savefig(fig)
        plot_div.close()
        print("Plot saved as:", file_name)



def add_geochrono(Y1, Y2, max_ma, min_ma):
    series = -np.array([298.9, 272.95, 259.51, 251.902, 247.2, 237, 201.3, 174.1, 163.5, 145, 100.5, 66, 56, 33.9,
                        23.03, 5.33, 2.58, 0.0117])
    cols = np.array(["#e87864", "#f68d76", "#f9b4a3", "#a05da5", "#b282ba", "#bc9dca", "#00b4eb", "#71cfeb", "#abe1fa",
                     "#A0C96D", "#BAD25F", "#F8B77D", "#FAC18A", "#FBCC98", "#FFED00", "#FFF7B2", "#FFF1C4", "#FEF6F2"])
    names = np.array(["E. P", "M. P", "L. Permian", "E. T", "M. Triassic", "Late Triassic", "E. Jurassic", "M. J",
                      "L. J", "Early K", "L. K", "Paleocene", "Eocene", "Oligocene", "Miocene", "Pli", "Ple", ""])
    for i in range(len(series)-1):
        if series[i] >= max_ma and series[i+1] < min_ma:
            plt.text(0.5*(series[i] + series[i+1]), 0.5*(Y2 + Y1), names[i], ha="center", va="center", fontsize=14)
            plt.fill_between(x=np.array([series[i], series[i + 1]]), y1=Y1, y2=Y2, color=cols[i], edgecolor="black")
        if series[i] < max_ma < series[i+1]:
            plt.text(0.5*(max_ma + series[i+1]), 0.5*(Y2 + Y1), names[i], ha="center", va="center", fontsize=14)
            plt.fill_between(x=np.array([series[i], series[i + 1]]), y1=Y1, y2=Y2, color=cols[i], edgecolor="black")
        if series[i] < min_ma < series[i+1]:
            plt.text(0.5 * (series[i] + min_ma), 0.5 * (Y2 + Y1), names[i], ha="center", va="center", fontsize=14)
            plt.fill_between(x=np.array([series[i], series[i + 1]]), y1=Y1, y2=Y2, color=cols[i], edgecolor="black")
        else:
            pass


def add_geochrono_no_labels(Y1, Y2, max_ma, min_ma):
    series = -np.array([298.9, 272.95, 259.51, 251.902, 247.2, 237, 201.3, 174.1, 163.5, 145, 100.5, 66, 56, 33.9,
                        23.03, 5.33, 2.58, 0.0117])
    cols = np.array(["#e87864", "#f68d76", "#f9b4a3", "#a05da5", "#b282ba", "#bc9dca", "#00b4eb", "#71cfeb", "#abe1fa",
                     "#A0C96D", "#BAD25F", "#F8B77D", "#FAC18A", "#FBCC98", "#FFED00", "#FFF7B2", "#FFF1C4", "#FEF6F2"])
    for i in range(len(series)-1):
        if series[i] >= max_ma and series[i+1] < min_ma:
            plt.fill_between(x=np.array([series[i], series[i + 1]]), y1=Y1, y2=Y2, color=cols[i], edgecolor="black")
        if series[i] < max_ma < series[i+1]:
            plt.fill_between(x=np.array([series[i], series[i + 1]]), y1=Y1, y2=Y2, color=cols[i], edgecolor="black")
        if series[i] < min_ma < series[i+1]:
            plt.fill_between(x=np.array([series[i], series[i + 1]]), y1=Y1, y2=Y2, color=cols[i], edgecolor="black")
        else:
            pass


def add_pt_events(height):
    plt.fill_between(x=np.array([-252]), y1=0, y2=height, color="red", alpha=0.3)
    plt.fill_between(x=np.array([-234, -232]), y1=0, y2=height, color="red", alpha=0.3)
    plt.fill_between(x=np.array([-201.3]), y1=0, y2=height, color="red", alpha=0.3)
    plt.text(-251.5, 0.05, 'PTME', fontsize=14.5, color='red')
    plt.text(-231.5, 0.05, 'CPE', fontsize=14.5, color='red')
    plt.text(-200.8, 0.05, 'TJME', fontsize=14.5, color='red')




def get_bins(test_features, empirical_features, indx, n_bins, start=0):
    stop = np.max([np.max([test_features[:, :, indx].flatten()]), np.max(empirical_features[:, indx])])
    bins = np.linspace(start=0, stop=stop, num=n_bins)
    return bins

def plot_feature_hists(test_features,
                       empirical_features,
                       show=False,
                       wd="",
                       output_name="Feature_plot",
                       n_bins=30,
                       features_names=None,
                       log_occurrences=False
                       ):

    fig = plt.figure(figsize=(12, 7), layout="constrained")

    ax1 = fig.add_subplot(331)
    bins = get_bins(test_features, empirical_features, 0, n_bins)
    plt.hist(test_features[:, :, 0].flatten(), bins=bins, alpha=0.5, density=True, log=True, color='C1')
    plt.hist(empirical_features[:, 0], bins=bins, alpha=0.5, density=True, log=True, color='C0')
    plt.xlabel("Taxa")
    plt.ylabel("Frequency")
    c0 = mpatches.Patch(color='C0', label="Empirical feature")
    c1 = mpatches.Patch(color='C1', label='Mean simulated feature')
    plt.legend(handles=[c0, c1])
    ax1.annotate("A", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax2 = fig.add_subplot(332)
    bins = get_bins(test_features, empirical_features, 1, n_bins)
    plt.hist(test_features[:, :, 1].flatten()+1, bins=bins, alpha=0.5, density=True, log=True, color='C1')
    plt.hist(empirical_features[:, 1], bins=bins, alpha=0.5, density=True, log=True, color='C0')
    if log_occurrences:
        plt.xscale("log")
    plt.xlabel("Occurrences")
    plt.ylabel("Frequency")
    ax2.annotate("B", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax3 = fig.add_subplot(333)
    bins = get_bins(test_features, empirical_features, 2, n_bins)
    plt.hist(test_features[:, :, 2].flatten(), bins=bins, alpha=0.5, density=True, log=True, color='C1')
    plt.hist(empirical_features[:, 2], bins=bins, alpha=0.5, density=True, log=True, color='C0')
    plt.xlabel("Singletons")
    plt.ylabel("Frequency")
    ax3.annotate("C", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax4 = fig.add_subplot(334)
    bins = get_bins(test_features, empirical_features, 3, n_bins)
    plt.hist(test_features[:, :, 3].flatten(), bins=bins, alpha=0.5, density=True, log=True, color='C1')
    plt.hist(empirical_features[:, 3], bins=bins, alpha=0.5, density=True, log=True, color='C0')
    plt.xlabel("Endemic taxa")
    plt.ylabel("Frequency")
    ax4.annotate("D", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax5 = fig.add_subplot(335)
    bins = get_bins(test_features, empirical_features, 5, n_bins)
    plt.hist(test_features[:, :, 5].flatten(), bins=bins, alpha=0.5, density=True, log=True, color='C1')
    plt.hist(empirical_features[:, 5], bins=bins, alpha=0.5, density=True, log=True, color='C0')
    plt.xlabel("Range-through diversity")
    plt.ylabel("Frequency")
    ax5.annotate("E", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax6 = fig.add_subplot(336)
    bins = get_bins(test_features, empirical_features,6, n_bins)
    plt.hist(test_features[:, :, 6].flatten(), bins=bins, alpha=0.5, density=True, log=True, color='C1')
    plt.hist(empirical_features[:, 6], bins=bins, alpha=0.5, density=True, log=True, color='C0')
    plt.xlabel("Sampled localities")
    plt.ylabel("Frequency")
    ax6.annotate("F", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    if features_names is not None:
        ax7 = fig.add_subplot(337)
        indx = np.array([i for i in range(len(features_names)) if "n_species_" in features_names[i]])
        bins = get_bins(test_features, empirical_features, indx, n_bins)
        plt.hist(test_features[:, :, indx].flatten(), bins=bins, alpha=0.5, density=True, log=True, color='C1')
        plt.hist(empirical_features[:, indx].flatten(), bins=bins, alpha=0.5, density=True, log=True, color='C0')
        plt.xlabel("Taxa per region")
        plt.ylabel("Frequency")
        ax7.annotate("G", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

        ax8 = fig.add_subplot(338)
        indx = np.array([i for i in range(len(features_names)) if "n_occs_" in features_names[i]])
        bins = get_bins(test_features, empirical_features, indx, n_bins)
        plt.hist(test_features[:, :, indx].flatten()+1, bins=bins, alpha=0.5, density=True, log=True, color='C1')
        plt.hist(empirical_features[:, indx].flatten(), bins=bins, alpha=0.5, density=True, log=True, color='C0')
        if log_occurrences:
            plt.xscale("log")
        plt.xlabel("Occurrences per region")
        plt.ylabel("Frequency")
        ax8.annotate("H", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

        ax9 = fig.add_subplot(339)
        indx = np.array([i for i in range(len(features_names)) if "n_locs_" in features_names[i]])
        bins = get_bins(test_features, empirical_features, indx, n_bins)
        plt.hist(test_features[:, :, indx].flatten(), bins=bins, alpha=0.5, density=True, log=True, color='C1')
        plt.hist(empirical_features[:, indx].flatten(), bins=bins, alpha=0.5, density=True, log=True, color='C0')
        plt.xlabel("Localities per region")
        plt.ylabel("Frequency")
        ax9.annotate("I", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)


    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, output_name + ".pdf")
        plot_errors = matplotlib.backends.backend_pdf.PdfPages(file_name)
        plot_errors.savefig(fig)
        plot_errors.close()


def plot_all_models(data_wd, loaded_models, present_diversity, clade_name, output_wd,
                    time_bins, min_age=0, n_predictions=1, scaling=None,
                    prediction_color="b", alpha=0.5, replicates=1,
                    ):
    # run predictions across all models
    fig = plt.figure(figsize=(12, 8))

    predictions = []

    for model_i in range(len(loaded_models)):
        model = loaded_models[model_i]['model']
        feature_rescaler = loaded_models[model_i]['feature_rescaler']

        for replicate in range(1, replicates + 1):
            features, info = prep_dd_input(data_wd,
                                              bin_duration_file='t_bins.csv',  # from old to recent, array of shape (t)
                                              locality_file='%s_localities.csv' % replicate,  # array of shape (a, t)
                                              locality_dir='Locality',
                                              taxon_dir="Species_occurrences",
                                              hr_time_bins=time_bins,  # array of shape (t)
                                              rescale_by_n_bins=True,
                                              no_age_u=True,
                                              replicate=replicate,
                                              present_diversity=present_diversity,
                                              debug=False)

            # from recent to old
            plot_time_axis = np.sort(time_bins) + min_age

            print_update("Running replicate n. %s" % replicate)

            # from recent to old
            pred_div = predict(features, model, feature_rescaler, n_predictions=n_predictions, dropout=False)

            pred = np.mean(np.exp(pred_div) - 1, axis=0)
            if scaling == "1-mean":
                den = np.mean(pred)
            elif scaling == "first-bin":
                den = pred[-1]
            else:
                den = 1

            pred /= den

            plt.step(-plot_time_axis,  # pred,
                     [pred[0]] + list(pred),
                     label="Mean prediction",
                     linewidth=2,
                     c=prediction_color,
                     alpha=alpha)

            predictions.append(pred)

    predictions = np.array(predictions)
    # print(predictions)

    add_geochrono(-0.025 * np.max(predictions), -0.1 * np.max(predictions), max_ma=-np.max(time_bins), min_ma=-np.min(time_bins))
    plt.ylim(bottom=-0.1 * np.max(predictions), top=1.05 * np.max(predictions))
    add_pt_events(height=2.5)
    plt.xlim(-np.max(time_bins), 3)
    plt.ylabel("Diversity", fontsize=15)
    plt.xlabel("Time (Ma)", fontsize=15)
    fig.show()
    file_name = os.path.join(output_wd, clade_name + ".pdf")
    dd_plot = matplotlib.backends.backend_pdf.PdfPages(file_name)
    dd_plot.savefig(fig)
    dd_plot.close()
    print("\nPlot saved as:", file_name)
    return features, predictions



def features_through_time(features_names, time_bins, sim_features, empirical_features, wd):
    for i in range(len(features_names)):
        # retrieve simulated features for plotting
        n_feat = np.mean(sim_features[:, :, i], axis=0)
        n_feat = np.insert(n_feat, -len(n_feat), values=n_feat[0])
        feat_01 = np.percentile(sim_features[:, :, i], q=1, axis=0)
        feat_01 = np.insert(feat_01, -len(feat_01), values=feat_01[0])
        feat_99 = np.percentile(sim_features[:, :, i], q=99, axis=0)
        feat_99 = np.insert(feat_99, -len(feat_99), values=feat_99[0])

        feat_05 = np.percentile(sim_features[:, :, i], q=5, axis=0)
        feat_05 = np.insert(feat_05, -len(feat_05), values=feat_05[0])
        feat_95 = np.percentile(sim_features[:, :, i], q=95, axis=0)
        feat_95 = np.insert(feat_95, -len(feat_95), values=feat_95[0])



        fig = plt.figure(figsize=(12, 8))

        emp_feat = empirical_features[:,i] + 0
        emp_feat = np.insert(emp_feat, -len(emp_feat), values=emp_feat[0])
        plt.step(-time_bins,
                 emp_feat,
                 label="Empirical feature",
                 linewidth=2,
                 color="C0")

        plt.step(-time_bins,
                 n_feat,
                 label="Simulated feature",
                 linewidth=2,
                 color="C1")

        plt.fill_between(-time_bins,
                         feat_01,
                         feat_99,
                         linewidth=2,
                         step="pre",
                         alpha=0.2,
                         color="C1")

        plt.fill_between(-time_bins,
                         feat_05,
                         feat_95,
                         linewidth=2,
                         step="pre",
                         alpha=0.2,
                         color="C1")

        plt.ylabel(features_names[i], fontsize=15)
        plt.xlabel("Time (Ma)", fontsize=15)
        c0 = mpatches.Patch(color='C0', label="Empirical feature")
        c1 = mpatches.Patch(color='C1', label='Mean simulated feature')
        plt.legend(handles=[c0, c1])
        file_name = os.path.join(wd, features_names[i] + "_through_time.pdf")
        plot = matplotlib.backends.backend_pdf.PdfPages(file_name)
        plot.savefig(fig)
        plot.close()
        plt.close()



def features_pca(features_names, sim_features, empirical_features, wd, instance_acccuracy=None):
    try:
        from sklearn.decomposition import PCA

        # print("sim_features.shape", sim_features.shape) #= ('rep', 'time bin', 'feature')

        pca = PCA(n_components=2)

        flattened_features = sim_features.reshape((-1, sim_features.shape[1] * sim_features.shape[2]))
        flattened_features_rescale = flattened_features / (1 + np.std(flattened_features, axis=0))
        pca.fit(flattened_features_rescale)

        pca_embed_features = pca.transform(flattened_features_rescale)
        pca_embed_empirical = pca.transform(
            empirical_features.flatten().reshape(1, -1) / (1 + np.std(flattened_features, axis=0)))


        if instance_acccuracy is not None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
            p = ax.scatter(pca_embed_features[:, 0], pca_embed_features[:, 1], c=instance_acccuracy[:, 0], cmap='viridis',
                           label="R^2")
            fig.colorbar(p, ax=ax, orientation='vertical', label="R^2")
            c0 = 'gray'
            c1 = 'darkcyan'

        else:
            fig = plt.figure(figsize=(8, 8))
            plt.scatter(pca_embed_features[:, 0], pca_embed_features[:, 1], c='C1')
            c0 = 'C0'
            c1 = 'C1'

        plt.axvline(pca_embed_empirical[:, 0], linestyle='--', color=c0)
        plt.axhline(pca_embed_empirical[:, 1], linestyle='--', color=c0)
        plt.scatter(pca_embed_empirical[:, 0], pca_embed_empirical[:, 1], c=c0)

        plt.ylabel("PCA2", fontsize=15)
        plt.xlabel("PCA1", fontsize=15)
        c0 = mpatches.Patch(color=c0, label="Empirical feature")
        c1 = mpatches.Patch(color=c1, label='Mean simulated feature')
        plt.legend(handles=[c0, c1])
        file_name = os.path.join(wd, "feature_pca.pdf")
        plot = matplotlib.backends.backend_pdf.PdfPages(file_name)
        plot.savefig(fig)
        plot.close()
        plt.close()

        # plot vs MSE
        if instance_acccuracy is not None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
            p = ax.scatter(pca_embed_features[:, 0], pca_embed_features[:, 1], c=instance_acccuracy[:, 1], cmap='viridis_r',
                           label="MSE")
            fig.colorbar(p, ax=ax, orientation='vertical', label="MSE")
            c0 = 'gray'
            c1 = 'darkcyan'


            plt.axvline(pca_embed_empirical[:, 0], linestyle='--', color=c0)
            plt.axhline(pca_embed_empirical[:, 1], linestyle='--', color=c0)
            plt.scatter(pca_embed_empirical[:, 0], pca_embed_empirical[:, 1], c=c0)

            plt.ylabel("PCA2", fontsize=15)
            plt.xlabel("PCA1", fontsize=15)
            c0 = mpatches.Patch(color=c0, label="Empirical feature")
            c1 = mpatches.Patch(color=c1, label='Mean simulated feature')
            plt.legend(handles=[c0, c1])
            file_name = os.path.join(wd, "feature_pca_MSE.pdf")
            plot = matplotlib.backends.backend_pdf.PdfPages(file_name)
            plot.savefig(fig)
            plot.close()
            plt.close()


        if instance_acccuracy is not None:
            res = np.hstack((pca_embed_features, instance_acccuracy))
            x = np.ones(4) * np.nan
            x[:2] = pca_embed_empirical
            res = np.vstack((x, res))
            res_pd = pd.DataFrame(res, columns=["PCA1", "PCA2", "R2", "MSE"])
            res_pd.to_csv(file_name.replace(".pdf", ".csv"))

    except ImportError:
        print("Cannot plot feature PCA, please install scikit-learn")
        print("https://scikit-learn.org/stable/install.html")
