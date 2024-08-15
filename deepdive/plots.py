import copy
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


# run imports from plotting file before running
def plot_properties(fossil_sim, sim, show=False):
    fig = plt.figure(figsize=(20, 10))

    fig.add_subplot(241)
    plt.plot(-fossil_sim.mid_time_bins, count_species(sim), '-')
    plt.xlabel("Time (Ma)")
    plt.ylabel("Number of Species")
    plt.gca().set_title("Species", fontweight="bold", fontsize=10)
    plt.grid(True)

    fig.add_subplot(242)
    plt.plot(-fossil_sim.mid_time_bins, count_occurrences(sim), '-', color="teal")
    plt.xlabel("Time (Ma)")
    plt.ylabel("Number of Occurrences")
    plt.gca().set_title("Occurrences", fontweight="bold", fontsize=10)
    plt.grid(True)

    fig.add_subplot(243)
    plt.plot(-fossil_sim.mid_time_bins, count_sampled_localities(sim), '-', color="orange")
    plt.xlabel("Time (Ma)")
    plt.ylabel("Number of Localities")
    plt.gca().set_title("Sampled Localities", fontweight="bold", fontsize=10)
    plt.grid(True)


    fig.add_subplot(244)
    plt.plot(-fossil_sim.mid_time_bins, count_singletons(sim), '-', color="red")
    plt.xlabel("Time (Ma)")
    plt.ylabel("Singletons (no. species)")
    plt.gca().set_title("Number of Singletons", fontweight="bold", fontsize=10)
    plt.grid(True)

    fig.add_subplot(245)
    plt.plot(-fossil_sim.mid_time_bins, count_endemics(sim), '-', color="green")
    plt.xlabel("Time (Ma)")
    plt.ylabel("Endemic Diversity (no. species)")
    plt.gca().set_title("Endemic Diversity", fontweight="bold", fontsize=10)
    plt.grid(True)

    fig.add_subplot(246)
    plt.plot(-fossil_sim.mid_time_bins, time_bins_duration(sim), '-', color="purple")
    plt.xlabel("Time (Ma)")
    plt.ylabel("Time bins duration")
    plt.gca().set_title("Time Bin Duration", fontweight="bold", fontsize=10)
    plt.grid(True)

    fig.add_subplot(247)
    plt.plot(-fossil_sim.mid_time_bins, get_range_through_diversity(sim), '-', color="darkred")
    plt.xlabel("Time (Ma)")
    plt.ylabel("Range-through Diversity (no. species)")
    plt.gca().set_title("Range-through Diversity", fontweight="bold", fontsize=10)
    plt.grid(True)

    if show:
        fig.show()
    else:
        file_name = "feature_plots.pdf"
        plot_div = matplotlib.backends.backend_pdf.PdfPages(file_name)
        fig.subplots_adjust(top=0.92)
        plot_div.savefig(fig)
        plot_div.close()
        print("Plot saved as:", file_name)


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


def plot_prediction(predicted=None,
                    true=None,
                    mid_time_bins=None,
                    lower=None,
                    upper=None,
                    sampled=None,
                    xlimit=None,
                    sqs_6=None,
                    index=0,
                    log=False,
                    show=True,
                    wd='',
                    name='',
                    title=None):
    if log:
        def transf(x): return x
    else:
        def transf(x): return np.exp(x)

    fig = plt.figure(figsize=(8, 4))
    if true is not None:
        plt.plot(mid_time_bins, transf(true)[index][::-1], linestyle='-', label="true Y")
    if lower is not None:
        print(transf(lower)[index][::-1].reshape(predicted[index].shape[0]))
        plt.fill_between(x=mid_time_bins,
                         y1=transf(lower)[index][::-1].reshape(predicted[index].shape[0]),
                         y2=transf(upper)[index][::-1].reshape(predicted[index].shape[0]),
                         color="orange", alpha=0.5)
        print(transf(upper)[index][::-1].reshape(predicted[index].shape[0]))
    plt.plot(mid_time_bins, transf(predicted)[index][::-1].reshape(predicted[index].shape[0]), '-',
             label="Predicted")
    plt.plot(mid_time_bins, transf(sampled)[index][::-1], '-', label="Sampled")
    plt.xlim(xlimit)
    # plt.plot(mid_time_bins, transf(range_through)[index][::-1], '-', label="range-through Y")
    if sqs_6 is not None:
        sqs_6 = np.array(sqs_6)
        sqs_6 = sqs_6[index, :]
        plt.plot(mid_time_bins, transf(sqs_6)[::-1], '-', label="SQS=0.6")
    plt.xlabel("Time (Ma)")
    plt.ylabel("Diversity")
    if title is None:
        plt.gca().set_title("Simulation n. %s" % index, fontweight="bold", fontsize=12)
    else:
        plt.gca().set_title(title, fontweight="bold", fontsize=12)
    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, "prediction_plots_%s.pdf" % name)
        plot_div = matplotlib.backends.backend_pdf.PdfPages(file_name)
        fig.subplots_adjust(top=0.92)
        plot_div.savefig(fig)
        plot_div.close()
        print("Plot saved as:", file_name)


def plot_dd_prediction(predicted=None,
                       true=None,
                       mid_time_bins=None,
                       lower=None,
                       upper=None,
                       vert=None,
                       sampled=None,
                       sqs_6=None,
                       index=0,
                       exp_transform=False,
                       log10_transform=False,
                       rescale01=False,
                       show=True,
                       wd='',
                       name='',
                       max_age=None,
                       min_age=None,
                       title=None,
                       return_plt=False):

    predicted_local = copy.deepcopy(predicted)
    upper_local = copy.deepcopy(upper)
    lower_local = copy.deepcopy(lower)
    sampled_local = copy.deepcopy(sampled)

    if exp_transform:
        def transf(x): return np.exp(x)
    elif log10_transform:
        def transf(x):
            return np.log10(x)
    else:
        def transf(x):
            return x

    if rescale01:
        # if upper_local is not None:
        #     m = np.min(lower_local)
        #     upper_local = upper_local - m
        #     lower_local = lower_local - m
        #     den = np.max(upper_local)
        #     upper_local = upper_local / den
        #     lower_local = lower_local / den
        # else:
        m = np.min(predicted_local)
        den = np.max(predicted_local)
        predicted_local = predicted_local - m
        predicted_local = predicted_local / den
        if sampled_local is not None:
            sampled_local = sampled_local / np.max(sampled_local)
        if upper_local is not None:
            upper_local = upper_local / den
            lower_local = lower_local / den

    if max_age is not None:
        predicted_local[mid_time_bins < -abs(max_age)] = None
        if upper_local is not None:
            lower_local[mid_time_bins < -abs(max_age)] = None
            upper_local[mid_time_bins < -abs(max_age)] = None
        if sampled_local is not None:
            sampled_local[mid_time_bins < -abs(max_age)] = None
    if min_age is not None:
        predicted_local[mid_time_bins > -abs(min_age)] = None
        if sampled_local is not None:
            sampled_local[mid_time_bins > -abs(min_age)] = None
        if upper_local is not None:
            lower_local[mid_time_bins > -abs(min_age)] = None
            upper_local[mid_time_bins > -abs(min_age)] = None
    fig = plt.figure(figsize=(8, 4))
    if true is not None:
        plt.plot(mid_time_bins, transf(true), linestyle='-', label="True")
    if lower_local is not None:
        print(transf(lower_local).reshape(predicted_local.shape[0]))
        plt.fill_between(x=mid_time_bins,
                         y1=transf(lower_local).reshape(predicted_local.shape[0]),
                         y2=transf(upper_local).reshape(predicted_local.shape[0]),
                         color="orange", alpha=0.5)
        print(transf(upper_local)[::-1].reshape(predicted_local.shape[0]))
    if sampled_local is not None:
        plt.plot(mid_time_bins, transf(sampled_local), '-', label="Sampled", color='#636363')
    plt.plot(mid_time_bins, transf(predicted_local).reshape(predicted_local.shape[0]), '-',
             label="Predicted", color="orange")
    # plt.plot(mid_time_bins, transf(range_through)[index][::-1], '-', label="range-through Y")
    if sqs_6 is not None:
        sqs_6 = np.array(sqs_6)
        sqs_6 = sqs_6[index, :]
        plt.plot(mid_time_bins, transf(sqs_6), '-', label="SQS=0.6")
    plt.xlabel("Time (Ma)")
    if rescale01:
        plt.ylabel("Relative diversity")
    else:
        plt.ylabel("Diversity")
    plt.legend(loc='upper left')
    plt.grid(True)

    if vert is not None:
        plt.axvline(x=-vert[0, 1], linestyle="--", color="red")
        plt.axvline(x=-vert[0, 2], linestyle="--", color="red")

    if log10_transform:
        if rescale01:
            ylt = np.array([0.1, 0.25, 0.5, 0.75, 1, 1.5])
        else:
            ylt = np.array([1, 10, 100, 500, 1000, 5000, 10000, 50000])
        plt.yticks(np.log10(ylt), labels=ylt)

    if title is None:
        pass
    else:
        plt.gca().set_title(title, fontweight="bold", fontsize=12)
    if max_age is not None:
        plt.xlim([-abs(max_age), 0])
    if show:
        fig.show()
    elif return_plt:
        return plt
    else:
        file_name = os.path.join(wd, "DeepDive_prediction_%s.pdf" % name)
        plot_div = matplotlib.backends.backend_pdf.PdfPages(file_name)
        fig.subplots_adjust(top=0.92)
        plot_div.savefig(fig)
        plot_div.close()
        print("Plot saved as:", file_name)


def plot_true_sim(true=None,
                  index=0,
                  log=False,
                  mid_time_bins=None,
                  show=True,
                  wd='',
                  name=''):
    if log:
        def transf(x): return x
    else:
        def transf(x): return np.exp(x)

    fig = plt.figure(figsize=(10, 15))
    plt.subplots_adjust(hspace=0.7)

    fig.add_subplot(231)
    plt.plot(mid_time_bins, transf(true)[index][::-1], '-', label="true Y")
    plt.xlabel("Time (Ma)")
    plt.ylabel("Diversity")
    plt.grid(True)

    fig.add_subplot(232)
    plt.plot(mid_time_bins, transf(true)[index+1][::-1], '-', label="true Y")
    plt.xlabel("Time (Ma)")
    plt.ylabel("Diversity")
    plt.grid(True)

    fig.add_subplot(233)
    plt.plot(mid_time_bins, transf(true)[index+2][::-1], '-', label="true Y")
    plt.xlabel("Time (Ma)")
    plt.ylabel("Diversity")
    plt.grid(True)

    fig.add_subplot(234)
    plt.plot(mid_time_bins, transf(true)[index+3][::-1], '-', label="true Y")
    plt.xlabel("Time (Ma)")
    plt.ylabel("Diversity")
    plt.grid(True)

    fig.add_subplot(235)
    plt.plot(mid_time_bins, transf(true)[index+4][::-1], '-', label="true Y")
    plt.xlabel("Time (Ma)")
    plt.ylabel("Diversity")
    plt.grid(True)

    fig.add_subplot(236)
    plt.plot(mid_time_bins, transf(true)[index+5][::-1], '-', label="true Y")
    plt.xlabel("Time (Ma)")
    plt.ylabel("Diversity")
    plt.grid(True)

    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, "sim_t_%s.pdf" % name)
        plot_div = matplotlib.backends.backend_pdf.PdfPages(file_name)
        fig.subplots_adjust(top=0.92)
        plot_div.savefig(fig)
        plot_div.close()
        print("Plot saved as:", file_name)


def plot_scatter(predicted,
                 true,
                 log=False,
                 reshape=True,
                 show=True,
                 dd=False,
                 rt=False,
                 sqs4=False,
                 sqs6=False,
                 wd='',
                 name=''):
    if log:
        def transf(x): return x
    else:
        def transf(x): return np.exp(x)

    true = true.flatten()
    if dd or rt:
        predicted = predicted[:, :, 0].flatten()
    if sqs4 or sqs6:
        predicted = predicted.flatten()
    true_i = transf(true)
    if reshape:
        predicted_i = transf(predicted).reshape(predicted.shape[0])
    else:
        predicted_i = transf(predicted)
    # define x-axis ticks for plotting
    ticks = np.array([1, 10, 100, 1000, 5000])
    fig = plt.figure(figsize=(8, 4))
    plt.scatter(true_i, predicted_i, color='C3', alpha=0.1)
    # add ticks in log space with labels showing their value (not log transformed)
    plt.xticks(ticks=np.log(ticks), labels=ticks)
    plt.yticks(ticks=np.log(ticks), labels=ticks)
    plt.xlabel("True Diversity")
    if dd:
        plt.ylabel("DeepDive Diversity")
        plt.gca().set_title("True vs DeepDive diversity", fontweight="bold", fontsize=12)
    if rt:
        plt.ylabel("Range-through Diversity")
        plt.gca().set_title("True vs Range-through diversity", fontweight="bold", fontsize=12)
    if sqs4:
        plt.ylabel("SQS Diversity, Q=0.4")
        plt.gca().set_title("True vs SQS diversity", fontweight="bold", fontsize=12)
    if sqs6:
        plt.ylabel("SQS Diversity, Q=0.6")
        plt.gca().set_title("True vs SQS diversity", fontweight="bold", fontsize=12)
    m, c = np.polyfit(true_i, predicted_i, 1)
    plt.plot(true_i, m*true_i+c, color="orange")
    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, "scatter_plots_%s.pdf" % name)
        plot_div = matplotlib.backends.backend_pdf.PdfPages(file_name)
        fig.subplots_adjust(top=0.92)
        plot_div.savefig(fig)
        plot_div.close()
        print("Plot saved as:", file_name)


# PLOTS OF SIMULATION PARAMETERS
# Number of occurrences per species
def occ_per_sp(sim):
    n_occs_per_sp = np.einsum('sat -> s', sim['fossil_data'])
    plt.hist(n_occs_per_sp, bins=np.arange(n_occs_per_sp.min(), n_occs_per_sp.max()+1))
    plt.title('Histogram of number of occurrences per species')
    plt.xlabel('Number of occurrences per species')
    plt.ylabel('Frequency')
    plt.show()


# PLOTS OF DISTRIBUTION OF RESULTS
def plot_dist_res(without_age_uncert, with_age_uncert, sqs_q6_without=False, sqs_q6_with=False):
    n_keys = np.size(without_age_uncert, axis=1)

    for i in range(n_keys):
        a = without_age_uncert[:, i]
        b = with_age_uncert[:, i]
        sns.set_style('whitegrid')
        sns.kdeplot(np.array(a), bw=0.5)
        sns.kdeplot(np.array(b), bw=0.5)

    if sqs_q6_without:
        c = sqs_q6_without.mse
        sns.kdeplot(np.array(a), bw=0.5)
        sns.kdeplot(np.array(c), bw=0.5)
    if sqs_q6_with:
        d = sqs_q6_with.mse
        sns.kdeplot(np.array(b), bw=0.5)
        sns.kdeplot(np.array(d), bw=0.5)
    plt.show()


def plot_comp(results,
              show=False,
              min_x=None,
              max_x=None,
              wd=''):
    all_p = []
    all_t = []
    for i in range(len(results)):
        p = results[i]["n_sampled_species"]
        t = results[i]["n_species"]
        all_p.append(p)
        all_t.append(t)
    all_p = np.array(all_p)
    all_t = np.array(all_t)

    fig = plt.figure(figsize=(8, 4))
    comp = np.divide(all_p, all_t)
    sns.kdeplot(comp,  fill=True, color="mediumturquoise", clip=(min_x, max_x))
    plt.xlabel("Completeness")
    plt.ylabel("Frequency")
    plt.xlabel("Completeness")
    print("median completeness is ", np.median(comp))
    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, "completeness_density.pdf")
        plot_com = matplotlib.backends.backend_pdf.PdfPages(file_name)
        plot_com.savefig(fig)
        plot_com.close()
        print("Plot saved as:", file_name)


def plot_pres(results,
              show=False,
              min_x=None,
              max_x=None,
              wd=''):
    all_br = []
    all_occs = []
    for i in range(len(results)):
        br = results[i]["tot_br_length"]
        occs = results[i]["n_occurrences"]
        all_br.append(br)
        all_occs.append(occs)
    all_br = np.array(all_br)
    all_occs = np.array(all_occs)

    fig = plt.figure(figsize=(8, 4))
    p = np.divide(all_occs, all_br)
    sns.kdeplot(p,  fill=True, color="cornflowerblue", clip=(min_x, max_x))
    plt.xlabel("Preservation rate")
    plt.ylabel("Frequency")
    plt.xlabel("Preservation rate")
    print("median preservation rate is ", np.median(p))
    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, "preservation_density.pdf")
        plot_pre = matplotlib.backends.backend_pdf.PdfPages(file_name)
        plot_pre.savefig(fig)
        plot_pre.close()
        print("Plot saved as:", file_name)


def plot_richness(results,
                  name="",
                  show=False,
                  wd=''):
    fig = plt.figure(figsize=(8, 4))
    if name == "sampled":
        all_p = []
        for i in range(len(results)):
            p = results[i]["n_sampled_species"]
            all_p.append(p)
        species = np.array(all_p)
    if name == "true":
        all_t = []
        for i in range(len(results)):
            t = results[i]["n_species"]
            all_t.append(t)
        species = np.array(all_t)
    plt.hist(species, bins=100)
    plt.grid(True)
    plt.ylabel("Frequency")
    plt.xlabel("Number of Species"+name)
    plt.gca().set_title("Species Richness", fontweight="bold", fontsize=12)
    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, name+"sp_richness_hist.pdf")
        plot_rich = matplotlib.backends.backend_pdf.PdfPages(file_name)
        plot_rich.savefig(fig)
        plot_rich.close()
        print("Plot saved as:", file_name)


def plot_occs(results,
              name="",
              show=False,
              wd=''):
    occurrences = []
    for i in range(len(results)):
        occs = results[i]["n_occurrences"]
        occurrences.append(occs)
    occurrences = np.array(occurrences)

    fig = plt.figure(figsize=(8, 4))
    plt.hist(occurrences, bins=100)
    plt.grid(True)
    plt.ylabel("Frequency")
    plt.xlabel("Number of Occurrences"+name)
    plt.gca().set_title("Occurrences", fontweight="bold", fontsize=12)

    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, name+"occs_hist.pdf")
        plot_occurrences = matplotlib.backends.backend_pdf.PdfPages(file_name)
        plot_occurrences.savefig(fig)
        plot_occurrences.close()
        print("Plot saved as:", file_name)


def plot_occs_per_sp(results,
                     name="",
                     show=False,
                     wd=''):
    all_p = []
    all_occs = []
    for i in range(len(results)):
        p = results[i]["n_sampled_species"]
        occs = results[i]["n_occurrences"]
        all_occs.append(occs)
        all_p.append(p)
    occurrences = np.array(all_occs)
    species = np.array(all_p)

    fig = plt.figure(figsize=(8, 4))
    occs_per_sp = occurrences/species
    plt.hist(occs_per_sp, bins=100)
    plt.grid(True)
    plt.ylabel("Frequency")
    plt.xlabel("Number of Occurrences per Species"+name)
    plt.gca().set_title("Occurrences per Species", fontweight="bold", fontsize=12)
    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, name+"occs_hist.pdf")
        plot_occs_sp = matplotlib.backends.backend_pdf.PdfPages(file_name)
        plot_occs_sp.savefig(fig)
        plot_occs_sp.close()
        print("Plot saved as:", file_name)


def plot_loc_rates(sim,
                   index=0,
                   show=True,
                   wd='',
                   name=''):
    fig = plt.figure(figsize=(15, 10))
    plt.subplots_adjust(wspace=0.3,
                        hspace=0.3)

    fig.add_subplot(331)
    sim_0 = sim[index]
    sim_0['locality_rate'][sim_0['locality_rate'] == 0] = np.nan
    plt.plot(sim_0['locality_rate'][0])
    plt.plot(sim_0['locality_rate'][1])
    plt.plot(sim_0['locality_rate'][2])
    plt.plot(sim_0['locality_rate'][3])
    plt.plot(sim_0['locality_rate'][4])
    plt.xlabel("Time (Ma)", fontsize=15)
    plt.ylabel("Number of localities", fontsize=15)

    sim_1 = sim[index+1]
    sim_1['locality_rate'][sim_1['locality_rate'] == 0] = np.nan
    fig.add_subplot(332)
    plt.plot(sim_1['locality_rate'][0])
    plt.plot(sim_1['locality_rate'][1])
    plt.plot(sim_1['locality_rate'][2])
    plt.plot(sim_1['locality_rate'][3])
    plt.plot(sim_1['locality_rate'][4])
    plt.xlabel("Time (Ma)", fontsize=15)
    plt.ylabel("Number of localities", fontsize=15)

    sim_2 = sim[index+2]
    sim_2['locality_rate'][sim_2['locality_rate'] == 0] = np.nan
    fig.add_subplot(333)
    plt.plot(sim_2['locality_rate'][0])
    plt.plot(sim_2['locality_rate'][1])
    plt.plot(sim_2['locality_rate'][2])
    plt.plot(sim_2['locality_rate'][3])
    plt.plot(sim_2['locality_rate'][4])
    plt.xlabel("Time (Ma)", fontsize=15)
    plt.ylabel("Number of localities", fontsize=15)

    sim_3 = sim[index+3]
    sim_3['locality_rate'][sim_3['locality_rate'] == 0] = np.nan
    fig.add_subplot(334)
    plt.plot(sim_3['locality_rate'][0])
    plt.plot(sim_3['locality_rate'][1])
    plt.plot(sim_3['locality_rate'][2])
    plt.plot(sim_3['locality_rate'][3])
    plt.plot(sim_3['locality_rate'][4])
    plt.xlabel("Time (Ma)", fontsize=15)
    plt.ylabel("Number of localities", fontsize=15)

    sim_4 = sim[index+4]
    sim_4['locality_rate'][sim_4['locality_rate'] == 0] = np.nan
    fig.add_subplot(335)
    plt.plot(sim_4['locality_rate'][0])
    plt.plot(sim_4['locality_rate'][1])
    plt.plot(sim_4['locality_rate'][2])
    plt.plot(sim_4['locality_rate'][3])
    plt.plot(sim_4['locality_rate'][4])
    plt.xlabel("Time (Ma)", fontsize=15)
    plt.ylabel("Number of localities", fontsize=15)

    sim_5 = sim[index+5]
    sim_5['locality_rate'][sim_5['locality_rate'] == 0] = np.nan
    fig.add_subplot(336)
    plt.plot(sim_5['locality_rate'][0])
    plt.plot(sim_5['locality_rate'][1])
    plt.plot(sim_5['locality_rate'][2])
    plt.plot(sim_5['locality_rate'][3])
    plt.plot(sim_5['locality_rate'][4])
    plt.xlabel("Time (Ma)", fontsize=15)
    plt.ylabel("Number of localities", fontsize=15)

    sim_6 = sim[index+6]
    sim_6['locality_rate'][sim_6['locality_rate'] == 0] = np.nan
    fig.add_subplot(337)
    plt.plot(sim_6['locality_rate'][0])
    plt.plot(sim_6['locality_rate'][1])
    plt.plot(sim_6['locality_rate'][2])
    plt.plot(sim_6['locality_rate'][3])
    plt.plot(sim_6['locality_rate'][4])
    plt.xlabel("Time (Ma)", fontsize=15)
    plt.ylabel("Number of localities", fontsize=15)

    sim_7 = sim[index+7]
    sim_7['locality_rate'][sim_7['locality_rate'] == 0] = np.nan
    fig.add_subplot(338)
    plt.plot(sim_7['locality_rate'][0])
    plt.plot(sim_7['locality_rate'][1])
    plt.plot(sim_7['locality_rate'][2])
    plt.plot(sim_7['locality_rate'][3])
    plt.plot(sim_7['locality_rate'][4])
    plt.xlabel("Time (Ma)", fontsize=15)
    plt.ylabel("Number of localities", fontsize=15)

    sim_8 = sim[index+8]
    sim_8['locality_rate'][sim_8['locality_rate'] == 0] = np.nan
    fig.add_subplot(339)
    plt.plot(sim_8['locality_rate'][0])
    plt.plot(sim_8['locality_rate'][1])
    plt.plot(sim_8['locality_rate'][2])
    plt.plot(sim_8['locality_rate'][3])
    plt.plot(sim_8['locality_rate'][4])
    plt.xlabel("Time (Ma)", fontsize=15)
    plt.ylabel("Number of localities", fontsize=15)

    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, name+"loc_rate.pdf")
        plot_locs = matplotlib.backends.backend_pdf.PdfPages(file_name)
        plot_locs.savefig(fig)
        plot_locs.close()
        print("Plot saved as:", file_name)


# Density plots of R2 - deepdive and sqs
def plot_r2(r2_dd,
            r2_sqs,
            show=True,
            wd='',
            plot_type="Density",
            min_x=None,
            max_x=None,
            name=''):
    fig = plt.figure(figsize=(8, 4))
    if plot_type == "Density":
        sns.kdeplot(r2_dd,  fill=True, clip=(min_x, max_x))
        sns.kdeplot(r2_sqs,  fill=True, color="C3", clip=(min_x, max_x))
        plt.xlabel("$R^{2}$")
    if plot_type == "Violin":
        my_pal = {"#1f77b4", "C3"}
        ax = sns.violinplot(data=[r2_sqs, r2_dd], palette=my_pal,  fill=True, cut=0, split=True)
        ax.set_xticklabels(["DeepDive", "SQS"])
        plt.yscale("log")
    if plot_type == "Boxplot":
        ax = sns.boxplot(data=[r2_dd, r2_sqs])
        ax.set_xticklabels(["DeepDive", "SQS"])
        plt.yscale("log")
    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, name+plot_type+"_R2.pdf")
        plot_r2 = matplotlib.backends.backend_pdf.PdfPages(file_name)
        plot_r2.savefig(fig)
        plot_r2.close()
        print("Plot saved as:", file_name)


def plot_mse(rmse_dd,
             rmse_sqs,
             show=False,
             plot_type="Density",
             min_x=None,
             max_x=None,
             wd='',
             name=''):
    fig = plt.figure(figsize=(8, 4))
    if plot_type == "Density":
        sns.kdeplot(rmse_dd,  fill=True, clip=(min_x, max_x))
        sns.kdeplot(rmse_sqs,  fill=True, color="C3", clip=(min_x, max_x))
        plt.xscale("log")
        # add ticks in log space with labels showing their value (not log transformed)
        # plt.xticks(ticks=np.log(ticks_x), labels=ticks_x)
        plt.xlabel("Relative MSE")
    if plot_type == "Violin":
        my_pal = {"#1f77b4", "C3"}
        ax = sns.violinplot(data=[rmse_dd, rmse_sqs], palette=my_pal,  fill=True, cut=0, split=True)
        ax.set_xticklabels(["DeepDive", "SQS"])
        plt.yscale("log")
    if plot_type == "Boxplot":
        ax = sns.boxplot(data=[rmse_dd, rmse_sqs])
        ax.set_xticklabels(["DeepDive", "SQS"])
        plt.yscale("log")
    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, name+plot_type+"_MSE.pdf")
        plot_mse = matplotlib.backends.backend_pdf.PdfPages(file_name)
        plot_mse.savefig(fig)
        plot_mse.close()
        print("Plot saved as:", file_name)


def plot_durations(t,
                   results,
                   mse,
                   level,
                   show=False,
                   wd=''):
    fig = plt.figure(figsize=(8, 4))
    if level == "species":
        all_sp_d = []
        for i in range(len(t)):
            sp_duration = results[i]["tot_br_length"] / results[i]["n_species"]
            all_sp_d.append(sp_duration)
        plt.scatter(np.array(all_sp_d), mse, color="mediumpurple", alpha=0.5)
        plt.xlabel("Species duration")
    if level == "clade":
        all_c_d = []
        for i in range(len(t)):
            clade_duration = len(t[i][t[i] > 0])
            all_c_d.append(clade_duration)
        plt.scatter(np.array(all_c_d), mse, color="mediumseagreen", alpha=0.5)
        plt.xlabel("Clade duration")
    plt.grid(True)
    plt.ylabel("MSE")
    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, level + "_duration.pdf")
        plot_d = matplotlib.backends.backend_pdf.PdfPages(file_name)
        plot_d.savefig(fig)
        plot_d.close()
        print("Plot saved as:", file_name)


def plot_completeness(results,
                      mse,
                      show=False,
                      wd=''):
    fig = plt.figure(figsize=(8, 4))
    all_p = []
    all_t = []
    for i in range(len(results)):
        p = results[i]["n_sampled_species"]
        t = results[i]["n_species"]
        all_p.append(p)
        all_t.append(t)
    sampled = np.array(all_p)
    true = np.array(all_t)
    comp = np.divide(sampled, true)
    plt.scatter(comp, mse, color="mediumturquoise", alpha=0.5)
    plt.grid(True)
    plt.ylabel("MSE")
    plt.xlabel("Completeness")
    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, "completeness.pdf")
        plot_c = matplotlib.backends.backend_pdf.PdfPages(file_name)
        plot_c.savefig(fig)
        plot_c.close()
        print("Plot saved as:", file_name)


def plot_preservation(results,
                      mse,
                      show=False,
                      wd=''):
    fig = plt.figure(figsize=(8, 4))
    all_br = []
    all_occs = []
    for i in range(len(results)):
        br = results[i]["tot_br_length"]
        occs = results[i]["n_occurrences"]
        all_br.append(br)
        all_occs.append(occs)
    all_br = np.array(all_br)
    all_occs = np.array(all_occs)
    p = np.divide(all_occs, all_br)
    plt.scatter(p, mse, color="cornflowerblue", alpha=0.5)
    plt.grid(True)
    plt.ylabel("MSE")
    plt.xlabel("Preservation rate")
    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, "preservation.pdf")
        plot_p = matplotlib.backends.backend_pdf.PdfPages(file_name)
        plot_p.savefig(fig)
        plot_p.close()
        print("Plot saved as:", file_name)


def plot_dataset_size(results,
                      mse,
                      show=False,
                      wd=''):
    fig = plt.figure(figsize=(8, 4))
    all_p = []
    for i in range(len(results)):
        p = results[i]["n_sampled_species"]
        all_p.append(p)
    all_p = np.array(all_p)
    plt.scatter(all_p, mse, color="mediumorchid", alpha=0.5)
    plt.grid(True)
    plt.ylabel("MSE")
    plt.xlabel("Number of species")
    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, "dataset_size.pdf")
        plot_ds = matplotlib.backends.backend_pdf.PdfPages(file_name)
        plot_ds.savefig(fig)
        plot_ds.close()
        print("Plot saved as:", file_name)


def boxplot_mse_extinct_extant(mse,
                               t,
                               show=False,
                               wd=''):
    fig = plt.figure(figsize=(8, 4))
    extant_div = t[:, 0]  # true extant diversity
    index_extant = np.where(extant_div > 0)  # find which clades are extant
    index_extinct = np.where(extant_div == 0)  # find which clades are extinct
    extant_mse = mse[index_extant]  # find the mse values for each subset of simulations
    extinct_mse = mse[index_extinct]
    ax = sns.boxplot(data=[mse, extant_mse, extinct_mse])
    ax.set_xticklabels(["All", "Extant clade", "Extinct clade"])
    plt.ylabel("MSE")
    plt.yscale("log")
    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, "Extinct_extant_MSE.pdf")
        plot_mse_ee = matplotlib.backends.backend_pdf.PdfPages(file_name)
        plot_mse_ee.savefig(fig)
        plot_mse_ee.close()
        print("Plot saved as:", file_name)


def plot_comp_r2(results,
                 r2,
                 show=False,
                 wd=''):
    fig = plt.figure(figsize=(8, 4))
    all_p = []
    all_t = []
    for i in range(len(results)):
        p = results[i]["n_sampled_species"]
        t = results[i]["n_species"]
        all_p.append(p)
        all_t.append(t)
    sampled = np.array(all_p)
    true = np.array(all_t)
    comp = np.divide(sampled, true)
    plt.scatter(comp, r2, color="mediumturquoise", alpha=0.5)
    plt.grid(True)
    plt.ylabel("R2")
    plt.xlabel("Completeness")
    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, "completeness_r2.pdf")
        plot_c_r = matplotlib.backends.backend_pdf.PdfPages(file_name)
        plot_c_r.savefig(fig)
        plot_c_r.close()
        print("Plot saved as:", file_name)


# GENERATE FIGURES USED IN DEEPDIVE PAPER
def figure_2(true_dd,
             true_sqs,
             predicted_dd,
             predicted_sqs,
             r2_dd,
             r2_sqs,
             rmse_dd,
             rmse_sqs,
             show=False,
             wd='',
             name='',):

    fig = plt.figure(figsize=(12, 7), layout="constrained")

    ax1 = fig.add_subplot(221)
    def transf(x): return np.exp(x)
    true = true_dd.flatten()
    predicted = predicted_dd[:, :, 0].flatten()
    true_i = transf(true)
    predicted_i = transf(predicted).reshape(predicted.shape[0])
    ticks = np.array([10, 100, 1000, 5000])  # define x-axis ticks for plotting
    plt.scatter(true_i, predicted_i, alpha=0.1)
    plt.plot(true_i, 1 * true_i, color="black")
    plt.xticks(ticks=np.log(ticks), labels=ticks)  # add ticks
    plt.yticks(ticks=np.log(ticks), labels=ticks)
    plt.xlabel("True Diversity")
    plt.ylabel("DeepDive Diversity")
    ax1.annotate("A", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax2 = fig.add_subplot(222)
    def transf(x): return x
    true = true_sqs.flatten()
    predicted = predicted_sqs.flatten()
    true_i = transf(true)
    predicted_i = transf(predicted).reshape(predicted.shape[0])
    ticks = np.array([10, 100, 1000, 5000])  # define x-axis ticks for plotting
    plt.scatter(true_i, predicted_i, color='C3', alpha=0.1)
    plt.plot(true_i, 1 * true_i, color="black")
    plt.xticks(ticks=np.log(ticks), labels=ticks)  # add ticks
    plt.yticks(ticks=np.log(ticks), labels=ticks)
    plt.xlabel("True Diversity")
    plt.ylabel("SQS Diversity, Q=0.6")
    ax2.annotate("B", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax3 = fig.add_subplot(223)
    ax3 = sns.boxplot(data=[r2_dd, r2_sqs], palette={'C3', '#1f77b4'})
    ax3.set_xticklabels(["DeepDive", "SQS"])
    plt.ylabel("R$^{2}$")
    ax3.annotate("C", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax4 = fig.add_subplot(224)
    ax4 = sns.boxplot(data=[rmse_dd, rmse_sqs], palette={'C3', '#1f77b4'})
    ax4.set_xticklabels(["DeepDive", "SQS"])
    ax4.set_yscale("log")
    ax4.set_yticks([0.001, 0.01, 0.1, 1])
    plt.ylabel("rMSE")
    ax4.annotate("D", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, "figure_2.png")
        plt.savefig("figure_2.png", dpi=300)
        plt.close()
        print("Plot saved as:", file_name)


def figure_3(predicted=None,
             true=None,
             mid_time_bins=None,
             lower=None,
             upper=None,
             sampled=None,
             xlimit=None,
             sqs_6=None,
             index1=106,
             index2=999,
             index3=7,
             index4=232,
             log=False,
             show=True,
             wd='',
             name='',
             title=None):

    fig = plt.figure(figsize=(12, 7), layout="constrained")

    ax1 = fig.add_subplot(221)
    axA = ax1.twinx()
    def transf(x): return np.exp(x)
    if true is not None:
        ax1.plot(mid_time_bins, transf(true)[index1][::-1], linestyle='-', label="true Y", color="green")
    if lower is not None:
        print(transf(lower)[index1][::-1].reshape(predicted[index1].shape[0]))
        ax1.fill_between(x=mid_time_bins,
                         y1=transf(lower)[index1][::-1].reshape(predicted[index1].shape[0]),
                         y2=transf(upper)[index1][::-1].reshape(predicted[index1].shape[0]),
                         color="b", alpha=0.2)
        print(transf(upper)[index1][::-1].reshape(predicted[index1].shape[0]))
    ax1.plot(mid_time_bins, transf(predicted)[index1][::-1].reshape(predicted[index1].shape[0]), '-',
             label="Predicted", color="b")
    ax1.plot(mid_time_bins, transf(sampled)[index1][::-1], '-', label="Sampled", color="orange")
    plt.xlim(-50, 0)
    if sqs_6 is not None:
        sqs_6 = np.array(sqs_6)
        sqs_index1 = sqs_6[index1, :]
        axA.plot(mid_time_bins, transf(sqs_index1)[::-1], '-', label="SQS=0.6", color="C3")
    ax1.set_xlabel("Time (Ma)")
    ax1.set_ylabel("Diversity")
    axA.set_ylabel("SQS diversity")
    # plt.legend(loc='upper left')
    ax1.annotate("A", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax2 = fig.add_subplot(222)
    axB = ax2.twinx()
    def transf(x): return np.exp(x)
    if true is not None:
        ax2.plot(mid_time_bins, transf(true)[index2][::-1], linestyle='-', label="true Y", color="green")
    if lower is not None:
        print(transf(lower)[index2][::-1].reshape(predicted[index2].shape[0]))
        ax2.fill_between(x=mid_time_bins,
                         y1=transf(lower)[index2][::-1].reshape(predicted[index2].shape[0]),
                         y2=transf(upper)[index2][::-1].reshape(predicted[index2].shape[0]),
                         color="b", alpha=0.2)
        print(transf(upper)[index2][::-1].reshape(predicted[index2].shape[0]))
    ax2.plot(mid_time_bins, transf(predicted)[index2][::-1].reshape(predicted[index2].shape[0]), '-',
             label="Predicted", color="b")
    ax2.plot(mid_time_bins, transf(sampled)[index2][::-1], '-', label="Sampled", color="orange")
    plt.xlim(-50, 0)
    if sqs_6 is not None:
        sqs_6 = np.array(sqs_6)
        sqs_index2 = sqs_6[index2, :]
        axB.plot(mid_time_bins, transf(sqs_index2)[::-1], '-', label="SQS=0.6", color="C3")
    ax2.set_xlabel("Time (Ma)")
    ax2.set_ylabel("Diversity")
    axB.set_ylabel("SQS diversity")
    ax2.annotate("B", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax3 = fig.add_subplot(223)
    axC = ax3.twinx()
    def transf(x): return np.exp(x)
    if true is not None:
        ax3.plot(mid_time_bins, transf(true)[index3][::-1], linestyle='-', label="true Y", color="green")
    if lower is not None:
        print(transf(lower)[index3][::-1].reshape(predicted[index3].shape[0]))
        ax3.fill_between(x=mid_time_bins,
                         y1=transf(lower)[index3][::-1].reshape(predicted[index3].shape[0]),
                         y2=transf(upper)[index3][::-1].reshape(predicted[index3].shape[0]),
                         color="b", alpha=0.2)
        print(transf(upper)[index3][::-1].reshape(predicted[index3].shape[0]))
    ax3.plot(mid_time_bins, transf(predicted)[index3][::-1].reshape(predicted[index3].shape[0]), '-',
             label="Predicted", color="b")
    ax3.plot(mid_time_bins, transf(sampled)[index3][::-1], '-', label="Sampled", color="orange")
    plt.xlim(-80, 0)
    if sqs_6 is not None:
        sqs_6 = np.array(sqs_6)
        sqs_index3 = sqs_6[index3, :]
        axC.plot(mid_time_bins, transf(sqs_index3)[::-1], '-', label="SQS=0.6", color="C3")
    ax3.set_xlabel("Time (Ma)")
    ax3.set_ylabel("Diversity")
    axC.set_ylabel("SQS diversity")
    ax3.annotate("C", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax4 = fig.add_subplot(224)
    axD = ax4.twinx()
    def transf(x): return np.exp(x)
    if true is not None:
        ax4.plot(mid_time_bins, transf(true)[index4][::-1], linestyle='-', label="true Y", color="green")
    if lower is not None:
        print(transf(lower)[index4][::-1].reshape(predicted[index4].shape[0]))
        ax4.fill_between(x=mid_time_bins,
                         y1=transf(lower)[index4][::-1].reshape(predicted[index4].shape[0]),
                         y2=transf(upper)[index4][::-1].reshape(predicted[index4].shape[0]),
                         color="b", alpha=0.2)
        print(transf(upper)[index4][::-1].reshape(predicted[index4].shape[0]))
    ax4.plot(mid_time_bins, transf(predicted)[index4][::-1].reshape(predicted[index4].shape[0]), '-',
             label="Predicted", color="b")
    ax4.plot(mid_time_bins, transf(sampled)[index4][::-1], '-', label="Sampled", color="orange")
    plt.xlim(-100, 0)
    if sqs_6 is not None:
        sqs_6 = np.array(sqs_6)
        sqs_index4 = sqs_6[index4, :]
        axD.plot(mid_time_bins, transf(sqs_index4)[::-1], '-', label="SQS, Q=0.6", color="C3")
    ax4.set_xlabel("Time (Ma)")
    ax4.set_ylabel("Diversity")
    axD.set_ylabel("SQS diversity")
    ax4.annotate("D", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, "figure_3.pdf")
        plot_fig3 = matplotlib.backends.backend_pdf.PdfPages(file_name)
        plot_fig3.savefig(fig)
        plot_fig3.close()
        print("Plot saved as:", file_name)


def figure_4(truth,
             results,
             mse,
             show=False,
             name="",
             wd=''):
    fig = plt.figure(figsize=(10, 15), layout="constrained")

    all_p = []
    all_t = []
    all_br = []
    all_occs = []
    for i in range(len(results)):
        p = results[i]["n_sampled_species"]
        t = results[i]["n_species"]
        br = results[i]["tot_br_length"]
        occs = results[i]["n_occurrences"]
        all_br.append(br)
        all_occs.append(occs)
        all_p.append(p)
        all_t.append(t)
    all_br = np.array(all_br)
    all_occs = np.array(all_occs)
    all_p = np.array(all_p)
    all_t = np.array(all_t)

    all_sp_d = []
    all_c_d = []
    for i in range(len(truth)):
        sp_duration = results[i]["tot_br_length"] / results[i]["n_species"]
        clade_duration = len(truth[i][truth[i] > 0])
        all_sp_d.append(sp_duration)
        all_c_d.append(clade_duration)
    all_sp_d = np.array(all_sp_d)
    all_c_d = np.array(all_c_d)

    ax1 = fig.add_subplot(321)
    comp = np.divide(all_p, all_t)
    plt.scatter(comp, mse, color="mediumturquoise", alpha=0.5)
    plt.grid(True)
    plt.ylabel("rMSE")
    plt.xlabel("Completeness")
    ax1.annotate("A", xy=(-0.1, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax2 = fig.add_subplot(322)
    p = np.divide(all_occs, all_br)
    plt.scatter(p, mse, color="cornflowerblue", alpha=0.5)
    plt.grid(True)
    plt.ylabel("rMSE")
    plt.xlabel("Preservation rate")
    ax2.annotate("B", xy=(-0.1, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax3 = fig.add_subplot(323)
    plt.scatter(all_p, mse, color="mediumorchid", alpha=0.5)
    plt.grid(True)
    plt.ylabel("rMSE")
    plt.xlabel("Number of sampled species")
    ax3.annotate("C", xy=(-0.1, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax4 = fig.add_subplot(324)
    extant_div = truth[:, 0]  # true extant diversity
    index_extant = np.where(extant_div > 0)  # find which clades are extant
    index_extinct = np.where(extant_div == 0)  # find which clades are extinct
    extant_mse = mse[index_extant]  # find the mse values for each subset of simulations
    extinct_mse = mse[index_extinct]
    ax = sns.boxplot(data=[mse, extant_mse, extinct_mse])
    ax.set_xticklabels(["All", "Extant clade", "Extinct clade"])
    plt.ylabel("log(rMSE)")
    plt.yscale("log")
    ax4.annotate("D", xy=(-0.1, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax5 = fig.add_subplot(325)
    plt.scatter(all_sp_d, mse, color="mediumpurple", alpha=0.5)
    plt.grid(True)
    plt.ylabel("rMSE")
    plt.xlabel("Average species duration")
    ax5.annotate("E", xy=(-0.1, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax6 = fig.add_subplot(326)
    plt.scatter(all_c_d, mse, color="mediumseagreen", alpha=0.5)
    plt.grid(True)
    plt.ylabel("rMSE")
    plt.xlabel("Clade duration")
    ax6.annotate("F", xy=(-0.1, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, name + "Figure_4.pdf")
        plot_fig4 = matplotlib.backends.backend_pdf.PdfPages(file_name)
        plot_fig4.savefig(fig)
        plot_fig4.close()
        print("Plot saved as:", file_name)


def S2(results,
       show=False,
       min_x=None,
       max_x=None,
       wd=''):
    all_p = []
    all_t = []
    all_br = []
    all_occs = []
    for i in range(len(results)):
        br = results[i]["tot_br_length"]
        occs = results[i]["n_occurrences"]
        p = results[i]["n_sampled_species"]
        t = results[i]["n_species"]
        all_br.append(br)
        all_occs.append(occs)
        all_p.append(p)
        all_t.append(t)
    all_br = np.array(all_br)
    all_occs = np.array(all_occs)
    all_p = np.array(all_p)
    all_t = np.array(all_t)

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(121)
    p = np.divide(all_occs, all_br)
    sns.kdeplot(p,  fill=True, color="cornflowerblue")  # clip=(min_x, max_x))
    plt.xlabel("Preservation rate")
    print("median preservation rate is ", np.median(p))
    ax1.annotate("A", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax2 = fig.add_subplot(122)
    comp = np.divide(all_p, all_t)
    sns.kdeplot(comp,  fill=True, color="mediumturquoise", clip=(min_x, max_x))
    plt.xlabel("Completeness")
    print("median completeness is ", np.median(comp))
    ax2.annotate("B", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, "supplement_2.pdf")
        plot_cp = matplotlib.backends.backend_pdf.PdfPages(file_name)
        plot_cp.savefig(fig)
        plot_cp.close()
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


def plot_example_trajectories(predicted=None,
                              mid_time_bins=np.arange(-100, 0),
                              upper=None,
                              lower=None,
                              true=None,
                              sampled=None,
                              sqs_6=None,
                              # range_through=Yltt_r,
                              # xlimit=(-80, 0),
                              index_1=None,
                              index_2=None,
                              index_3=None,
                              index_4=None,
                              show=False,
                              color="orange",
                              wd=None):
    def transf(x): return np.exp(x)

    fig = plt.figure(figsize=(12, 7))

    ax1 = fig.add_subplot(221)
    plt.plot(mid_time_bins, transf(true)[index_1][::-1], linestyle='-', label="true Y", color=color)
    # if lower is not None:
        # print(transf(lower)[index][::-1].reshape(predicted[index].shape[0]))
    #     plt.fill_between(x=mid_time_bins,
    #                      y1=transf(lower)[index][::-1].reshape(predicted[index].shape[0]),
    #                      y2=transf(upper)[index][::-1].reshape(predicted[index].shape[0]),
    #                      color="orange", alpha=0.5)
    #     print(transf(upper)[index][::-1].reshape(predicted[index].shape[0]))
    #
    # if predicted is not None:
    #     plt.plot(mid_time_bins, transf(predicted)[index][::-1].reshape(predicted[index].shape[0]), '-', label="Predicted")
    # if sampled is not None:
    #     plt.plot(mid_time_bins, transf(sampled)[index][::-1], '-', label="Sampled")
    #
    plt.xlim(-50, 0)
    # plt.plot(mid_time_bins, transf(range_through)[index][::-1], '-', label="range-through Y")

    # if sqs_6 is not None:
    #     sqs_6 = np.array(sqs_6)
    #     sqs_6 = sqs_6[index, :]
    #     plt.plot(mid_time_bins, transf(sqs_6)[::-1], '-', label="SQS=0.6")
    #
    plt.xlabel("Time (Ma)")
    plt.ylabel("Diversity")
    ax1.annotate("A", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax2 = fig.add_subplot(222)
    plt.plot(mid_time_bins, transf(true)[index_2][::-1], linestyle='-', label="true Y", color=color)
    plt.xlim(-50, 0)
    plt.xlabel("Time (Ma)")
    plt.ylabel("Diversity")
    ax2.annotate("B", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax3 = fig.add_subplot(223)
    plt.plot(mid_time_bins, transf(true)[index_3][::-1], linestyle='-', label="true Y", color=color)
    plt.xlim(-80, 0)
    plt.xlabel("Time (Ma)")
    plt.ylabel("Diversity")
    ax3.annotate("C", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax4 = fig.add_subplot(224)
    plt.plot(mid_time_bins, transf(true)[index_4][::-1], linestyle='-', label="true Y", color=color)
    plt.xlim(-100, 0)
    plt.xlabel("Time (Ma)")
    plt.ylabel("Diversity")
    ax4.annotate("D", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    if show:
        fig.show()
    else:
        file_name = os.path.join("simulation_eg_plots.pdf")
        plot_div = matplotlib.backends.backend_pdf.PdfPages(file_name)
        fig.subplots_adjust(top=0.92)
        plot_div.savefig(fig)
        plot_div.close()
        print("Plot saved as:", file_name)


# PLOT DEEPDIVE, BIAS TESTS AND SQS
def plot_comparison(true_dd,
                    predicted_dd,
                    predicted_sqs,
                    r2_dd,
                    r2_sqs,
                    r2_dd_area,
                    r2_sqs_area,
                    r2_dd_time,
                    r2_sqs_time,
                    r2_dd_taxa,
                    r2_sqs_taxa,
                    r2_dd_spike,
                    r2_sqs_spike,
                    r2_dd_ddme,
                    r2_sqs_ddme,
                    show=False,
                    wd='',
                    name=''):

    fig = plt.figure(figsize=(12, 7), layout="constrained")
    red_points = mpatches.Patch(color='C3', label='SQS')
    blue_points = mpatches.Patch(color='#1f77b4', label='DeepDive')
    fig.legend(handles=[blue_points, red_points], loc="outside lower center", frameon=False, ncol=2, fontsize=12)

    gs = GridSpec(2, 6, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0:3])
    Ytest_01 = np.einsum('sb,s->sb', true_dd, 1 / np.max(true_dd, 1))  # rescale values to range [0, 1]
    Ytest_01[Ytest_01 == 0] = np.nan  # remove zeros
    Ytest_pred_natural_scale_01 = np.einsum('sb,s->sb', predicted_dd,
                                             1 / np.max(predicted_dd, 1))
    plt.scatter(Ytest_01.flatten(), Ytest_pred_natural_scale_01.flatten(), alpha=0.01)  # non rescaled DD
    plt.plot(Ytest_01.flatten(), Ytest_01.flatten(), color="black")
    plt.xlim(left=0, right=1)
    plt.ylim(bottom=0, top=1)
    plt.xlabel("Simulated relative diversity", fontsize=12)
    plt.ylabel("DeepDive estimated\nrelative diversity", fontsize=12)
    ax1.annotate("A", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax2 = fig.add_subplot(gs[0, 3:6])
    sqs_6_01 = np.einsum('sb,s->sb', predicted_sqs, 1 / np.nanmax(predicted_sqs, 1))
    sqs_6_01[sqs_6_01 == 0] = np.nan
    plt.scatter(Ytest_01.flatten(), sqs_6_01.flatten(), color="C3", alpha=0.01)
    plt.plot(Ytest_01.flatten(), Ytest_01.flatten(), color="black")
    plt.xlim(left=0, right=1)
    plt.ylim(bottom=0, top=1)
    plt.xlabel("Simulated relative diversity", fontsize=12)
    plt.ylabel("SQS estimated\nrelative diversity", fontsize=12)
    ax2.annotate("B", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax3 = fig.add_subplot(gs[1, 0])
    my_pal = {"deepdive": "#1f77b4", "sqs": "C3"}
    dat = pd.DataFrame(np.array([r2_dd, r2_sqs]).T)
    dat.columns = ("deepdive", "sqs")
    ax3 = sns.violinplot(data=dat, palette=my_pal, fill=True, density_norm="width", cut=0)
    ax3.set_xticklabels([])
    ax3.tick_params(bottom=False)
    plt.ylabel("$R^2$", fontsize=12)
    plt.xlabel("Original settings", fontsize=12)
    ax3.annotate("C", xy=(-0.5, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax4 = fig.add_subplot(gs[1, 1])
    dat = pd.DataFrame(np.array([r2_dd_time, r2_sqs_time]).T)
    dat.columns = ("deepdive", "sqs")
    ax4 = sns.violinplot(data=dat, palette=my_pal, fill=True, density_norm="width", cut=0)
    ax4.set_xticklabels([])
    ax4.tick_params(bottom=False)
    plt.ylabel("$R^2$", fontsize=12)
    plt.xlabel("Temporal bias", fontsize=12)
    ax4.annotate("D", xy=(-0.5, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax5 = fig.add_subplot(gs[1, 2])
    dat = pd.DataFrame(np.array([r2_dd_taxa, r2_sqs_taxa]).T)
    dat.columns = ("deepdive", "sqs")
    ax5 = sns.violinplot(data=dat, palette=my_pal, fill=True, density_norm="width", cut=0)
    ax5.set_xticklabels([])
    ax5.tick_params(bottom=False)
    plt.ylabel("$R^2$", fontsize=12)
    plt.xlabel("Taxonomic bias", fontsize=12)
    ax5.annotate("E", xy=(-0.5, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax6 = fig.add_subplot(gs[1, 3])
    dat = pd.DataFrame(np.array([r2_dd_area, r2_sqs_area]).T)
    dat.columns = ("deepdive", "sqs")
    ax6 = sns.violinplot(data=dat, palette=my_pal, fill=True, density_norm="width", cut=0)
    ax6.set_xticklabels([])
    ax6.tick_params(bottom=False)
    plt.ylabel("$R^2$", fontsize=12)
    plt.xlabel("Spatial bias", fontsize=12)
    ax6.annotate("F", xy=(-0.5, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax7 = fig.add_subplot(gs[1, 4])
    dat = pd.DataFrame(np.array([r2_dd_spike, r2_sqs_spike]).T)
    dat.columns = ("deepdive", "sqs")
    ax7 = sns.violinplot(data=dat, palette=my_pal, fill=True, density_norm="width", cut=0)
    ax7.set_xticklabels([])
    ax7.tick_params(bottom=False)
    plt.ylabel("$R^2$", fontsize=12)
    plt.xlabel("MSME", fontsize=12)
    ax7.annotate("G", xy=(-0.5, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax8 = fig.add_subplot(gs[1, 5])
    dat = pd.DataFrame(np.array([r2_dd_ddme, r2_sqs_ddme]).T)
    dat.columns = ("deepdive", "sqs")
    ax8 = sns.violinplot(data=dat, palette=my_pal, fill=True, density_norm="width", cut=0)
    ax8.set_xticklabels([])
    ax8.tick_params(bottom=False)
    plt.ylabel("$R^2$", fontsize=12)
    plt.xlabel("DDME", fontsize=12)
    ax8.annotate("H", xy=(-0.5, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)


    if show:
        fig.show()
    else:
        file_name = "comparison_plot_new_model.png"
        plt.savefig(file_name, dpi=300)
        plt.close()
        print("Plot saved as:", file_name)


def plot_relative_mse(rmse_dd,
                      rmse_sqs,
                      rmse_dd_a,
                      rmse_sqs_a,
                      rmse_dd_time,
                      rmse_sqs_time,
                      rmse_dd_tax,
                      rmse_sqs_tax,
                      rmse_dd_spike,
                      rmse_sqs_spike,
                      rmse_dd_ddme,
                      rmse_sqs_ddme,
                      show=False,
                      wd='',
                      name=''):

    fig = plt.figure(figsize=(15, 7), layout="constrained")
    red_points = mpatches.Patch(color='C3', label='SQS')
    blue_points = mpatches.Patch(color='#1f77b4', label='DeepDive')
    fig.legend(handles=[blue_points, red_points], loc="outside lower center", frameon=False, ncol=2, fontsize=16)

    ax1 = fig.add_subplot(161)
    my_pal = {"deepdive": "#1f77b4", "sqs": "C3"}
    dat = pd.DataFrame(np.array([rmse_dd, rmse_sqs]).T)
    dat.columns = ("deepdive", "sqs")
    ax1 = sns.boxplot(data=dat, palette=my_pal)
    ax1.set_yscale("log")
    ax1.set_yticks([0.0001, 0.001, 0.01, 0.1, 1])
    ax1.set_xticklabels([])
    ax1.tick_params(bottom=False)
    plt.ylim(0.0001, 1)
    plt.ylabel("rMSE", fontsize=16)
    plt.xlabel("Original settings", fontsize=16)
    ax1.annotate("A", xy=(-0.3, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax2 = fig.add_subplot(162)
    dat = pd.DataFrame(np.array([rmse_dd_time, rmse_sqs_time]).T)
    dat.columns = ("deepdive", "sqs")
    ax2 = sns.boxplot(data=dat, palette=my_pal)
    ax2.set_yscale("log")
    ax2.set_yticks([0.0001, 0.001, 0.01, 0.1, 1])
    ax2.set_xticklabels([])
    ax2.tick_params(bottom=False)
    plt.ylim(0.0001, 1)
    plt.ylabel("rMSE", fontsize=16)
    plt.xlabel("Temporal bias", fontsize=16)
    ax2.annotate("B", xy=(-0.3, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax3 = fig.add_subplot(163)
    dat = pd.DataFrame(np.array([rmse_dd_tax, rmse_sqs_tax]).T)
    dat.columns = ("deepdive", "sqs")
    ax3 = sns.boxplot(data=dat, palette=my_pal)
    ax3.set_yscale("log")
    ax3.set_yticks([0.0001, 0.001, 0.01, 0.1, 1])
    ax3.set_xticklabels([])
    ax3.tick_params(bottom=False)
    plt.ylim(0.0001, 1)
    plt.ylabel("rMSE", fontsize=16)
    plt.xlabel("Taxonomic bias", fontsize=16)
    ax3.annotate("C", xy=(-0.3, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax4 = fig.add_subplot(164)
    dat = pd.DataFrame(np.array([rmse_dd_a, rmse_sqs_a]).T)
    dat.columns = ("deepdive", "sqs")
    ax4 = sns.boxplot(data=dat, palette=my_pal)
    ax4.set_yscale("log")
    ax4.set_yticks([0.0001, 0.001, 0.01, 0.1, 1])
    ax4.set_xticklabels([])
    ax4.tick_params(bottom=False)
    plt.ylim(0.0001, 1)
    plt.ylabel("rMSE", fontsize=16)
    plt.xlabel("Spatial bias", fontsize=16)
    ax4.annotate("D", xy=(-0.3, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax5 = fig.add_subplot(165)
    dat = pd.DataFrame(np.array([rmse_dd_spike, rmse_sqs_spike]).T)
    dat.columns = ("deepdive", "sqs")
    ax5 = sns.boxplot(data=dat, palette=my_pal)
    ax5.set_yscale("log")
    ax5.set_yticks([0.0001, 0.001, 0.01, 0.1, 1])
    ax5.tick_params(bottom=False)
    ax5.set_xticklabels([])
    plt.ylim(0.0001, 1)
    plt.ylabel("rMSE", fontsize=16)
    plt.xlabel("MSME", fontsize=16)
    ax5.annotate("E", xy=(-0.3, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax6 = fig.add_subplot(166)
    dat = pd.DataFrame(np.array([rmse_dd_ddme, rmse_sqs_ddme]).T)
    dat.columns = ("deepdive", "sqs")
    ax6 = sns.boxplot(data=dat, palette=my_pal)
    ax6.set_yscale("log")
    ax6.set_yticks([0.0001, 0.001, 0.01, 0.1, 1])
    ax6.set_xticklabels([])
    ax6.tick_params(bottom=False)
    plt.ylim(0.0001, 1)
    plt.ylabel("rMSE", fontsize=16)
    plt.xlabel("DDME", fontsize=16)
    ax6.annotate("F", xy=(-0.3, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    if show:
        fig.show()
    else:
        file_name = "relative_mse_tests.pdf"
        plt.savefig(file_name, dpi=300)
        plt.close()
        print("Plot saved as:", file_name)


# HISTOGRAMS
def plot_hists(results,
               f_singletons,
               singletons,
               n_occs_per_tax,
               n_taxa,
               n_occs,
               show=False,
               wd="",
               name=""):

    fig = plt.figure(figsize=(12, 7), layout="constrained")

    all_p = []
    all_occs = []
    for i in range(len(results)):
        p = results[i]["n_sampled_species"]
        occs = results[i]["n_occurrences"]
        all_occs.append(occs)
        all_p.append(p)
    all_occs = np.array(all_occs)
    all_p = np.array(all_p)
    occs_per_sp = all_occs/all_p


    ax1 = fig.add_subplot(221)
    plt.hist(all_occs, bins=30, alpha=0.5, log=True)
    plt.axvline(x=n_occs, color="black")
    plt.xlabel("Occurrences per data set")
    ax1.annotate("A", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax2 = fig.add_subplot(222)
    plt.hist(all_p, bins=30, alpha=0.5, log=True)
    plt.axvline(x=n_taxa, color="black")
    plt.xlabel("Taxa per data set")
    ax2.annotate("B", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax3 = fig.add_subplot(223)
    plt.hist(occs_per_sp, bins=30, alpha=0.5)
    plt.axvline(x=n_occs_per_tax, color="black")
    plt.xlabel("Occurrences per sampled species")
    ax3.annotate("C", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax4 = fig.add_subplot(224)
    plt.hist(f_singletons, bins=30, alpha=0.5)
    plt.axvline(x=singletons, color="black")
    plt.xlabel("Frequency of singletons")
    ax4.annotate("D", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, "hists.png")
        plt.savefig("hists.png", dpi=300)
        plt.close()
        print("Plot saved as:", file_name)


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
    plt.hist(test_features[:, :, 0].flatten(), bins=bins, alpha=0.5, density=True, log=True)
    plt.hist(empirical_features[:, 0], bins=bins, alpha=0.5, density=True, log=True)
    plt.xlabel("Taxa")
    plt.ylabel("Frequency")
    ax1.annotate("A", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax2 = fig.add_subplot(332)
    bins = get_bins(test_features, empirical_features, 1, n_bins)
    plt.hist(test_features[:, :, 1].flatten()+1, bins=bins, alpha=0.5, density=True, log=True)
    plt.hist(empirical_features[:, 1], bins=bins, alpha=0.5, density=True, log=True)
    if log_occurrences:
        plt.xscale("log")
    plt.xlabel("Occurrences")
    plt.ylabel("Frequency")
    ax2.annotate("B", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax3 = fig.add_subplot(333)
    bins = get_bins(test_features, empirical_features, 2, n_bins)
    plt.hist(test_features[:, :, 2].flatten(), bins=bins, alpha=0.5, density=True, log=True)
    plt.hist(empirical_features[:, 2], bins=bins, alpha=0.5, density=True, log=True)
    plt.xlabel("Singletons")
    plt.ylabel("Frequency")
    ax3.annotate("C", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax4 = fig.add_subplot(334)
    bins = get_bins(test_features, empirical_features, 3, n_bins)
    plt.hist(test_features[:, :, 3].flatten(), bins=bins, alpha=0.5, density=True, log=True)
    plt.hist(empirical_features[:, 3], bins=bins, alpha=0.5, density=True, log=True)
    plt.xlabel("Endemic taxa")
    plt.ylabel("Frequency")
    ax4.annotate("D", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax5 = fig.add_subplot(335)
    bins = get_bins(test_features, empirical_features, 5, n_bins)
    plt.hist(test_features[:, :, 5].flatten(), bins=bins, alpha=0.5, density=True, log=True)
    plt.hist(empirical_features[:, 5], bins=bins, alpha=0.5, density=True, log=True)
    plt.xlabel("Range-through diversity")
    plt.ylabel("Frequency")
    ax5.annotate("E", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    ax6 = fig.add_subplot(336)
    bins = get_bins(test_features, empirical_features,6, n_bins)
    plt.hist(test_features[:, :, 6].flatten(), bins=bins, alpha=0.5, density=True, log=True)
    plt.hist(empirical_features[:, 6], bins=bins, alpha=0.5, density=True, log=True)
    plt.xlabel("Sampled localities")
    plt.ylabel("Frequency")
    ax6.annotate("F", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

    if features_names is not None:
        ax7 = fig.add_subplot(337)
        indx = np.array([i for i in range(len(features_names)) if "n_species_area" in features_names[i]])
        bins = get_bins(test_features, empirical_features, indx, n_bins)
        plt.hist(test_features[:, :, indx].flatten(), bins=bins, alpha=0.5, density=True, log=True)
        plt.hist(empirical_features[:, indx].flatten(), bins=bins, alpha=0.5, density=True, log=True)
        plt.xlabel("Taxa per region")
        plt.ylabel("Frequency")
        ax7.annotate("G", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

        ax8 = fig.add_subplot(338)
        indx = np.array([i for i in range(len(features_names)) if "n_occs_area" in features_names[i]])
        bins = get_bins(test_features, empirical_features, indx, n_bins)
        plt.hist(test_features[:, :, indx].flatten()+1, bins=bins, alpha=0.5, density=True, log=True)
        plt.hist(empirical_features[:, indx].flatten(), bins=bins, alpha=0.5, density=True, log=True)
        if log_occurrences:
            plt.xscale("log")
        plt.xlabel("Occurrences per region")
        plt.ylabel("Frequency")
        ax8.annotate("H", xy=(-0.15, 1), xycoords="axes fraction", fontweight="bold", fontsize=16)

        ax9 = fig.add_subplot(339)
        indx = np.array([i for i in range(len(features_names)) if "n_locs_area" in features_names[i]])
        bins = get_bins(test_features, empirical_features, indx, n_bins)
        plt.hist(test_features[:, :, indx].flatten(), bins=bins, alpha=0.5, density=True, log=True)
        plt.hist(empirical_features[:, indx].flatten(), bins=bins, alpha=0.5, density=True, log=True)
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

def plot_error_through_time(Ytest_r,
                            mean_prediction,
                            abs_path,
                            nmean_prediction=None,
                            test_folder="test"):
    error = (mean_prediction-Ytest_r)**2  # square difference between mean_prediction and ytest
    pd.DataFrame(error).to_csv(os.path.join(abs_path + '/test_sets/' + test_folder + '/error.csv'), index=False)
    mean_error = np.mean(error, axis=0)

    fig = plt.figure(figsize=(12, 7), layout="constrained")
    plt.plot(mean_error)

    if nmean_prediction is not None:
        new_model_error = (nmean_prediction-Ytest_r)**2  # square difference between mean_prediction and ytest
        pd.DataFrame(new_model_error).to_csv(os.path.join(abs_path + '/test_sets/' + test_folder + '/new_model_error.csv'), index=False)
        new_model_mean_error = np.mean(new_model_error, axis=0)

        plt.plot(new_model_mean_error)

    plt.axvline(x=66, color='tab:gray', linestyle="dashed")
    plt.axvline(x=16, color='tab:gray', linestyle="dashed")
    plt.xlabel('Time (Ma)')
    plt.ylabel('MSE')
    plt.gca().invert_xaxis()
    blue_line = mpatches.Patch(color='#1f77b4', label='Original model')
    orange_line = mpatches.Patch(color='#ff7f0e', label='Model with more diversity dependence and mass extinctions')
    fig.legend(handles=[blue_line, orange_line], loc="outside lower center", frameon=False, ncol=2)
    file_name = os.path.join("error_tracking_plot.pdf")
    plot_errors = matplotlib.backends.backend_pdf.PdfPages(file_name)
    plot_errors.savefig(fig)
    plot_errors.close()
    print("Plot saved as:", file_name)


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
        feat_10 = np.percentile(sim_features[:, :, i], q=1, axis=0)
        feat_10 = np.insert(feat_10, -len(feat_10), values=feat_10[0])
        feat_90 = np.percentile(sim_features[:, :, i], q=99, axis=0)
        feat_90 = np.insert(feat_90, -len(feat_90), values=feat_90[0])

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
                         feat_10,
                         feat_90,
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