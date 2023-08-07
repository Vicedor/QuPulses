import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import util.pulse as p
from typing import List, Dict, Union, Optional, Tuple

plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 20})

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'


class SubPlotOptions:
    def __init__(self, xlabel: Optional[str], ylabel: Optional[str], legendloc: str = 'upper right',
                 xlim: Optional[Tuple[float, float]] = None, ylim: Optional[Tuple[float, float]] = None):
        self.xlabel: str = xlabel
        self.ylabel: str = ylabel
        self.legendloc: str = legendloc
        self.xlim: Optional[Tuple[int, int]] = xlim
        self.ylim: Optional[Tuple[int, int]] = ylim


class LineOptions:
    def __init__(self, linetype: str, linewidth: int, color: str, label: str):
        self.linetype: str = linetype
        self.linewidth: int = linewidth
        self.color: str = color
        self.label: str = label


def simple_plot(xs_list: List[np.ndarray], ys_list: List[np.ndarray], line_options: List[LineOptions],
                plot_options: SubPlotOptions, title: str):
    plt.figure()
    for i in range(len(xs_list)):
        plt.plot(xs_list[i], ys_list[i], line_options[i].linetype, linewidth=line_options[i].linewidth,
                 color=line_options[i].color, label=line_options[i].label)
    plt.xlabel(plot_options.xlabel)
    plt.ylabel(plot_options.ylabel)
    plt.title(title)
    plt.legend(loc=plot_options.legendloc, frameon=False)
    plt.grid()
    if plot_options.xlim is not None:
        plt.xlim(plot_options.xlim)
    if plot_options.ylim is not None:
        plt.ylim(plot_options.ylim)
    plt.show()


def simple_subplots(xs_lists: List[List[np.ndarray]], ys_lists: List[List[np.ndarray]],
                    line_options_lists: List[List[LineOptions]],
                    plot_options_list: List[SubPlotOptions], title: str):
    fig, ax4 = plt.subplots(figsize=(8, 8))#, dpi=300)

    n_plots = len(xs_lists)
    for i in range(n_plots):
        plt.subplot(n_plots, 1, i + 1)
        plot_subplot(xs_lists[i], ys_lists[i], line_options_lists[i], plot_options_list[i])
    plt.title = title
    plt.show()


def plot_subplot(xs: Union[np.ndarray, List[np.ndarray]], ys: Union[np.ndarray, List[np.ndarray]],
                 line_options: Union[LineOptions, List[LineOptions]], plot_options: SubPlotOptions):
    """
    Plots a subplot of a matplotlib subplot class. Plots multiple plots in the same subplot as specified by parameters
    :param xs: A list of x-values for each plot
    :param ys: A list of y-values for each plot
    :param line_options: An object of options for each line.
    :param plot_options: An object of options for each plot.
    """
    if not isinstance(xs, list):
        xs = [xs]
    if not isinstance(ys, list):
        ys = [ys]
    if not isinstance(line_options, list):
        line_options = [line_options]

    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        option = line_options[i]
        plt.plot(x, np.real(y), option.linetype, linewidth=option.linewidth, color=option.color,
                 label=option.label)
    plt.xlabel(plot_options.xlabel)
    plt.ylabel(plot_options.ylabel)
    plt.grid()
    if plot_options.xlim is not None:
        plt.xlim(plot_options.xlim)
    if plot_options.ylim is not None:
        plt.ylim(plot_options.ylim)
    plt.legend(loc=plot_options.legendloc, frameon=False)


def plot_system_contents(times: np.ndarray, pulses: Union[p.Pulse, List[p.Pulse]],
                         pulse_options: Union[Tuple[LineOptions, LineOptions],
                                              List[Tuple[LineOptions, LineOptions]]],
                         contents: Union[np.ndarray, List[np.ndarray]],
                         content_options: Union[LineOptions, List[LineOptions]]):
    """
    Plots the system contents, as in Kiilerich's short paper. First plot is the pulse modes, next is the g(t)
    coefficients. Last are the mode contents
    :param times: An array of the times, which the functions are evaluated at
    :param pulses: The pulses used in the simulation for the u(t) and g(t) functions
    :param pulse_options: The plotting options for the pulses. A tuple of dictionaries for each plot. Options are
                          "linetype", "linewidth", "color" and "label"
    :param contents: The contents of the pulse and system modes
    :param content_options: The plotting options for the pulse content. A dictionary of options for each plot. Options
                            are "linetype", "linewidth", "color" and "label"
    """
    if not isinstance(pulses, list):
        pulses = [pulses]
    if not isinstance(pulse_options, list):
        pulse_options = [pulse_options]
    if not isinstance(content_options, list):
        content_options = [content_options]
    if not isinstance(contents, list):
        contents = [contents]

    nT = times.size

    u_lists = []
    g_lists = []
    for pulse in pulses:
        # Set up u(t) and g(t)
        g_list = np.zeros(nT, dtype=np.complex_)
        u_list = np.zeros(nT)
        for k in range(0, nT):
            g_list[k] = pulse.g(times[k]) ** 2
            u_list[k] = np.real(pulse.u(times[k]))
        u_lists.append(u_list)
        g_lists.append(g_list)

    fig, ax4 = plt.subplots(figsize=(8, 8))#, dpi=300)

    plt.subplot(3, 1, 1)
    xs = [times for _ in range(len(u_lists))]
    options = [pulse_option[0] for pulse_option in pulse_options]
    plot_subplot(xs, u_lists, options, SubPlotOptions(xlabel=None, ylabel='$\mathrm{Modes}$', legendloc='upper right'))
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='on',  # ticks along the bottom edge are off
        top='on',  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    plt.subplot(3, 1, 2)
    xs = [times for i in range(len(g_lists))]
    options = [pulse_option[1] for pulse_option in pulse_options]
    plot_subplot(xs, g_lists, options,  SubPlotOptions(xlabel=None, ylabel='$\mathrm{Rates}\, (\gamma)$',
                 legendloc='center right'))
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='on',  # ticks along the bottom edge are off
        top='on',  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.yticks((0, 4, 8), ('$0.0$', '$4.0$', '$8.0$'))
    plt.ylim((-8, 8))

    plt.subplot(3, 1, 3)
    xs = [times for i in range(len(contents))]
    plot_subplot(xs, contents, content_options,
                 SubPlotOptions(xlabel='Time (units of $\gamma^{-1}$)', ylabel='$\mathrm{Exctiations}$', legendloc='center right'))

    #plt.savefig('test.png', bbox_inches='tight')
    plt.show()


def plot_autocorrelation(autocorr_mat, vs, eigs, times):
    fig, ax = plt.subplots(figsize=(8, 16))

    plt.subplot(2, 1, 1)
    real_autocorr_mat1 = np.real(autocorr_mat)
    im = plt.imshow(real_autocorr_mat1)
    plt.colorbar(im)

    plt.subplot(2, 1, 2)
    for i, v in enumerate(vs):
        plt.plot(times, v, "-", linewidth=4, label=f"$v_{i} (n_{i}={np.real(eigs[i]):.2f})$")
    plt.legend(loc='center right', frameon=False)

    plt.show()


def plot_arm_populations(taus, arm0_populations, arm1_populations):
    plt.figure()
    plt.plot(taus, arm0_populations, label="arm0")
    plt.plot(taus, arm1_populations, label="arm1")
    plt.xlabel("$\\tau$")
    plt.ylabel("Photon content")
    plt.legend()
    plt.show()


def plot_fidelities(xs, fidelity_aa, fidelity_ab, fidelity_ba, xlabel="", title=""):
    plt.figure()
    plt.plot(xs, fidelity_aa, "-", label="aa")
    plt.plot(xs, fidelity_ab, "--", label="ab")
    plt.plot(xs, fidelity_ba, ":", label="ba")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Fidelity")
    plt.legend()
    plt.show()
