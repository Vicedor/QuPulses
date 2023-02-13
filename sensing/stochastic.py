import qutip as qt
import SLH.network as nw
import util.plots as plots
from util.quantumsystem import QuantumSystem
from util.plots import SubPlotOptions, LineOptions


def photon_counting(system: QuantumSystem, plot=False):
    """
    Performs photon counting experiment as in Khanahmadi 2022. The function uses the qutip photocurrent method
    :param system: The quantum system to perform the photon-counting experiment on
    :param plot: Boolean of whether to plot or not. Note: very specific expectation observables are needed for the plot
                 to make sense.
    :return: Tuple of the result object from qutip photocurrent method, and an array of measurements at each time step
    """
    times = system.times
    T = times[-1]
    nT = len(times)

    total_system: nw.Component = system.create_component()
    result: qt.solver.Result = qt.photocurrent_mesolve(total_system.H, system.psi0, times, [], total_system.get_Ls(),
                                                       system.get_expectation_observables(), store_measurement=True,
                                                       options=qt.Options(store_states=True), normalize=False)
    measurements = result.measurement[0] * T / nT

    if plot:
        pulse_options, content_options = system.get_plotting_options()
        plot_options = [SubPlotOptions("Time", "Content"),
                        SubPlotOptions("Time", "Content"),
                        SubPlotOptions("Time", "Jump")]
        plots.simple_subplots([[times], [times], [times]], [[result.expect[0]], [result.expect[1]], [measurements]],
                              [[content_options[0]], [content_options[1]],
                               [LineOptions(linetype='-', linewidth=4, color='b', label='signal')]],
                              plot_options, "Photon counting")

    return result, measurements


def homodyne_detection(system: QuantumSystem, plot=False):
    """
    Performs homodyne detection experiment as in Khanahmadi 2022. The function uses the qutip smesolve method
    :param system: The quantum system to perform the homodyne detection experiment on
    :param plot: Boolean of whether to plot or not. Note: very specific expectation observables are needed for the plot
                 to make sense.
    :return: Tuple of the result object from qutip smesolve method, and an array of measurements at each time step
    """
    times = system.times
    T = times[-1]
    nT = len(times)

    total_system: nw.Component = system.create_component()
    result: qt.solver.Result = qt.smesolve(total_system.H, system.psi0, times, [], total_system.get_Ls(),
                                           system.get_expectation_observables(), store_measurement=True,
                                           method="homodyne")
    measurements = result.measurement[0] * T / nT

    if plot:
        pulse_options, content_options = system.get_plotting_options()
        plot_options = [SubPlotOptions("Time", "Content"),
                        SubPlotOptions("Time", "Content"),
                        SubPlotOptions("Time", "Signal")]
        plots.simple_subplots([[times], [times], [times]],
                              [[result.expect[0]], [result.expect[1]], [measurements]],
                              [[content_options[0]], [content_options[1]],
                               [LineOptions(linetype='-', linewidth=4, color='b', label='signal')]],
                              plot_options, "Homodyne detection")
    return result, measurements


def hypothesis_testing(system: QuantumSystem, plot=False):
    """
    Performs hypothesis testing experiment as in Khanahmadi 2022. The function uses the qutip mcsolve method
    :param system: The quantum system to perform the hypothesis testing experiment on. The system should have an
                   ancillary degree of freedom, which determines which hypothesis to evolve according to. The initial
                   state of this ancillary degree of freedom is the initial probabilities for the hypotheses.
    :param plot: Boolean of whether to plot or not. Note: very specific expectation observables are needed for the plot
                 to make sense.
    :return: Tuple of the result object from qutip mcsolve method, and an array of measurements at each time step
    """
    times = system.times
    total_system: nw.Component = system.create_component()
    result: qt.solver.Result = qt.mcsolve(total_system.H, system.psi0, times, total_system.get_Ls(),
                                          system.get_expectation_observables(), ntraj=1,
                                          options=qt.Options(store_states=True))
    measurement_times = result.col_times[0]
    measurements = [0 for _ in times]
    for measurement_time in measurement_times:
        for i, t in enumerate(times):
            if t > measurement_time:
                measurements[i] = 1
                break

    if plot:
        pulse_options, content_options = system.get_plotting_options()
        plot_options = [SubPlotOptions("Time", "Content"),
                        SubPlotOptions("Time", "Prob. for hypothesis", ylim=(-0.01, 1.01)),
                        SubPlotOptions("Time", "Jump")]
        plots.simple_subplots([[times, times], [times, times], [times]],
                              [[result.expect[0], result.expect[1]],
                               [result.expect[2], result.expect[3]],
                               [measurements]],
                              [[content_options[0], content_options[1]], [content_options[2], content_options[3]],
                               [LineOptions(linetype='-', linewidth=4, color='b', label='signal')]],
                              plot_options, "Photon counting")
    return result, measurements
