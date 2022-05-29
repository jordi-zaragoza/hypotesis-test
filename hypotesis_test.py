import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import scipy.stats as st
from ipywidgets import interact

plt.style.use('seaborn')


def get_statistic_params(function, t_statistic, *degrees_of_freedom):
    x = np.linspace(-5, 5, 1000)
    y_pdf = function.pdf(x, *degrees_of_freedom)
    idx_statistic = (np.abs(x - t_statistic)).argmin()
    y_t_statistic_pdf = function.pdf(t_statistic, *degrees_of_freedom)
    text2 = "(" + str(round(function.cdf(t_statistic, *degrees_of_freedom), 3)) + ", " + str(
        round(t_statistic, 2)) + ")"
    return x, y_pdf, idx_statistic, y_t_statistic_pdf, text2


def draw_statistic(function, axes, t_statistic, color, *degrees_of_freedom):

    x, y_pdf, idx_statistic, y_t_statistic_pdf, text2 = get_statistic_params(function, t_statistic, *degrees_of_freedom)

    axes.plot(x, y_pdf, label='pdf')
    axes.set_title("Distribution")
    axes.set_xlabel("x")
    axes.set_xlim(-6, 6)
    axes.fill_between(x[:idx_statistic], y_pdf[:idx_statistic], color, alpha=0.25)
    axes.vlines(t_statistic, ymin=0, ymax=y_t_statistic_pdf, color=color)
    axes.text(2.5, 0.3, text2, style='italic', color="black", fontsize=15)


def get_critical_params(function, alpha, *degrees_of_freedom):
    x = np.linspace(-5, 5, 1000)
    y_pdf = function.pdf(x, *degrees_of_freedom)
    idx_critical = (np.abs(x - function.ppf(alpha, *degrees_of_freedom))).argmin()
    t_critical = function.ppf(alpha, *degrees_of_freedom)
    y_t_critical_pdf = function.pdf(t_critical, *degrees_of_freedom)
    text1 = "(" + str(round(alpha, 3)) + ", " + str(round(t_critical, 2)) + ")"
    return x, y_pdf, idx_critical, t_critical, y_t_critical_pdf, text1


def draw_critical(side, function, axes, alpha, *degrees_of_freedom):

    x, y_pdf, idx_critical, t_critical, y_t_critical_pdf, text1 = get_critical_params(function, alpha,
                                                                                      *degrees_of_freedom)

    axes.vlines(t_critical, ymin=0, ymax=y_t_critical_pdf, color="red")
    axes.text(3.5 * t_critical / abs(t_critical) - 1.5, 0.3 - abs(t_critical / 10), text1, style='italic', color="red",
              fontsize=15)
    {
        'left': lambda: axes.fill_between(x[:idx_critical], y_pdf[:idx_critical], color="red", alpha=0.15),
        'right': lambda: axes.fill_between(x[idx_critical:], y_pdf[idx_critical:], color="red", alpha=0.15)
    }.get(side, lambda: print('Error, select either left or right side'))()


def draw_critical_one_side(side, function, axes, alpha, *degrees_of_freedom):
    draw_critical(side, function, axes, alpha, *degrees_of_freedom)


def draw_critical_two_sides(function, axes, alpha, *degrees_of_freedom):
    draw_critical_one_side('left', function, axes, alpha / 2, *degrees_of_freedom)
    draw_critical_one_side('right', function, axes, 1 - alpha / 2, *degrees_of_freedom)


def plot_critical_and_statistic(function, t_statistic, side, alpha=0.05, *degrees_of_freedom):

    fig, axes = plt.subplots(1, 1, figsize=(6, 6))

    draw_statistic(function, axes, t_statistic, "black", *degrees_of_freedom)
    {
        'two': lambda: draw_critical_two_sides(function, axes, alpha, *degrees_of_freedom),
        'right': lambda: draw_critical_one_side('right', function, axes, 1 - alpha, *degrees_of_freedom),
        'left': lambda: draw_critical_one_side('left', function, axes, alpha, *degrees_of_freedom)
    }.get(side, lambda: print('Incorrect side selected, please select right, left or two sides'))()

    plt.show()


def use_params(function, t_statistic, side, alpha, degrees_of_freedom):
    args = (function, t_statistic, side, alpha)

    if degrees_of_freedom:
        args = args + (degrees_of_freedom,)

    plot_critical_and_statistic(*args)


def get_dof_widget(with_dof=False):
    if with_dof:
        return widgets.IntSlider(
            value=10,
            min=2,
            max=30,
            step=1
        )
    else:
        return widgets.fixed(None)


def run_widgets(function, with_dof=False):
    interact(use_params,
             function=widgets.fixed(function),
             t_statistic=widgets.BoundedFloatText(
                 min=-100,
                 max=100,
                 value=-1.8,
                 description='T-statistic:',
                 disabled=False,
                 step=0.05
             ),
             side=widgets.Dropdown(
                 options=['right', 'left', 'two'],
                 value='two',
                 description='Type:'
             ),
             alpha=widgets.BoundedFloatText(
                 value=0.05,
                 description='Alpha:',
                 disabled=False,
                 step=0.005
             ),
             degrees_of_freedom=get_dof_widget(with_dof))


def test(function_name):
    """
    This function creates an interactive plot for the hypothesis test
    Input (str) functions supported:
        'norm' -> for a normal distribution
        't' -> for a t-student distribution
    """
    {
        't': lambda: run_widgets(function=st.t, with_dof=True),
        'norm': lambda: run_widgets(function=st.norm)
    }.get(function_name, lambda: 'Function {} not supported'.format(function_name))()
