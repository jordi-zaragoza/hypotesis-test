import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import scipy.stats as st
from ipywidgets import interact

plt.style.use('seaborn')


def draw_statistic(function, axes, t_statistic, color, *degrees_of_freedom):
    x = np.linspace(-5, 5, 1000)
    y_pdf = function.pdf(x, *degrees_of_freedom)
    idx_statistic = (np.abs(x - t_statistic)).argmin()
    y_t_statistic_ppf = function.pdf(t_statistic, *degrees_of_freedom)
    text2 = "(" + str(round(function.cdf(t_statistic, *degrees_of_freedom), 3)) + ", " + str(
        round(t_statistic, 2)) + ")"

    axes.plot(x, y_pdf, label='pdf')
    axes.set_title("Distribution")
    axes.set_xlabel("x")
    axes.set_xlim(-6, 6)
    axes.fill_between(x[:idx_statistic], y_pdf[:idx_statistic], color, alpha=0.25)
    axes.vlines(t_statistic, ymin=0, ymax=y_t_statistic_ppf, color=color)
    axes.text(2.5, 0.3, text2, style='italic', color="black", fontsize=15)


def draw_critical(function, axes, alpha, color, *degrees_of_freedom):
    t_critical = function.ppf(alpha, *degrees_of_freedom)
    y_t_critical_pdf = function.pdf(t_critical, *degrees_of_freedom)
    text1 = "(" + str(round(alpha, 3)) + ", " + str(round(t_critical, 2)) + ")"

    axes.vlines(t_critical, ymin=0, ymax=y_t_critical_pdf, color=color)
    axes.text(3.5 * t_critical / abs(t_critical) - 1.5, 0.3 - abs(t_critical / 10), text1, style='italic', color="red",
              fontsize=15)


def fill_critical(side, function, axes, alpha, color, *degrees_of_freedom):
    x = np.linspace(-5, 5, 1000)
    y_pdf = function.pdf(x, *degrees_of_freedom)
    idx = (np.abs(x - function.ppf(alpha, *degrees_of_freedom))).argmin()

    if side == 'left':
        axes.fill_between(x[:idx], y_pdf[:idx], color=color, alpha=0.15)
    else:
        axes.fill_between(x[idx:], y_pdf[idx:], color=color, alpha=0.15)


def plot_critical_and_statistic(function, t_statistic, side="two", alpha=0.05, *degrees_of_freedom):

    fig, axes = plt.subplots(1, 1, figsize=(6, 6))

    draw_statistic(function, axes, t_statistic, "black", *degrees_of_freedom)

    if side == "two":
        draw_critical(function, axes, alpha / 2, "red", *degrees_of_freedom)
        fill_critical('left', function, axes, alpha / 2, "red", *degrees_of_freedom)

        draw_critical(function, axes, 1 - (alpha / 2), "red", *degrees_of_freedom)
        fill_critical('right', function, axes, 1 - alpha / 2, "red", *degrees_of_freedom)

    elif side == "right":
        draw_critical(function, axes, 1 - alpha, "red", *degrees_of_freedom)
        fill_critical('right', function, axes, 1 - alpha, "red", *degrees_of_freedom)

    else:
        draw_critical(function, axes, alpha, "red", *degrees_of_freedom)
        fill_critical('left', function, axes, alpha, "red", *degrees_of_freedom)

    plt.show()


def use_params(function, t_statistic, side, alpha, degrees_of_freedom):
    if degrees_of_freedom is None:
        args = (function, t_statistic, side, alpha)
    else:
        args = (function, t_statistic, side, alpha, degrees_of_freedom)

    plot_critical_and_statistic(*args)


def run_widgets(function, dof_widget):
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
             degrees_of_freedom=dof_widget)


def test(function_name):
    """
    This function creates an interactive plot for the hypothesis test
    Input (str) functions supported:
        'norm' -> for a normal distribution
        't' -> for a t-student distribution
    """
    if function_name == 't':
        function = st.t
        dof_widget = widgets.IntSlider(
            value=10,
            min=2,
            max=30,
            step=1
        )
        run_widgets(function, dof_widget)

    elif function_name == 'norm':
        function = st.norm
        dof_widget = widgets.fixed(None)
        run_widgets(function, dof_widget)

    else:
        print('Function {} not supported'.format(function_name))
