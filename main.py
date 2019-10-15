import math

import matplotlib.pyplot as plt
import numpy as np
import progressbar
from numba import jit


@jit(nopython=True)
def rand_num(min_value, max_value):
    return np.random.uniform(min_value, max_value)


@jit(nopython=True)
def f(x):
    # return x
    # return math.sin(x)
    return pow(x, 3)


def graph(max_iterations, lim_min_x, lim_max_x, lim_min_y, lim_max_y, function):
    red_color = "#ff0000"
    green_color = "#018706"

    for _ in progressbar.progressbar(range(max_iterations)):
        x = rand_num(lim_min_x, lim_max_x)
        y = rand_num(lim_min_y, lim_max_y)

        color = green_color if y < function(x) else red_color

        plt.scatter(x, y, color=color, s=2)

    plt.show()


def show_summary(setup):
    print("### A simple Monte Carlo simulation ###\n")
    print("Number of iterations:\t\t", setup["max_iterations"])
    print("Max value of x axis:\t\t", setup["lim_min_x"])
    print("Min value of x axis:\t\t", setup["lim_max_x"])
    print("Max value of y axis:\t\t", setup["lim_min_y"])
    print("Min value of y axis:\t\t", setup["lim_max_y"])
    print()


def setup_sin():
    def sin_function(x):
        return math.sin(x)

    return {
        "max_iterations": 5000,  # int(input("Inform the max number if iterations: "))
        "lim_min_x": -10,
        "lim_max_x": 10,
        "lim_min_y": -1.5,
        "lim_max_y": 1.5,
        "function": sin_function
    }


def setup_y_equal_x():
    def sin_function(x):
        return math.pow(x, 2) / math.log10(x)
        # return x

    return {
        "max_iterations": 10000,  # int(input("Inform the max number if iterations: "))
        "lim_min_x": 0,
        "lim_max_x": 3,
        "lim_min_y": -50,
        "lim_max_y": 50,
        "function": sin_function
    }


if __name__ == '__main__':
    setup = setup_y_equal_x()

    show_summary(setup)

    print("Initializing simulation")
    graph(
        max_iterations=setup["max_iterations"],
        lim_min_x=setup["lim_min_x"],
        lim_max_x=setup["lim_max_x"],
        lim_min_y=setup["lim_min_y"],
        lim_max_y=setup["lim_max_y"],
        function=setup["function"]
    )
    print("Done.")
