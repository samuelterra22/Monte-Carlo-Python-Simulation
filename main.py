import math

import matplotlib.pyplot as plt
import numpy as np
import progressbar
from numba import jit


@jit(nopython=True)
def rand_num(min_value, max_value):
    return np.random.uniform(min_value, max_value)


def graph(max_iterations, lim_min_x, lim_max_x, lim_min_y, lim_max_y, function, function_name):
    red_color = "#ff0000"
    green_color = "#018706"

    for _ in progressbar.progressbar(range(max_iterations)):
        x = rand_num(lim_min_x, lim_max_x)
        y = rand_num(lim_min_y, lim_max_y)

        color = green_color if y < function(x) else red_color

        plt.scatter(x, y, color=color, s=2)

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Function " + function_name + " with " + str(max_iterations) + " points")
    plt.show()


def show_summary(setup_data):
    print("### A simple Monte Carlo simulation ###\n")
    print("Number of iterations:\t\t", setup_data["max_iterations"])
    print("Max value of x axis:\t\t", setup_data["lim_min_x"])
    print("Min value of x axis:\t\t", setup_data["lim_max_x"])
    print("Max value of y axis:\t\t", setup_data["lim_min_y"])
    print("Min value of y axis:\t\t", setup_data["lim_max_y"])
    print()


def setup_fx3():
    def f_x3_function(x):
        return pow(x, 3)

    return {
        "max_iterations": 5000,
        "lim_min_x": -10,
        "lim_max_x": 10,
        "lim_min_y": -10,
        "lim_max_y": 10,
        "function": f_x3_function,
        "function_name": "x^3",
    }


def setup_sin():
    def sin_function(x):
        return math.sin(x)

    return {
        "max_iterations": 5000,
        "lim_min_x": -10,
        "lim_max_x": 10,
        "lim_min_y": -1.5,
        "lim_max_y": 1.5,
        "function": sin_function,
        "function_name": "sin(x)",
    }


def setup_y_equal_x():
    def y_equal_x_function(x):
        return x

    return {
        "max_iterations": 10000,
        "lim_min_x": 0,
        "lim_max_x": 3,
        "lim_min_y": -50,
        "lim_max_y": 50,
        "function": y_equal_x_function,
        "function_name": "x = y",
    }


if __name__ == '__main__':
    setup = setup_fx3()

    show_summary(setup)

    print("Initializing simulation")
    graph(
        max_iterations=setup["max_iterations"],
        lim_min_x=setup["lim_min_x"],
        lim_max_x=setup["lim_max_x"],
        lim_min_y=setup["lim_min_y"],
        lim_max_y=setup["lim_max_y"],
        function=setup["function"],
        function_name=setup["function_name"]
    )
    print("Done.")
