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
    print("")
    print("Summary:")
    print("\tNumber of iterations:\t", setup_data["max_iterations"])
    print("\tMax value of x axis:\t", setup_data["lim_min_x"])
    print("\tMin value of x axis:\t", setup_data["lim_max_x"])
    print("\tMax value of y axis:\t", setup_data["lim_min_y"])
    print("\tMin value of y axis:\t", setup_data["lim_max_y"])
    print("\tFunction:\t\t", setup_data["function_name"])
    print()


def setup_y_equal_x():
    def y_equal_x_function(x):
        return x

    return {
        "max_iterations": 10000,
        "lim_min_x": -50,
        "lim_max_x": 50,
        "lim_min_y": -50,
        "lim_max_y": 50,
        "function": y_equal_x_function,
        "function_name": "f(x) = x",
    }


def setup_fx2():
    def f_x2_function(x):
        return math.pow(x, 2)

    return {
        "max_iterations": 10000,
        "lim_min_x": -15,
        "lim_max_x": 15,
        "lim_min_y": 0,
        "lim_max_y": 15,
        "function": f_x2_function,
        "function_name": "f(x) = x^2",
    }


def setup_fx3():
    def f_x3_function(x):
        return math.pow(x, 3)

    return {
        "max_iterations": 10000,
        "lim_min_x": -10,
        "lim_max_x": 10,
        "lim_min_y": -10,
        "lim_max_y": 10,
        "function": f_x3_function,
        "function_name": "f(x) = x^3",
    }


def setup_sin():
    def sin_function(x):
        return math.sin(x)

    return {
        "max_iterations": 10000,
        "lim_min_x": -10,
        "lim_max_x": 10,
        "lim_min_y": -1.5,
        "lim_max_y": 1.5,
        "function": sin_function,
        "function_name": "f(x) = sin(x)",
    }


def setup_fx6():
    def f_x6_function(x):
        return math.pow(x, 6)

    return {
        "max_iterations": 10000,
        "lim_min_x": -5,
        "lim_max_x": 5,
        "lim_min_y": -1,
        "lim_max_y": 15,
        "function": f_x6_function,
        "function_name": "f(x) = x^6",
    }


def setup_x2logx():
    def f_x2logx_function(x):
        return math.pow(x, 2) / math.log10(x)

    return {
        "max_iterations": 10000,
        "lim_min_x": 0,
        "lim_max_x": 2.5,
        "lim_min_y": -70,
        "lim_max_y": 70,
        "function": f_x2logx_function,
        "function_name": "f(x) = x^2 / log(x)",
    }


def setup_tgx():
    def f_tgx_function(x):
        return math.tan(x)

    return {
        "max_iterations": 10000,
        "lim_min_x": -1.5,
        "lim_max_x": 1.5,
        "lim_min_y": -10,
        "lim_max_y": 10,
        "function": f_tgx_function,
        "function_name": "f(x) = tg(x)",
    }


def setup_log_x():
    def f_log_x_function(x):
        return math.tan(x)

    return {
        "max_iterations": 10000,
        "lim_min_x": 0,
        "lim_max_x": 1.5,
        "lim_min_y": -1,
        "lim_max_y": 40,
        "function": f_log_x_function,
        "function_name": "f(x) = log(x)",
    }


if __name__ == '__main__':
    print("### A simple Monte Carlo simulation ###\n")
    print("[1] f(x) = x")
    print("[2] f(x) = x^2")
    print("[3] f(x) = x^3")
    print("[4] f(x) = x^6")
    print("[5] f(x) = sin(x)")
    print("[6] f(x) = x^2 / log(x)")
    print("[7] f(x) = f(x) = tg(x)")
    print("[8] f(x) = f(x) = log(x)")
    option = int(input("Choose the function: "))

    setup = {}

    if option == 1:
        setup = setup_y_equal_x()
    elif option == 2:
        setup = setup_fx2()
    elif option == 3:
        setup = setup_fx3()
    elif option == 4:
        setup = setup_fx6()
    elif option == 5:
        setup = setup_sin()
    elif option == 6:
        setup = setup_x2logx()
    elif option == 7:
        setup = setup_tgx()
    elif option == 8:
        setup = setup_log_x()

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
