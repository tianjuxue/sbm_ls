import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection


def build_base_pore(coords_fn, n_points, c1, c2):
    thetas = [float(i) * 2 * math.pi / n_points for i in range(n_points)]
    radii = [coords_fn(float(i) * 2 * math.pi / n_points, c1, c2) for i in range(n_points)]
    points = [(rtheta * np.cos(theta), rtheta * np.sin(theta))
              for rtheta, theta in zip(radii, thetas)]
    return np.array(points), np.array(radii), np.array(thetas)

def coords_fn(theta, c1, c2):
    return np.sqrt((1 + c1 * np.cos(4 * theta) + c2 * np.cos(8 * theta)))

def build_pore_polygon(base_pore_points, offset):
    points = [[p[0] + offset[0], p[1] + offset[1]] for p in base_pore_points]
    points = affine_group(points)
    points = np.asarray(points)
    pore = Polygon(points)
    return pore

def affine_transformation(point):
    point[0], point[1] = point[0] + 0.*point[1], point[1] - 0.*point[1]
    return point

def affine_group(points):
    points = [affine_transformation(point) for point in points]
    return points

def plot_grid(c1_list, c2_list, save=False, pore_number='', group_plot=False):
    L0 = 4
    pore_radial_resolution = 120
    n_cells = 3 if group_plot else 1
    patches = []
    colors = []

    points = [[0, 0], [n_cells*L0, 0], [n_cells*L0, n_cells*L0], [0, n_cells*L0]]
    points = affine_group(points)

    frame = Polygon(np.asarray(points))
    patches.append(frame)

    colors.append((0., 0., 0.))
    # colors.append((254./255.,127./255.,156./255.))

    for i in range(n_cells):
        for j in range(n_cells):
            c1 = c1_list[i]
            c2 = c2_list[j]
            base_pore_points, radii, thetas = build_base_pore(
                coords_fn, pore_radial_resolution, c1, c2)            
            pore = build_pore_polygon(
                base_pore_points, offset=(L0 * (i + 0.5), L0 * (j + 0.5)))

            patches.append(pore)
            colors.append((1,1,1))

    fig, ax = plt.subplots()
    p = PatchCollection(patches, alpha=1, edgecolor=None, facecolor=colors)
    ax.add_collection(p)
    plt.axis('equal')
    plt.axis('off')
    if save:
        fig.savefig("data/pdf/pores/pore_{}.pdf".format(pore_number), bbox_inches='tight')
        fig.savefig("data/png/pores/pore_{}.png".format(pore_number), bbox_inches='tight')


def plot_single(pore_number):
    c1 = ((pore_number // 3) - 1) * 0.2
    c2 = ((pore_number % 3) - 1) * 0.2
    plot_grid([c1], [c2], save=True, pore_number=pore_number)


if __name__ == '__main__':
    c1_list = [-0.2, 0., 0.2]
    c2_list = [-0.2, 0., 0.2]
    plot_grid(c1_list, c2_list, group_plot=True)

    total_pore_number = 9
    for i in range(total_pore_number):
        plot_single(i)

    plt.show()