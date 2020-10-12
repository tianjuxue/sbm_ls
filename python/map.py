import numpy as np
import matplotlib.pyplot as plt


def vis_step(cycle):
    raw_points = np.loadtxt('data/text/raw_points{:03}.txt'.format(cycle))
    map_points = np.loadtxt('data/text/map_points{:03}.txt'.format(cycle))

    fig = plt.figure(cycle)
    for i in range(len(raw_points)):
        # plt.plot([raw_points[i][0], map_points[i][0]], [raw_points[i][1], map_points[i][1]], color='blue')
        # plt.arrow(raw_points[i][0], raw_points[i][1], map_points[i][0] - raw_points[i][0], map_points[i][1] - raw_points[i][1], width=0.0005, head_width=0.0010)
        plt.arrow(raw_points[i][0], raw_points[i][1], map_points[i][0] - raw_points[i][0], map_points[i][1] - raw_points[i][1])
        # plt.scatter(map_points[i][0], map_points[i][1], color='blue')
    
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal', adjustable='box')


if __name__ == '__main__':
    for i in range(0, 1):
        vis_step(i)
    plt.show() 