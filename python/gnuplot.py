import PyGnuplot as gp
import numpy as np


def convergence_plot_multiplot(error_flag):
    # gp.c('set format y "%.1f"')
    gp.c('set multiplot layout 3, 3  title "Multiplot layout" font ",12" scale 1, 1')
    gp.c('set title "convergence tests" font ",10"')
    gp.c('set xlabel "h" font ",10"')
    gp.c('set ylabel "error" font ",10"')
    gp.c('set xtics font "Helvetica, 10"')
    gp.c('set ytics font "Helvetica, 10"')
    gp.c('set style data linespoints')
    gp.c('set key top left')
    gp.c('set key box')
    gp.c('set logscale')
    # gp.c('set autoscale')

    total_pore_number = 9
    for i in range(total_pore_number):
        newton_file_name = '"data/dat/pore_' + str(i) + '_newton.dat"'
        bs_file_name = '"data/dat/pore_' + str(i) + '_bs.dat"'
        gp.c('plot ' + newton_file_name + ' u 1:' + str(error_flag) + ' title "newton" lc "blue" pt 5 ps 1 lt 1 lw 1 , '
             + bs_file_name + ' u 1:' + str(error_flag) + ' title "bs" lc "red" pt 7 ps 1 lt 1 lw 1')
        # point_type point_size line_width
    gp.c('unset multiplot')


def convergence_plot_single(pore_number, error_flag):
    if error_flag == L2_flag:
        label_name = 'L2 error'
    elif error_flag == SD_flag:
        label_name = 'SD error'
    elif error_flag == interface_flag:
        label_name = 'Interface error'
    else:
        assert 0, "Wrong error_flag!"

    gp.c('set terminal qt ' + str(pore_number))
    # gp.c('set title "Convergence tests" font ",14"')
    # gp.c('set xlabel "Mesh size h" font ",12"')
    # gp.c('set ylabel "' + label_name + '" font ",12"')
    gp.c('set xtics font ",12"')
    gp.c('set ytics font ",12"')
    gp.c('set style data linespoints')
    gp.c('set key top left font ",12"')
    gp.c('set logscale')
    gp.c('set format y "10^{%T}"')
    # gp.c('set xrange [0.01:0.2]')
    # gp.c('set yrange [1e-8:1e-2]')
    gp.c('set autoscale')
    gp.c('set size square')
    # gp.c('set lmargin 0')
    # gp.c('set rmargin 0')
    # gp.c('set bmargin 0')
    # gp.c('set tmargin 0')

    newton_file_name = 'data/dat/pore_{}_newton.dat'.format(pore_number)
    bs_file_name = 'data/dat/pore_{}_bs.dat'.format(pore_number)
    newton_array = np.loadtxt(newton_file_name)
    bs_array = np.loadtxt(bs_file_name)
    h_ratio = np.log((newton_array[0, 0] / newton_array[-1, 0]))
    newton_ratio = np.log(
        (newton_array[0, error_flag - 1] / newton_array[-1, error_flag - 1]))
    bs_ratio = np.log(
        (bs_array[0, error_flag - 1] / bs_array[-1, error_flag - 1]))
    newton_ratio = "Map-MD (OC={:.3f})".format(newton_ratio / h_ratio)
    bs_ratio = "Map-BS (OC={:.3f})".format(bs_ratio / h_ratio)

    plotting_command = 'plot "' + newton_file_name + '" u 1:' + str(error_flag) + \
                       ' title "' + newton_ratio + '" lc "blue" pt 5 ps 1 lt 1 lw 2 , "' + bs_file_name + '" u 1:' + \
                       str(error_flag) + ' title "' + bs_ratio + \
                       '" lc "red" pt 7 ps 1 lt 1 lw 2'
    # gp.c(plotting_command)

    gp.c('set terminal pdf')
    path = 'data/pdf/results/pore_{}_error_{}.pdf'.format(
        pore_number, error_flag)
    gp.c('set output "' + path + '"')
    gp.c(plotting_command)

    gp.c('set terminal png')
    path = 'data/png/results/pore_{}_error_{}.png'.format(
        pore_number, error_flag)
    gp.c('set output "' + path + '"')
    gp.c(plotting_command)


if __name__ == '__main__':
    L2_flag = 2
    SD_flag = 3
    interface_flag = 4

    convergence_plot_single(4, L2_flag)
    total_pore_number = 9
    for i in range(total_pore_number):
        convergence_plot_single(i, SD_flag)

    for i in range(total_pore_number):
        convergence_plot_single(i, interface_flag)

    # X = np.arange(10)
    # Y = np.sin(X/(2*np.pi))
    # Z = Y**2.0
    # gp.s([X, Y, Z], filename='data/gnuplot/tmp.dat')
    # gp.figure(1)
    # gp.c('plot "data/gnuplot/tmp.dat" u 1:2 title "line1" lc "blue" pt 1 ps 2 lt 1 lw 2, \
    #       "data/gnuplot/tmp.dat" u 1:3 title "line2" lc "red" pt 2 ps 2 lt 1 lw 2')  # point_type point_size line_width
    # gp.c('unset key')
    # gp.pdf(filename='data/gnuplot/myfigure.pdf', width=14, height=9, fontsize=12)
