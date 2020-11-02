import PyGnuplot as gp
import numpy as np

L2_flag = 0
H1_flag = 1
Linf_flag = 2
SD_flag = 3
interface_flag = 4
total_number_errors = 5

NARROW_BAND = 0
GLOBAL = 1

PORE_CASE = 0
STAR_CASE = 1 
SPHERE_CASE = 2
TORUS_CASE = 3

CIRCLE_PORE = 4

def convergence_plot_single(case_number, pore_number, domain_flag, error_flag, size=0.65, debug=False):
    if error_flag == L2_flag:
        label_name = '$\\\\mathcal{{E}}_{L^2}$'  
    elif error_flag == H1_flag:
        label_name = '$\\\\mathcal{{E}}_{H^1}$'  
    elif error_flag == Linf_flag:
        label_name = '$\\\\mathcal{{E}}_{L^{{\\\\infty}}}$'
    elif error_flag == SD_flag:
        label_name = '$\\\\mathcal{{E}}_{{\\\\rm{{SD}}}}$'
    elif error_flag == interface_flag:
        label_name = '$\\\\mathcal{{E}}_{{\\\\rm{{Int}}}}$'
    else:
        assert 0, "Wrong error_flag!"

    gp.c('set terminal epslatex')

    if case_number == PORE_CASE:
        saving_path = 'set output "data/latex/case_{}/pore_{}_domain_flag_{}_error_flag_{}.tex"'.format(case_number, 
            pore_number, domain_flag, error_flag)
    else:
        saving_path = 'set output "data/latex/case_{}/domain_flag_{}_error_flag_{}.tex"'.format(case_number, error_flag)

    if debug == True:
        saving_path = 'set output "example.tex"'

    gp.c(saving_path)

    gp.c('set size {}, {}'.format(size, size))
    # gp.c('set terminal qt ' + str(pore_number))

    # gp.c('set title "Convergence tests" font ",14"')
    gp.c('set xlabel "$h$"')
    gp.c('set ylabel "{}"'.format(label_name))
    # gp.c('set xtics font ",12"')
    # gp.c('set ytics font ",12"')
    gp.c('set style data linespoints')
    # gp.c('set key bottom right')
    gp.c('set logscale')
    gp.c('set format y "$10^{%T}$"')
    # gp.c('set xrange [0.01:0.2]')
    # gp.c('set yrange [1e-8:1e-2]')
    # gp.c('set autoscale')
    gp.c('set lmargin 0.5')
    gp.c('set rmargin 0.5')
    gp.c('set bmargin 0.5')
    gp.c('set tmargin 0.5')
    gp.c('set size square')
    gp.c('set xlabel offset 0,1')
    gp.c('set ylabel offset 1,0')

    if case_number == PORE_CASE:
        newton_file_name = 'data/dat/convergence/case_0/pore_{}_newton.dat'.format(pore_number)
        bs_file_name = 'data/dat/convergence/case_0/pore_{}_bs.dat'.format(pore_number)
        laplace_file_name = 'data/dat/convergence/case_0/pore_{}_laplace.dat'.format(pore_number)

        newton_array = np.loadtxt(newton_file_name)
        bs_array = np.loadtxt(bs_file_name)
        laplace_array = np.loadtxt(laplace_file_name)

        newton_array = newton_array[:, 0:total_number_errors + 1] if domain_flag == GLOBAL else newton_array[:, total_number_errors + 1:]
        bs_array = bs_array[:, 0:total_number_errors + 1] if domain_flag == GLOBAL else bs_array[:, total_number_errors + 1:]

        h_ratio = np.log((newton_array[0, 0] / newton_array[-1, 0]))
        newton_ratio = np.log((newton_array[0, error_flag + 1] / newton_array[-1, error_flag + 1]))
        bs_ratio = np.log((bs_array[0, error_flag + 1] / bs_array[-1, error_flag + 1]))
        laplace_ratio = np.log((laplace_array[0, error_flag + 1] / laplace_array[-1, error_flag + 1]))

        newton_ratio = "$\\\\phi^0 - \\\\boldsymbol{{M}}_a$ (OC={:.3f})".format(newton_ratio / h_ratio)
        bs_ratio = "$\\\\phi^0 - \\\\boldsymbol{{M}}_b$ (OC={:.3f})".format(bs_ratio / h_ratio)
        laplace_ratio = "$\\\\psi - \\\\boldsymbol{{M}}_a$ (OC={:.3f})".format(laplace_ratio / h_ratio)

        error_flag_gnu = error_flag + 2 if domain_flag == GLOBAL else error_flag + total_number_errors + 3
        if domain_flag == GLOBAL:
            plotting_command = 'plot "{}" u 1:{} title "{}" lc "red" pt 5 ps 2 lt 1 lw 4 , \
                "{}" u 1:{} title "{}" lc "blue" pt 7 ps 2 lt 1 lw 4'.format(newton_file_name,
                                                                            error_flag_gnu,
                                                                            newton_ratio,
                                                                            bs_file_name,
                                                                            error_flag_gnu,
                                                                            bs_ratio)
        else:        
            plotting_command = 'plot "{}" u 1:{} title "{}" lc "red" pt 5 ps 2 lt 1 lw 4 , \
                "{}" u 1:{} title "{}" lc "green" pt 9 ps 2 lt 1 lw 4, \
                "{}" u 1:{} title "{}" lc "blue" pt 7 ps 2 lt 1 lw 4'.format(newton_file_name,
                                                                            error_flag_gnu,
                                                                            newton_ratio,
                                                                            laplace_file_name,
                                                                            error_flag + 2,
                                                                            laplace_ratio,
                                                                            bs_file_name,
                                                                            error_flag_gnu,
                                                                            bs_ratio)
    else:
        pass

    gp.c(plotting_command)
    gp.c('set out')

    # gp.c('set terminal pdf')
    # path = 'data/pdf/results/case_{}/pore_{}_error_{}.pdf'.format(case_number, pore_number, error_flag)
    # gp.c('set output "' + path + '"')
    # gp.c(plotting_command)

    # gp.c('set terminal png')
    # path = 'data/png/results/case_{}/pore_{}_error_{}.png'.format(case_number, pore_number, error_flag)
    # gp.c('set output "' + path + '"')
    # gp.c(plotting_command)


def produce_plots():
    for domain_flag in range(0, 2):
        convergence_plot_single(case_number, CIRCLE_PORE, domain_flag, L2_flag)
        convergence_plot_single(case_number, CIRCLE_PORE, domain_flag, H1_flag)
        convergence_plot_single(case_number, CIRCLE_PORE, domain_flag, Linf_flag)
        total_pore_number = 9
        for i in range(total_pore_number):
            convergence_plot_single(case_number, i, domain_flag, SD_flag)
            convergence_plot_single(case_number, i, domain_flag, interface_flag)

        

if __name__ == '__main__':
    case_number = PORE_CASE
    domain_flag = NARROW_BAND
    # convergence_plot_single(case_number, 4, domain_flag, interface_flag, size=0.65, debug=False)
 
    produce_plots() 