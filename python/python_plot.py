import numpy as np
import matplotlib.pyplot as plt

# For latex plotting format
plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


def base_plot(fig_id, x_collection, y_collection, legend_collection, x_label, y_label, 
                title_label, x_log, y_log, legend_fontsize=18, legend_loc=None, 
                figsize=(8, 6), linewidth=4, markersize=12, markers=['s', 'o', '^'], 
                colors=['red', 'blue', 'limegreen'], reference_OC=None, addiontal_legend=None):
    assert x_collection.shape == y_collection.shape, "Data having inconsistent shapes!"

    fig, ax = plt.subplots(num=fig_id, figsize=figsize)
    num_graphs = len(x_collection)

    for i in range(num_graphs):
        plt.plot(x_collection[i], y_collection[i], linestyle='-', marker=markers[i], markersize=markersize, color=colors[i], label=legend_collection[i])

        plt.tick_params(labelsize=18)
        plt.legend(fontsize=legend_fontsize, loc=legend_loc)
        plt.xlabel(x_label, fontsize=20)
        plt.ylabel(y_label, fontsize=22, rotation=90)
        plt.title(title_label, fontsize=18)

        if x_log:
            plt.xscale("log")
        if y_log:
            plt.yscale("log")

        ax.get_xaxis().set_tick_params(which='minor', labelsize=18)
        ax.get_xaxis().set_tick_params(which='major', labelsize=18)
        ax.get_yaxis().set_tick_params(which='minor', labelsize=18)
        ax.get_yaxis().set_tick_params(which='major', labelsize=18)

    if reference_OC is not None:
        p1 = [1.5*get_mesh_size(7), np.min(y_collection[:, -2])]
        p2 = [1.9*get_mesh_size(7), np.min(y_collection[:, -2])]
        p3 = [p2[0], np.exp(reference_OC*np.log(p2[0]/p1[0]))*p1[1]]
        ax.plot([p1[0], p2[0], p3[0], p1[0]], [p1[1], p2[1], p3[1], p1[1]], color='black')  

        plt.text(1.6*get_mesh_size(7), np.exp(0.7*reference_OC*np.log(p2[0]/p1[0]))*p1[1], r'{}'.format(reference_OC), fontsize=18)


    if addiontal_legend is not None:
        plt.annotate(addiontal_legend, (0.05, 0.9), fontsize=18, xycoords='axes fraction')


    return fig


def get_mesh_size(REFINEMENT):
    DOMAIN_LENGTH = 2
    DIVISION = np.power(2, REFINEMENT)
    H = 2 * DOMAIN_LENGTH / DIVISION * np.sqrt(2.)
    return H


def get_y_label(error_type):
    if error_type == 'err_1':
        y_label = r'$\mathcal{E}_{L^{1}}$'
    elif error_type == 'err_2':
        y_label = r'$\mathcal{E}_{L^{2}}$'
    elif error_type == 'err_h1':
        y_label = r'$\mathcal{E}_{H^{1}}$'
    elif error_type == 'err_inf':
        y_label = r'$\mathcal{E}_{L^{\infty}}$'
    elif error_type == 'err_sd':
        y_label = r'$\mathcal{E}_{\textrm{SD}}$'
    elif error_type == 'err_int':
        y_label = r'$\mathcal{E}_{\textrm{Int}}$'
    elif error_type == 'err_vol':
        y_label = r'$\mathcal{E}_{\textrm{Vol}}$'
    else:
        assert 0, "Unknown error type" 
    return y_label


def get_reference_OC(error_type):
    if error_type == 'err_1':
        reference_OC = None
    elif error_type == 'err_2':
        reference_OC = 2
    elif error_type == 'err_h1':
        reference_OC = 1
    elif error_type == 'err_inf':
        reference_OC = None
    elif error_type == 'err_sd':
        reference_OC = 1
    elif error_type == 'err_int':
        reference_OC = None
    elif error_type == 'err_vol':
        reference_OC = None
    else:
        assert 0, "Unknown error type" 
    return reference_OC


def peanut_get_err(error_type, scheme, REFINEMENT):
    return np.load('data/numpy/finite_difference/{}_{}_refinement_{}.npy'.format(error_type, scheme, REFINEMENT))
    

def peanut_plot_iterations_helper(initial_ls, error_type, fig_id):
    y_label = get_y_label(error_type)
    REFINEMENTS = [5, 6, 7, 8]
    y_collection = []
    legend_collection = []
    for REFINEMENT in REFINEMENTS:
        err_fd_subcell = peanut_get_err(error_type, 'fd_subcell-' + initial_ls, REFINEMENT)
        y_collection.append(err_fd_subcell[:11])
        legend_collection.append(r'$h={:.3f}$'.format(get_mesh_size(REFINEMENT)))

    y_collection = np.asarray(y_collection)
    x_collection = np.repeat(np.linspace(0, y_collection.shape[1] - 1, y_collection.shape[1]).reshape(1, -1), y_collection.shape[0], axis=0)
    
    # legend_collection = [r'$h=0.176$', r'$h=0.088$', r'$h=0.044$', r'$h=0.022$']

    x_label = 'Time step'
    title_label =''

    fig = base_plot(fig_id, x_collection, y_collection, legend_collection, x_label, y_label, title_label, False, True, 
        figsize=(10, 6), markersize=8, legend_fontsize=18, linewidth=2, markers=['s', 'o', '^', 'v'], 
                colors=['red', 'blue', 'limegreen', 'purple'])
 
    fig.savefig('data/pdf/finite_difference/steps_{}_{}.pdf'.format(initial_ls, error_type), bbox_inches='tight')


def peanut_plot_iterations():
    initial_ls='quadratic'

    # peanut_plot_iterations_helper(initial_ls, 'err_1', 0)
    # peanut_plot_iterations_helper(initial_ls, 'err_inf', 1)
    peanut_plot_iterations_helper(initial_ls, 'err_int', 2)


def peanut_plot_convergence_helper(initial_ls, error_type, fig_id):
    REFINEMENTS = [5, 6, 7, 8]
    y_label = get_y_label(error_type)
    mesh_sizes = np.array([get_mesh_size(REFINEMENT) for REFINEMENT in REFINEMENTS])
    x_collection = np.repeat(mesh_sizes.reshape(1, -1), 3, axis=0)
    y_collection = np.zeros_like(x_collection)

    for i in range(len(REFINEMENTS)):
        err_fd_simple = peanut_get_err(error_type, 'fd_simple-' + initial_ls, REFINEMENTS[i])
        err_fd_subcell = peanut_get_err(error_type, 'fd_subcell-' + initial_ls, REFINEMENTS[i])
        err_fem = peanut_get_err(error_type, 'case_4-' + initial_ls, REFINEMENTS[i])
        y_collection[0, i] = err_fd_simple[-1]
        y_collection[1, i] = err_fd_subcell[-1]
        y_collection[2, i] = err_fem[-1]

    legend_collection = [r'FDM - standard', r'FDM - subcell fix', r'FEM - this work']

    for i in range(len(x_collection)):
        OC = np.log(y_collection[i][-1] / y_collection[i][0]) / np.log(x_collection[i][-1] / x_collection[i][0])
        legend_collection[i] += r' (OC = {:.2f})'.format(OC)
    
    x_label = r'$h$'
    title_label = ''
    reference_OC = get_reference_OC(error_type)

    addiontal_legend = r'Initial level set: $\phi^{0, a}$' if initial_ls == 'quadratic' else r'Initial level set: $\phi^{0, b}$'

    fig = base_plot(fig_id, x_collection, y_collection, legend_collection, x_label, y_label, title_label, 
        True, True, legend_fontsize=16, legend_loc='lower right', reference_OC=reference_OC, addiontal_legend=addiontal_legend)

    fig.savefig('data/pdf/finite_difference/{}_{}.pdf'.format(initial_ls, error_type), bbox_inches='tight')


def peanut_plot_convergence():
    initial_ls_collection = ['quadratic', 'sin']
    for i in range(len(initial_ls_collection)):
        peanut_plot_convergence_helper(initial_ls_collection[i], 'err_1', 3 * i)
        # peanut_plot_convergence_helper(initial_ls_collection[i], 'err_inf', 3 * i + 1)
        peanut_plot_convergence_helper(initial_ls_collection[i], 'err_int', 3 * i + 2)


def dealii_get_error_column_index(error_type):
    if error_type == 'err_2':
        index = 1
    elif error_type == 'err_h1':
        index = 2
    elif error_type == 'err_inf':
        index = 3
    elif error_type == 'err_sd':
        index = 4
    elif error_type == 'err_int':
        index = 5
    elif error_type == 'err_vol':
        index = 6
    else:
        assert 0, "Unknown error type"
    return index


def dealii_get_legend(narrow_band_info):
    if narrow_band_info == 'global_trivial':
        label = r'$\phi^0$ - full domain'
    elif narrow_band_info == 'narrow_trivial':
        label = r'$\phi^0$ - narrow band'
    elif narrow_band_info == 'narrow_laplace':
        label = r'$\psi$ - narrow band'
    else:
        assert 0, "Unknown narrow band information" 
    return label
  

def dealii_plot_convergece_helper(case_name, error_type, narrow_band_info_set, fig_id):
    y_label = get_y_label(error_type)
    x_collection = []
    y_collection = []
    legend_collection = []
    for narrow_band_info in narrow_band_info_set:
        errors = np.genfromtxt("data/dat/convergence_second_submission/{}{}.dat".format(case_name, narrow_band_info), dtype=float)
        x_collection.append(errors[:, 0])
        y_collection.append(errors[:, dealii_get_error_column_index(error_type)])
        OC = np.log(y_collection[-1][-1] / y_collection[-1][0]) / np.log(x_collection[-1][-1] / x_collection[-1][0])
        legend_collection.append(dealii_get_legend(narrow_band_info) + r' (OC = {:.2f})'.format(OC))

    x_collection = np.asarray(x_collection)
    y_collection = np.asarray(y_collection)

    x_label = r'$h$'
    title_label = ''
    reference_OC = get_reference_OC(error_type)

    fig = base_plot(fig_id, x_collection, y_collection, legend_collection, x_label, y_label, title_label, 
        True, True, legend_fontsize=18, reference_OC=reference_OC)
 
    fig.savefig('data/pdf/convergence_results/{}{}.pdf'.format(case_name, error_type), bbox_inches='tight')


def dealii_plot_convergence():
    cases = ['case_0/pore_0_', 'case_0/pore_4_', 'case_2/', 'case_3/']
    error_types = ['err_2', 'err_h1', 'err_inf', 'err_sd', 'err_int', 'err_vol']
    narrow_band_info = ['global_trivial', 'narrow_trivial', 'narrow_laplace']

    fig_id = 0

    for i in range(3):
        dealii_plot_convergece_helper(cases[0], error_types[i + 3], narrow_band_info[:], fig_id)
        fig_id += 1

    for i in range(6):
        dealii_plot_convergece_helper(cases[1], error_types[i], narrow_band_info[:], fig_id)
        fig_id += 1

    for i in range(6):
        dealii_plot_convergece_helper(cases[2], error_types[i], narrow_band_info[1:2], fig_id)
        fig_id += 1

    for i in range(2):
        dealii_plot_convergece_helper(cases[3], error_types[i + 3], narrow_band_info[1:2], fig_id)
        fig_id += 1


if __name__ == '__main__':
    print("\n")
    peanut_plot_iterations()
    # peanut_plot_convergence()
    # dealii_plot_convergence()
    plt.show()