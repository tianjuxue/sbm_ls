import numpy as np
import matplotlib.pyplot as plt
import sys
import meshio
import glob
import os

# Set some printing formats
# np.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
# np.set_printoptions(precision=3)


def simulator(REFINEMENT, simulate, compute_error, generate_vtk):
    '''
    REFINEMENT is typically 5, 6, 7 or 8 in this project

    There are three key steps:
    1. compute the finite difference solutions and save to .dat files (see time_integrator)
    2. compute errors using .dat files (see post_process_compute_error)
    3. output .vtk files for visualization using .dat files (see post_process_generate_vtk)

    The three flags (simulate, compute_error, generate_vtk) control if the steps should be executed.
    '''

    DOMAIN_LENGTH = 2
    DIVISION = np.power(2, REFINEMENT)
    X1D = np.linspace(-DOMAIN_LENGTH, DOMAIN_LENGTH, DIVISION + 1)
    Y1D = np.linspace(-DOMAIN_LENGTH, DOMAIN_LENGTH, DIVISION + 1)
    X_COO, Y_COO = np.meshgrid(X1D, Y1D, indexing='ij')
    H = 2 * DOMAIN_LENGTH / DIVISION
    CFL = 0.45
    EPS = 1e-10

    NARROW_BAND = 0
    GLOBAL = 1


    def get_active_global():
        '''narrow band solutions may have fewer dofs, so set them as active/inactive
        '''
        active_table = np.zeros((DIVISION + 1, DIVISION + 1))
        active_table = 1 - active_table
        active_indices = set(map(tuple, np.stack(np.where(active_table == 1), axis=1))) # O(1) of query for key in set
        return active_indices


    def get_active_narrow_band(scheme, step):
        '''
        Each row of narrow_band_solutions contains 6 values: cell center x, cell center y, the 4 FEM solutions values at vertices
        active_indices looks like {(i, j):[rank, value], ...} where value is the solution value corresponding to node (i, j)
        active_cells looks like [[(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)], ...]
        '''
        narrow_band_solutions = np.genfromtxt("data/dat/finite_difference/{}/"
            "refinement_{}/solution_step_{}.dat".format(scheme, REFINEMENT, step), dtype=float)
        active_indices = {}
        active_cells = []
        for row in narrow_band_solutions:
            i = int((row[0] + DOMAIN_LENGTH) // H)
            j = int((row[1] + DOMAIN_LENGTH) // H)
            if (i, j) not in active_indices:
                active_indices[(i, j)] = [len(active_indices), row[2]]
            if (i + 1, j) not in active_indices:
                active_indices[(i + 1, j)] = [len(active_indices), row[3]]
            if (i, j + 1) not in active_indices:
                active_indices[(i, j + 1)] = [len(active_indices), row[4]]
            if (i + 1, j + 1) not in active_indices:
                active_indices[(i + 1, j + 1)] = [len(active_indices), row[5]]
            active_cells.append([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])

        narrow_band_cell_centers = narrow_band_solutions[:, :2]
        return active_indices, active_cells, narrow_band_cell_centers


    # def target_sdf(x, y):
    #     return np.sqrt(x*x + y*y) - 1

    # def initial_level_set(x, y):
    #     return x*x + y*y - 1

    # def initial_level_set(x, y):
    #     return ((x - 1)**2 + (y - 1)**2 + 0.1) * target_sdf


    def target_sdf(x, y):
        r = 1.;
        a = 0.7;
        b = np.sqrt(np.power(r, 2.) - np.power(a, 2.));
        if (a - x) / np.sqrt((np.power(a - x, 2.) + np.power(y, 2.))) >= a / r and (a + x) / np.sqrt((np.power(a + x, 2.) + np.power(y, 2.))) >= a / r:
            if y >= 0:
                value = -np.sqrt(np.power(x, 2.) +  np.power(y - b, 2.));
            else:
                value = -np.sqrt(np.power(x, 2.) +  np.power(y + b, 2.)); 
        else:
            if x >= 0:
                value = np.sqrt(np.power(x - a, 2.) + np.power(y, 2.)) - r;
            else:
                value = np.sqrt(np.power(x + a, 2.) + np.power(y, 2.)) - r;
        return value


    def quadratic_function(x, y):
        return ((x - 1)**2 + (y - 1)**2 + 0.1)


    def sin_function(x, y):
        k = 6
        return 0.5 * np.sin(k * np.pi * x) * np.sin(k * np.pi * y) + 1


    def initial_level_set(x, y, scheme):
        if 'quadratic' in scheme:
            return quadratic_function(x, y) * target_sdf(x, y)
        elif 'sin' in scheme:
            return sin_function(x, y) * target_sdf(x, y)
        else:
            assert 0, "Unknown initial level set!"


    # Should use numpy where or something similar to broadcast, being lazy here...
    def initial_level_set_interpolation(scheme):
        solution = np.zeros_like(X_COO)
        for i in range(solution.shape[0]):
            for j in range(solution.shape[1]):
                solution[i, j] = initial_level_set(X_COO[i, j], Y_COO[i, j], scheme)
        return solution


    # FEM uses projection as the initialization, not interpolation. 
    # So let us be consistent
    def initial_level_set_projection(active_indices):
        solution = np.zeros_like(X_COO)
        for (i, j), value in active_indices.items():
            solution[i, j] = value[1]
        return solution


    def hard_sign_func(x):
        return 1. if x >= 0 else -1


    def soft_sign_func(x):
        return x / np.sqrt(x**2 + H**2) 


    def minmod(a, b):
        '''The minmod function is zero when the two arguments have different signs, 
        and takes the argument with smaller absolute value when the two have the same sign.
        '''
        if a * b > 0:
            if np.absolute(a) <= np.absolute(b):
                return a
            else:
                return b
        else:
            return 0.


    def H_Godunov(a, b, c, d, sgn_phi_0):
        if sgn_phi_0 >= 0:
            return np.sqrt(max(min(a, 0)**2, max(b, 0)**2) + max(min(c, 0)**2, max(d, 0)**2))
        else:
            return np.sqrt(max(max(a, 0)**2, min(b, 0)**2) + max(max(c, 0)**2, min(d, 0)**2))

    def D_xx(i, j, solution, active_indices):
        if (i, j) not in active_indices or \
           (i - 1, j) not in active_indices or \
           (i + 1, j) not in active_indices:
            return 0.
        return (solution[i - 1, j] - 2*solution[i, j] + solution[i + 1, j]) / (H**2)


    def D_yy(i, j, solution, active_indices):
        if (i, j) not in active_indices or \
           (i, j - 1) not in active_indices or \
           (i, j + 1) not in active_indices:
            return 0.
        return (solution[i, j - 1] - 2*solution[i, j] + solution[i, j + 1]) / (H**2)


    def D_x_plus(i, j, solution, active_indices, scheme):
        if (i, j) not in active_indices or \
           (i + 1, j) not in active_indices:
           return H, 0.
        Delta_x_plus_val, func_val = Delta_x_plus(i, j, solution, active_indices, scheme)
        return Delta_x_plus_val, (func_val - solution[i, j]) / Delta_x_plus_val - \
               Delta_x_plus_val / 2 * minmod(D_xx(i, j, solution, active_indices), D_xx(i + 1, j, solution, active_indices))


    def D_x_minus(i, j, solution, active_indices, scheme):
        if (i, j) not in active_indices or \
           (i - 1, j) not in active_indices:
           return H, 0.
        Delta_x_minus_val, func_val = Delta_x_minus(i, j, solution, active_indices, scheme)
        return Delta_x_minus_val, (solution[i, j] - func_val) / Delta_x_minus_val + \
               Delta_x_minus_val / 2 * minmod(D_xx(i, j, solution, active_indices), D_xx(i - 1, j, solution, active_indices))


    def D_y_plus(i, j, solution, active_indices, scheme):
        if (i, j) not in active_indices or \
           (i, j + 1) not in active_indices:
           return H, 0.
        Delta_y_plus_val, func_val = Delta_y_plus(i, j, solution, active_indices, scheme)
        return Delta_y_plus_val, (func_val - solution[i, j]) / Delta_y_plus_val - \
               Delta_y_plus_val / 2 * minmod(D_yy(i, j, solution, active_indices), D_yy(i, j + 1, solution, active_indices))


    def D_y_minus(i, j, solution, active_indices, scheme):
        if (i, j) not in active_indices or \
           (i, j - 1) not in active_indices:
           return H, 0.
        Delta_y_minus_val, func_val = Delta_y_minus(i, j, solution, active_indices, scheme)
        return Delta_y_minus_val, (solution[i, j] - func_val) / Delta_y_minus_val + \
               Delta_y_minus_val / 2 * minmod(D_yy(i, j, solution, active_indices), D_yy(i, j - 1, solution, active_indices))


    def Delta_helper(sgn_phi_0_self, sgn_phi_0_neighbour, sgn_phi_0_self_friend, sgn_phi_0_neighbour_friend):
        sgn_phi_0_xx = minmod(sgn_phi_0_self_friend - 2 * sgn_phi_0_self + sgn_phi_0_neighbour, 
                              sgn_phi_0_self - 2 * sgn_phi_0_neighbour + sgn_phi_0_neighbour_friend)
        if np.absolute(sgn_phi_0_xx) > EPS:
            D = (sgn_phi_0_xx / 2. - sgn_phi_0_self - sgn_phi_0_neighbour)**2 - 4 * sgn_phi_0_self * sgn_phi_0_neighbour
            return H * (1./2. + (sgn_phi_0_self - sgn_phi_0_neighbour - hard_sign_func(sgn_phi_0_self - sgn_phi_0_neighbour)*np.sqrt(D)) / sgn_phi_0_xx)
        else:
            return H * sgn_phi_0_self / (sgn_phi_0_self - sgn_phi_0_neighbour)


    def Delta_x_plus(i, j, solution, active_indices, scheme):
        sgn_phi_0_self = initial_level_set(X_COO[i, j], Y_COO[i, j], scheme)
        sgn_phi_0_neighbour = initial_level_set(X_COO[i + 1, j], Y_COO[i + 1, j], scheme)
        if sgn_phi_0_self * sgn_phi_0_neighbour >= 0 or 'fd_simple' in scheme:
            return H, solution[i + 1, j]
        else:
            assert (i - 1, j) in active_indices and (i + 2, j) in active_indices
            sgn_phi_0_self_friend = initial_level_set(X_COO[i - 1, j], Y_COO[i - 1, j], scheme)
            sgn_phi_0_neighbour_friend = initial_level_set(X_COO[i + 2, j], Y_COO[i + 2, j], scheme)
            return Delta_helper(sgn_phi_0_self, sgn_phi_0_neighbour, sgn_phi_0_self_friend, sgn_phi_0_neighbour_friend), 0.


    def Delta_x_minus(i, j, solution, active_indices, scheme):
        sgn_phi_0_self = initial_level_set(X_COO[i, j], Y_COO[i, j], scheme)
        sgn_phi_0_neighbour = initial_level_set(X_COO[i - 1, j], Y_COO[i - 1, j], scheme)
        if sgn_phi_0_self * sgn_phi_0_neighbour >= 0 or 'fd_simple' in scheme:
            return H, solution[i - 1, j]
        else:
            assert (i + 1, j) in active_indices and (i - 2, j) in active_indices
            sgn_phi_0_self_friend = initial_level_set(X_COO[i + 1, j], Y_COO[i + 1, j], scheme)
            sgn_phi_0_neighbour_friend = initial_level_set(X_COO[i - 2, j], Y_COO[i - 2, j], scheme)
            return Delta_helper(sgn_phi_0_self, sgn_phi_0_neighbour, sgn_phi_0_self_friend, sgn_phi_0_neighbour_friend), 0.
           

    def Delta_y_plus(i, j, solution, active_indices, scheme):
        sgn_phi_0_self = initial_level_set(X_COO[i, j], Y_COO[i, j], scheme)
        sgn_phi_0_neighbour = initial_level_set(X_COO[i, j + 1], Y_COO[i, j + 1], scheme)
        if sgn_phi_0_self * sgn_phi_0_neighbour >= 0 or 'fd_simple' in scheme:
            return H, solution[i, j + 1]
        else:
            assert (i, j - 1) in active_indices and (i, j + 2) in active_indices
            sgn_phi_0_self_friend = initial_level_set(X_COO[i, j - 1], Y_COO[i, j - 1], scheme)
            sgn_phi_0_neighbour_friend = initial_level_set(X_COO[i, j + 2], Y_COO[i, j + 2], scheme)
            return Delta_helper(sgn_phi_0_self, sgn_phi_0_neighbour, sgn_phi_0_self_friend, sgn_phi_0_neighbour_friend), 0.


    def Delta_y_minus(i, j, solution, active_indices, scheme):
        sgn_phi_0_self = initial_level_set(X_COO[i, j], Y_COO[i, j], scheme)
        sgn_phi_0_neighbour = initial_level_set(X_COO[i, j - 1], Y_COO[i, j - 1], scheme)
        if sgn_phi_0_self * sgn_phi_0_neighbour >= 0 or 'fd_simple' in scheme:
            return H, solution[i, j - 1]
        else:
            assert (i, j + 1) in active_indices and (i, j - 2) in active_indices
            sgn_phi_0_self_friend = initial_level_set(X_COO[i, j + 1], Y_COO[i, j + 1], scheme)
            sgn_phi_0_neighbour_friend = initial_level_set(X_COO[i, j - 2], Y_COO[i, j - 2], scheme)
            return Delta_helper(sgn_phi_0_self, sgn_phi_0_neighbour, sgn_phi_0_self_friend, sgn_phi_0_neighbour_friend), 0.


    ###############################################################################################
    # Step 2: compute errors

    def error_L_inf(solution, active_indices, scheme):
        L_inf = 0.
        for (i, j) in active_indices:
            sol = solution[i, j] if scheme == 'debug' else active_indices[(i, j)][1] 
            # if sol > -0.8 and np.absolute(X_COO[i, j]) + np.absolute(Y_COO[i, j]) > EPS:
            # if np.absolute(sol) < 1.2 * H:
            local_error = np.absolute(sol - target_sdf(X_COO[i, j], Y_COO[i, j]))
            if local_error > L_inf:
                L_inf = local_error
        return L_inf


    def error_L_1(solution, active_indices, scheme):
        L_1 = 0.
        for (i, j) in active_indices:
            sol = solution[i, j] if scheme == 'debug' else active_indices[(i, j)][1] 
            local_error = np.absolute(sol - target_sdf(X_COO[i, j], Y_COO[i, j]))
            L_1 += local_error * H**2
        return L_1


    def interpolate(x, y, solution, active_indices, scheme):    
        i = int((x + DOMAIN_LENGTH) // H)
        j = int((y + DOMAIN_LENGTH) // H)
        assert (i, j) in active_indices and (i + 1, j) in active_indices and \
        (i, j + 1) in active_indices and (i + 1, j + 1) in active_indices

        if scheme == 'debug':
            sol11 = solution[i, j]
            sol21 = solution[i + 1, j]
            sol12 = solution[i, j + 1]
            sol22 = solution[i + 1, j + 1]
        else:
            sol11 = active_indices[(i, j)][1]
            sol21 = active_indices[(i + 1, j)][1]
            sol12 = active_indices[(i, j + 1)][1]
            sol22 = active_indices[(i + 1, j + 1)][1]

        xi = (x + DOMAIN_LENGTH - i * H) / H
        eta = (y + DOMAIN_LENGTH - j * H) / H

        value = sol11 * (1 - xi) * (1 - eta) + \
                sol21 * xi * (1 - eta) + \
                sol12 * (1 - xi) * eta + \
                sol22 * xi * eta

        return value


    def error_interface_helper(solution, active_indices, quad_x, quad_y, scheme):
        num_intervals = len(quad_x) - 1
        error = 0
        for i in range(num_intervals):
            x1 = quad_x[i]
            y1 = quad_y[i]
            x2 = quad_x[i + 1]
            y2 = quad_y[i + 1]
            value_start = interpolate(x1, y1, solution, active_indices, scheme)**2
            value_end = interpolate(x2, y2, solution, active_indices, scheme)**2
            interval_measure = np.sqrt(np.power(x1 - x2, 2.) + np.power(y1 - y2, 2.))
            error += (value_start + value_end) / 2 * interval_measure
        return error


    def error_interface(solution, active_indices, scheme):
        num_intervals = 100000 # Computational bottleneck, how can we make sure the quad points are large enough?
        left_quad_x =  np.cos(np.linspace(np.arccos(0.7 / 1.), 2 * np.pi - np.arccos(0.7 / 1.), num_intervals + 1)) - 0.7
        left_quad_y =  np.sin(np.linspace(np.arccos(0.7 / 1.), 2 * np.pi - np.arccos(0.7 / 1.), num_intervals + 1))
        right_quad_x =  np.cos(np.linspace(np.arccos(0.7 / 1.) - np.pi, np.pi - np.arccos(0.7 / 1.), num_intervals + 1)) + 0.7
        right_quad_y =  np.sin(np.linspace(np.arccos(0.7 / 1.) - np.pi, np.pi - np.arccos(0.7 / 1.), num_intervals + 1))

        error_left = error_interface_helper(solution, active_indices, left_quad_x, left_quad_y, scheme)
        error_right = error_interface_helper(solution, active_indices, right_quad_x, right_quad_y, scheme)

        return np.sqrt(error_left + error_right)


    def batch_error(scheme):
        err_Linf = []
        err_interface = []
        err_L1 = []
        files = os.listdir('data/dat/finite_difference/{}/refinement_{}/'.format(scheme, REFINEMENT))
        num_files = len(files)
        # NOT IMPLEMENTED WARNING!
        # We need to switch between the following few options (mostly for plotting)
        # Right now, the switch is controlled by commenting
        steps = [1000]
        # steps = [i for i in range(11)]
        for step in steps:
        # for step in range(num_files):
            if step % 1 == 0:
                active_indices, active_cells, _ = get_active_narrow_band(scheme, step)
                e_L1 = error_L_1(None, active_indices, scheme)
                e_Linf = error_L_inf(None, active_indices, scheme)
                e_interface = error_interface(None, active_indices, scheme)
                # e_Linf = 0.
                # e_interface = 0.
                print("Refinement {}: step {}, {} L_1 error is {}, L_inf error is {}" 
                    " interface error is {}".format(REFINEMENT, step, scheme, e_L1, e_Linf, e_interface))
                err_L1.append(e_L1)
                err_Linf.append(e_Linf)
                err_interface.append(e_interface)
        err_L1 = np.array(err_L1)
        err_Linf = np.array(err_Linf)
        err_interface = np.array(err_interface)

        np.save('data/numpy/finite_difference/err_1_{}_refinement_{}.npy'.format(scheme, REFINEMENT), err_L1)
        np.save('data/numpy/finite_difference/err_inf_{}_refinement_{}.npy'.format(scheme, REFINEMENT), err_Linf)
        np.save('data/numpy/finite_difference/err_int_{}_refinement_{}.npy'.format(scheme, REFINEMENT), err_interface)


    ###############################################################################################
    # Step 3: output .vtk files

    def clean_folder(directory_path):
        files = glob.glob(directory_path)
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                print('Failed to delete {}, reason: {}' % (f, e))


    def output_vtk_global(solution, scheme, step):
        if step == 0:
            clean_folder('data/vtk/finite_difference/global/{}/refinement_{}/*'.format(scheme, REFINEMENT))
        
        cells = []
        for i in range(DIVISION):
            for j in range(DIVISION):
                cells.append([i*(DIVISION + 1) + j, (i + 1)*(DIVISION + 1) + j, (i + 1)*(DIVISION + 1) + j + 1, i*(DIVISION + 1) + j + 1])
        cells = [('quad', np.array(cells))]

        X_COO_flat = X_COO.flatten()
        Y_COO_flat = Y_COO.flatten()
        Z_COO_flat = np.zeros_like(X_COO_flat)
        solution_flat = solution.flatten()
        points = np.stack((X_COO_flat, Y_COO_flat, Z_COO_flat), axis=1)
        point_data = {'u': solution_flat}
        meshio.Mesh(points, cells, point_data=point_data).write("data/vtk/"
            "finite_difference/global/{}/refinement_{}/u{}.vtk".format(scheme, REFINEMENT, step))


    def output_vtk_narrow_band(solution, active_indices, active_cells, scheme, step):
        if step == 0:
            clean_folder('data/vtk/finite_difference/narrow_band/{}/refinement_{}/*'.format(scheme, REFINEMENT))
        
        cells = []
        for cell in active_cells:
            cells.append([active_indices[cell[0]][0], active_indices[cell[1]][0], active_indices[cell[2]][0], active_indices[cell[3]][0]])
        cells = [('quad', np.array(cells))]

        X_COO_flat = np.zeros(len(active_indices))
        Y_COO_flat = np.zeros(len(active_indices))
        Z_COO_flat = np.zeros_like(X_COO_flat)
        solution_flat = np.zeros(len(active_indices))
        for (i, j), value in active_indices.items():
            X_COO_flat[value[0]] = X_COO[i, j]
            Y_COO_flat[value[0]] = Y_COO[i, j]
            if scheme == 'debug':
                solution_flat[value[0]] = solution[i, j]
            else:
                solution_flat[value[0]] = value[1] 
        points = np.stack((X_COO_flat, Y_COO_flat, Z_COO_flat), axis=1)
        point_data = {'u': solution_flat}
        meshio.Mesh(points, cells, point_data=point_data).write("data/vtk/"
            "finite_difference/narrow_band/{}/refinement_{}/u_{}.vtk".format(scheme, REFINEMENT, step))


    def batch_dat_to_vtk(scheme):
        files = os.listdir('data/dat/finite_difference/{}/refinement_{}/'.format(scheme, REFINEMENT))
        num_files = len(files)
        for step in range(num_files):
            active_indices, active_cells, _ = get_active_narrow_band(scheme, step)
            if step % 10 == 0:
                print("Processing {}".format(step))
            output_vtk_narrow_band(None, active_indices, active_cells, scheme, step)


    ###############################################################################################
    # Step 1: finite difference simulations and save to .dat files

    def save_solution_to_dat(solution, active_cells, narrow_band_cell_centers, step, scheme):
        if step == 0:
            clean_folder('data/dat/finite_difference/{}/refinement_{}/*'.format(scheme, REFINEMENT))

        num_vertices = 4
        solution_summary = np.zeros((len(active_cells), num_vertices))
        for i in range(len(active_cells)):
            m, k = active_cells[i][0]
            solution_summary[i, 0] = solution[m, k]
            solution_summary[i, 1] = solution[m + 1, k]
            solution_summary[i, 2] = solution[m, k + 1]
            solution_summary[i, 3] = solution[m + 1, k + 1]
        solution_summary = np.concatenate((narrow_band_cell_centers, solution_summary), axis=1)
        np.savetxt("data/dat/finite_difference/"
            "{}/refinement_{}/solution_step_{}.dat".format(scheme, REFINEMENT, step), solution_summary)


    def time_integrator(domain_flag, scheme):
        compute_intermediate_results = False # flag to compute error and save vtk (debug mode)
        if domain_flag == NARROW_BAND:
            if 'quadratic' in scheme:
                fem_scheme = 'case_4-quadratic'
            elif 'sin' in scheme:
                fem_scheme = 'case_4-sin'
            else:
                assert 0, "Unknown scheme!"
            active_indices, active_cells, narrow_band_cell_centers = get_active_narrow_band(fem_scheme, 0) # case_4 contains FEM results
            # solution_old = initial_level_set_projection(active_indices)
            solution_old = initial_level_set_interpolation(scheme)
        else:
            active_indices = get_active_global()
            solution_old = initial_level_set_interpolation(scheme)
        
        solution_new = np.array(solution_old, copy=True)

        def output(step):
            print("Refinement {}: step {}, {}".format(REFINEMENT, step, scheme))
            if domain_flag == NARROW_BAND:
                if compute_intermediate_results: 
                    output_vtk_narrow_band(solution_new, active_indices, active_cells, 'debug', step)
                save_solution_to_dat(solution_new, active_cells, narrow_band_cell_centers, step, scheme)
            else:
                output_vtk_global(solution_new, 'debug', step)

            if compute_intermediate_results or domain_flag == GLOBAL:
                L_1 = error_L_1(solution_new, active_indices, 'debug')
                # L_inf = error_L_inf(solution_new, active_indices, 'debug')
                # L_int = error_interface(solution_new, active_indices, 'debug')

                active_indices_step, _, _ = get_active_narrow_band(fem_scheme, step)
                L_1_FEM = error_L_1(None, active_indices_step, fem_scheme)
                # L_inf_FEM = error_L_inf(None, active_indices_step, fem_scheme)
                # L_int_FEM = error_interface(None, active_indices_step, fem_scheme)

                print("FD  L_1 error is {}".format(L_1))
                print("FEM L_1 error is {}".format(L_1_FEM))
                print("\n")

                # print("FD  L_inf error is {}".format(L_inf))
                # print("FEM L_inf error is {}".format(L_inf_FEM))
                # print("\n")
                
                # print("FD  interface error is {}".format(L_int))
                # print("FEM interface error is {}".format(L_int_FEM))
                # print("\n")

        output(0)
        for step in range(1000):
            for (i, j) in active_indices:
                sgn_phi_0 = initial_level_set(X_COO[i, j], Y_COO[i, j], scheme)
                Delta_x_plus_val, D_x_plus_val = D_x_plus(i, j, solution_old, active_indices, scheme)
                Delta_x_minus_val, D_x_minus_val = D_x_minus(i, j, solution_old, active_indices, scheme)
                Delta_y_plus_val, D_y_plus_val = D_y_plus(i, j, solution_old, active_indices, scheme)
                Delta_y_minus_val, D_y_minus_val =  D_y_minus(i, j, solution_old, active_indices, scheme)
                H_G = H_Godunov(D_x_plus_val, D_x_minus_val, D_y_plus_val,D_y_minus_val, sgn_phi_0)
                Delta_t = CFL * np.min(np.array([Delta_x_plus_val, Delta_x_minus_val, Delta_y_plus_val, Delta_y_minus_val]))
                if 'fd_subcell' in scheme:
                    solution_new[i, j] = solution_old[i, j] - Delta_t * hard_sign_func(sgn_phi_0) * (H_G - 1)
                elif 'fd_simple' in scheme:
                    solution_new[i, j] = solution_old[i, j] - Delta_t * soft_sign_func(sgn_phi_0) * (H_G - 1)
                else:
                    assert 0, "Unknown scheme!"

            solution_old = np.array(solution_new, copy=True)
            output(step + 1)


    def step1_finite_difference_simulation():
        '''C++ code yields .dat files for FEM computation, so no simulation for FEM.
        '''
        time_integrator(NARROW_BAND, 'fd_simple-quadratic')
        time_integrator(NARROW_BAND, 'fd_subcell-quadratic')
        time_integrator(NARROW_BAND, 'fd_simple-sin')
        time_integrator(NARROW_BAND, 'fd_subcell-sin')

        # To compare with literature
        # time_integrator(GLOBAL, 'fd_subcell-quadratic')
 

    def step2_post_process_compute_error():
        batch_error('fd_simple-quadratic')
        batch_error('fd_subcell-quadratic')
        batch_error('case_4-quadratic')
        batch_error('fd_simple-sin')
        batch_error('fd_subcell-sin')
        batch_error('case_4-sin')


    def step3_post_process_generate_vtk():
        batch_dat_to_vtk('fd_simple-quadratic')
        batch_dat_to_vtk('fd_subcell-quadratic')
        batch_dat_to_vtk('case_4-quadratic')
        batch_dat_to_vtk('fd_simple-sin')
        batch_dat_to_vtk('fd_subcell-sin')
        batch_dat_to_vtk('case_4-sin')


    if simulate:
        step1_finite_difference_simulation()

    if compute_error:
        step2_post_process_compute_error()
 
    if generate_vtk:
        step3_post_process_generate_vtk()


def run():
    REFINEMENTS = [5, 6, 7, 8]
    for REFINEMENT in REFINEMENTS:
        simulator(REFINEMENT=REFINEMENT, simulate=False, compute_error=False, generate_vtk=False)
  
    # simulator(REFINEMENT=8, simulate=True, compute_error=False, generate_vtk=False)


if __name__ == '__main__':
    print("\n")
    run()
