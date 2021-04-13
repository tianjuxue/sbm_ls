import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.mplot3d import Axes3D  
import PyGnuplot as gp


np.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
np.set_printoptions(precision=5)
np.random.seed(1)

DIM = 3
DOMAIN_SIZE = 2
REFERENCE_SIZE = 1
DIVISION = 2
surface = 'sphere'
ORDER = 1
NUM_DIRECTIONS = 2


def torus2(x, y, z):
    value = 2*y*(y**2 - 3*x**2)*(1 - z**2) + (x**2 + y**2)**2 - (9*z**2 - 1)*(1 - z**2)
    return value

def grad_torus2(x, y, z):
    grad_x = 4 * x * (np.power(x, 2) + np.power(y, 2)) - 12 * x * y * (1 -  np.power(z, 2))
    grad_y = 2 * (1 - np.power(z, 2)) * (np.power(y, 2) - 3 * np.power(x, 2)) + 4 * y * (np.power(x, 2) + np.power(y, 2)) + 4 * np.power(y, 2) * (1 - np.power(z, 2))
    grad_z = -4 * y * z * (np.power(y, 2) - 3 * np.power(x, 2)) - 18 * (1 - np.power(z, 2)) * z + 2 * (9 * np.power(z, 2) - 1) * z
    grad = np.array([grad_x, grad_y, grad_z])
    return grad

def sphere(x, y, z):
    value = np.power(x, 2) + np.power(y, 2) + np.power(z, 2) - 1
    return value

def grad_sphere(x, y, z):
    grad_x = 2*x
    grad_y = 2*y
    grad_z = 2*z
    grad = np.array([grad_x, grad_y, grad_z])
    return grad

def plane(x, y, z):
    value = x - 1./3.
    return value

def grad_plane(x, y, z):
    return np.array([1., 0., 0.])

def quad_points_volume(density=4):
    quad_points = []
    step = 2 * REFERENCE_SIZE / density
    for i in range(density):
        for j in range(density):
            for k in range(density):
                quad_points.append(np.array([-REFERENCE_SIZE + step / 2. + i * step,  
                                             -REFERENCE_SIZE + step / 2. + j * step, 
                                             -REFERENCE_SIZE + step / 2. + k * step]))
    return np.asarray(quad_points)

def quad_points_surface(density=8):
    step = 2 * REFERENCE_SIZE / density
    weight = np.power(2 * REFERENCE_SIZE, 2) / np.power(density, 2)
    points_collection = []
    normal_vectors = []
    for d in range(DIM):
        for r in range(NUM_DIRECTIONS):
            quad_points = np.zeros((np.power(density, 2), DIM))
            for i in range(density):
                for j in range(density):
                    quad_points[i * density + j, d] = (2 * r - 1) * REFERENCE_SIZE
                    quad_points[i * density + j, (d + 1) % DIM] = -REFERENCE_SIZE + step / 2. + i * step
                    quad_points[i * density + j, (d + 2) % DIM] = -REFERENCE_SIZE + step / 2. + j * step
            points_collection.append(quad_points)
            normal_vector = np.zeros(DIM)
            normal_vector[d] = (2 * r - 1) * REFERENCE_SIZE
            normal_vectors.append(normal_vector)
    return np.asarray(points_collection), np.asarray(normal_vectors), weight

def divergence_free_functions(ref_point):
    x = ref_point[0]
    y = ref_point[1]
    z = ref_point[2]
    f0 = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    f1 = np.array([[z, 0., 0.], [0., z, 0.], [y, 0., 0.], [0., y, -z], [0., 0., y], [x, 0., -z], [0., x, 0.], [0., 0., x]])
    f2 = np.array([[z*z, 0., 0.], [0., z*z, 0.], [y*z, 0., 0.], [0., 2*y*z, -z*z], [y*y, 0., 0.], [0., y*y, -2*y*z], [0., 0., y*y], 
                   [2*x*z, 0, -z*z], [0., x*z, 0.], [x*y, 0., -y*z], [0, x*y, -x*z], [0., 0., x*y], [x*x, 0., -2*x*z], [0, x*x, 0], [0., 0., x*x]])

    if ORDER == 0:
        function_collection = f0
    elif ORDER == 1:
        function_collection = np.concatenate((f0, f1), axis=0)
    elif ORDER == 2:
        function_collection = np.concatenate((f0, f1, f2), axis=0)
    else:
        assert 0

    return function_collection


def level_set(point):
    if len(point.shape) == 1:
        x = point[0]
        y = point[1]
        z = point[2]
    else:
        x = point[:, 0]
        y = point[:, 1]
        z = point[:, 2]

    if surface == 'torus2':
        return torus2(x, y, z)
    elif surface == 'sphere':
        print("pay your price for using global variables (value)")
        assert 0
        return sphere(x, y, z)
    elif surface == 'plane':
        return plane(x, y, z)
    else:
        assert 0

def grad_level_set(point):
    x = point[0]
    y = point[1]
    z = point[2]
    if surface == 'torus2':
        return grad_torus2(x, y, z)
    elif surface == 'sphere':
        print("pay your price for using global variables (grad)")
        return grad_sphere(x, y, z)
    elif surface == 'plane':
        return grad_plane(x, y, z)
    else:
        assert 0

def normal_ref(ref_point, point_c, scale):
    physical_point = to_physical(ref_point, point_c, scale)
    grad = grad_level_set(physical_point)
    return grad / np.linalg.norm(grad)

def level_set_ref(ref_point, point_c, scale):
    physical_point = to_physical(ref_point, point_c, scale)
    return level_set(physical_point)

def heaviside_ref(ref_point, point_c, scale):
    physical_point = to_physical(ref_point, point_c, scale)
    value = level_set(physical_point)
    return heaviside(-value)

def heaviside(value):
    return 1 if value > 0 else 0

def to_reference(physical_point, point_c, scale):
    return (physical_point - point_c) / scale

def to_physical(ref_point, point_c, scale):
    return scale * ref_point + point_c

def to_id_xyz(element_id, base):
    id_z = element_id % base
    element_id = element_id // base
    id_y = element_id % base
    element_id = element_id // base    
    id_x = element_id % base
    element_id = element_id // base
    return id_x, id_y, id_z 

def to_id(id_x, id_y, id_z, base):
    return id_x * np.power(base, 2) + id_y * base + id_z

def get_vertices(id_x, id_y, id_z, h):
    vertices = []
    vertices_per_direction = 2
    for i in range(vertices_per_direction):
        for j in range(vertices_per_direction):
            for k in range(vertices_per_direction):
                vertices.append(np.array([-DOMAIN_SIZE + (id_x + i) * h, -DOMAIN_SIZE + (id_y + j) * h, -DOMAIN_SIZE + (id_z + k) * h]))
    return vertices

def breakout_id(element_id, base):
    id_x, id_y, id_z = to_id_xyz(element_id, base)
    new_ids = []
    vertices_per_direction = 2
    for i in range(vertices_per_direction):
        for j in range(vertices_per_direction):
            for k in range(vertices_per_direction):
                new_ids.append(to_id(DIVISION * id_x + i, DIVISION * id_y + j, DIVISION * id_z + k, DIVISION * base))
    return new_ids

def brute_force(base):
    ids_cut = []
    h = 2 * DOMAIN_SIZE / base
    for id_x in range(base):
        print("id_x is {}".format(id_x))
        print(len(ids_cut) / np.power(base, 3))
        for id_y in range(base):
            for id_z in range(base):
                vertices = get_vertices(id_x, id_y, id_z, h)
                cut_flag, _, _ = is_cut(vertices)
                if cut_flag:
                    ids_cut.append(to_id(id_x, id_y, id_z, base))

    return ids_cut


def is_cut(vertices):
    negative_flag = False
    positive_flag = False
    for vertice in vertices:
        value = level_set(vertice)
        if value >= 0:
            positive_flag = True
        else:
            negative_flag = True
    return negative_flag and positive_flag, negative_flag, positive_flag


def point_c_and_scale(element_id, base):
    id_x, id_y, id_z = to_id_xyz(element_id, base)
    h = 2 * DOMAIN_SIZE / base
    point_c =  np.array([-DOMAIN_SIZE + (id_x + 1./2.) * h, -DOMAIN_SIZE + (id_y + 1./2.) * h, -DOMAIN_SIZE + (id_z + 1./2.) * h ])
    scale = h / (2 * REFERENCE_SIZE)
    return point_c, scale


def generate_cut_elements():
    start_refinement_level = 5
    end_refinement_level = 7
    start_base = np.power(DIVISION, start_refinement_level)
    ids_cut = brute_force(start_base)
    total_ids = []
    total_refinement_levels = []
    total_ids.append(ids_cut)
    total_refinement_levels.append(start_refinement_level)
    print("refinement_level {}, length of inds {}".format(start_refinement_level, len(ids_cut)))
    for refinement_level in range(start_refinement_level, end_refinement_level):
        ids_cut_new = []
        base = np.power(DIVISION, refinement_level)
        h = 2 * DOMAIN_SIZE / base
        for element_id in ids_cut:
            sub_ids = breakout_id(element_id, base)
            for sub_id in sub_ids:
                sub_id_x, sub_id_y, sub_id_z = to_id_xyz(sub_id, base * DIVISION)
                cut_flag, _, _ = is_cut(get_vertices(sub_id_x, sub_id_y, sub_id_z, h / DIVISION))
                if cut_flag:
                    ids_cut_new.append(sub_id)
        ids_cut = ids_cut_new
        total_ids.append(ids_cut)
        total_refinement_levels.append(refinement_level + 1)
        print("refinement_level {}, length of inds {}".format(refinement_level + 1, len(ids_cut)))

    print("len of total_refinement_levels {}".format(len(total_refinement_levels)))
    np.savez('data/numpy/sbi/{}_cut_element_ids.npz'.format(surface), ids=total_ids, refinement_level=total_refinement_levels, allow_pickle=True)
    return total_ids, total_refinement_levels


def main():
    data = np.load('data/numpy/sbi/{}_cut_element_ids.npz'.format(surface), allow_pickle=True)
    total_ids = data['ids']
    total_refinement_levels = data['refinement_level']
    
    index = -1
    ids_cut = total_ids[index]
    refinement_level = total_refinement_levels[index]
    base = np.power(DIVISION, refinement_level)
    h = 2 * DOMAIN_SIZE / base

    print("refinement_level is {} with h being {}, number of elements cut is {}".format(refinement_level, h, len(ids_cut)))

    ground_truth = 4 * np.pi
    hmf_result = 0

    if ORDER == 0:
        K = 3 # number of divergence-free functions
    elif ORDER == 1:
        K = 11
    elif ORDER == 2:
        K = 26
    else:
        assert 0

    quad_points_to_save = []
    weights_to_save = []

    for ele in range(0, len(ids_cut)):
    # for ele in range(0, 3):
        element_id = ids_cut[ele]

        point_c, scale = point_c_and_scale(element_id, base)
        q_points_v = quad_points_volume(ORDER + 2)
        q_points_s, normal_vectors_s, weight_s = quad_points_surface(8)

        N = len(q_points_v)
        A = np.zeros((K, N))
        for i, q_point_v in enumerate(q_points_v):
            functions_v = divergence_free_functions(q_point_v)
            normal_vectors_v = normal_ref(q_point_v, point_c, scale)
            A[:, i] = np.sum(functions_v * normal_vectors_v, axis=1)

        b = np.zeros(K)
        for i, q_points_one_surface in enumerate(q_points_s):
            normal_vector_s = normal_vectors_s[i]
            for q_point_s in q_points_one_surface:
                functions_s = divergence_free_functions(q_point_s)
                h_value = heaviside_ref(q_point_s, point_c, scale)
                for k in range(K):
                    b[k] += -h_value * np.sum(functions_s[k] * normal_vector_s) * weight_s

        U, s, Vh = np.linalg.svd(A, full_matrices=True)
        S = np.diag([np.power(1./sigma, 2) if sigma > 1e-12 else 0 for sigma in s])

        B = np.matmul(np.matmul(U, S), np.transpose(U))

        # print(np.linalg.matrix_rank(B))

        # quad_weights = np.dot(np.matmul(np.transpose(A), np.linalg.inv(np.matmul(A, np.transpose(A)))), b)
        quad_weights = np.dot(np.matmul(np.transpose(A), B), b)
        # print(b)
        # print(quad_weights)
        # print(np.max(quad_weights))
        # print(np.min(quad_weights))
        # print("\n")
        hmf_result += np.sum(quad_weights) * np.power(scale, 2)
        print("Progress {:.5f}%, contribution {:.5f}, current hmf_result is {:.5f}, and gt is {:.5f}".format((ele + 1)/len(ids_cut)*100,
                                                                                                          np.sum(quad_weights), hmf_result, ground_truth)) 
        quad_points_to_save.append(to_physical(q_points_v, point_c, scale))
        weights_to_save.append(quad_weights * np.power(scale, 2))

    case_no = 2 if surface == 'sphere' else 3
    np.savetxt('data/dat/surface_integral/case_{}_quads.dat'.format(case_no), np.asarray(quad_points_to_save).reshape(-1, DIM))
    np.savetxt('data/dat/surface_integral/case_{}_weights.dat'.format(case_no), np.asarray(weights_to_save).reshape(-1))

    print("Error is {:.5f}".format(hmf_result - ground_truth))



################################################################################################################################
# Shifted boundary integration scheme
################################################################################################################################


def neighbors(element_id, base, h):
    id_xyz = to_id_xyz(element_id, base)
    faces = []
    min_id = 0
    max_id = base - 1
    for d in range(DIM):
        for r in range(NUM_DIRECTIONS):
            tmp = np.zeros(DIM)
            for i in range(DIM):
                tmp[i] = id_xyz[i]
            tmp[d] = id_xyz[d] + (2 * r - 1)
            if tmp[d] >= min_id and tmp[d] <= max_id:
                id_x, id_y, id_z = tmp
                vertices = get_vertices(id_x, id_y, id_z, h)
                cut_flag, negative_flag, positive_flag = is_cut(vertices)
                if not cut_flag and negative_flag:
                    faces.append([element_id, d*NUM_DIRECTIONS + r])
    return faces


def triangle_area(a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

def sbm_map_newton(point, function_value, function_gradient):
    tol = 1e-8
    res = 1.
    relax_param = 1.

    phi = function_value(point)
    grad_phi = function_gradient(point)
    target_point = np.array(point)

    step = 0
    while res > tol:
      delta1 = -phi * grad_phi / np.dot(grad_phi, grad_phi)
      delta2 = (point - target_point) - np.dot(point - target_point, grad_phi) / np.dot(grad_phi, grad_phi) * grad_phi
      target_point = target_point + relax_param * (delta1 + delta2)
      phi = function_value(target_point)
      grad_phi = function_gradient(target_point)
      res = np.absolute(phi) + np.linalg.norm(np.cross(grad_phi, (point - target_point)))
      step += 1

    # print(step)
    return target_point


def estimate_weights(shifted_q_point, d, step):
    num_boundary_points = 4 if DIM == 3 else 2
    boundary_points = np.zeros((num_boundary_points, DIM))
    for r in range(NUM_DIRECTIONS):
        for s in range(NUM_DIRECTIONS):
            boundary_points[r*NUM_DIRECTIONS + s, d] = shifted_q_point[d]
            boundary_points[r*NUM_DIRECTIONS + s, (d + 1) % DIM] = shifted_q_point[(d + 1) % DIM] + step / 2. * (2 * r - 1)
            boundary_points[r*NUM_DIRECTIONS + s, (d + 2) % DIM] = shifted_q_point[(d + 2) % DIM] + step / 2. * (2 * s - 1)

    mapped_boundary_points = np.zeros((num_boundary_points, DIM))
    for i, b_point in enumerate(boundary_points):
        mapped_boundary_points[i] = sbm_map_newton(b_point, level_set, grad_level_set)  

    mapped_q_point = sbm_map_newton(shifted_q_point, level_set, grad_level_set)

    weight = triangle_area(mapped_boundary_points[0], mapped_boundary_points[1], mapped_q_point) + \
             triangle_area(mapped_boundary_points[0], mapped_boundary_points[2], mapped_q_point) + \
             triangle_area(mapped_boundary_points[3], mapped_boundary_points[2], mapped_q_point) + \
             triangle_area(mapped_boundary_points[3], mapped_boundary_points[1], mapped_q_point)

    return mapped_q_point, weight


def process_face(face, base, h, quad_level):
    step = h / quad_level
    mapped_quad_points = []
    weights = []
    element_id, face_number = face
    id_xyz = to_id_xyz(element_id, base)
    d = face_number // NUM_DIRECTIONS
    r = face_number % NUM_DIRECTIONS
    shifted_quad_points = np.zeros((np.power(quad_level, 2), DIM))
    for i in range(quad_level):
        for j in range(quad_level):
            shifted_quad_points[i * quad_level + j, d] = -DOMAIN_SIZE + (id_xyz[d] + r) * h
            shifted_quad_points[i * quad_level + j, (d + 1) % DIM] = -DOMAIN_SIZE + id_xyz[(d + 1) % DIM]* h + step / 2. + i * step
            shifted_quad_points[i * quad_level + j, (d + 2) % DIM] = -DOMAIN_SIZE + id_xyz[(d + 2) % DIM]* h + step / 2. + j * step

    for shifted_q_point in shifted_quad_points:
        mapped_quad_point, weight = estimate_weights(shifted_q_point, d, step)
        mapped_quad_points.append(mapped_quad_point)
        weights.append(weight)

    return mapped_quad_points, weights


def compute_qw(quad_levels=[3], mesh_index=2, name='sbi_tests'):
    data = np.load('data/numpy/sbi/{}_cut_element_ids.npz'.format(surface), allow_pickle=True)
    total_ids = data['ids']
    total_refinement_levels = data['refinement_level']

    ids_cut = total_ids[mesh_index]
    refinement_level = total_refinement_levels[mesh_index]
    base = np.power(DIVISION, refinement_level)
    h = 2 * DOMAIN_SIZE / base
    print("\nrefinement_level is {} with h being {}, number of elements cut is {}".format(refinement_level, h, len(ids_cut)))

    faces = []
    for ele in range(0, len(ids_cut)):
        element_id = ids_cut[ele]
        faces += neighbors(element_id, base, h)

    for quad_level in quad_levels:
        mapped_quad_points = []
        weights = []
        ground_truth = 4 * np.pi

        for i, f in enumerate(faces):
            mapped_quad_points_f, weights_f = process_face(faces[i], base, h, quad_level)
            mapped_quad_points += mapped_quad_points_f
            weights += weights_f
            if i % 100 == 0:
                print("Progress {:.5f}%, weights {:.5f}, and gt is {:.5f}".format((i + 1)/len(faces)*100, np.sum(np.array(weights)), ground_truth))

        print(np.sum(np.array(weights)))

        case_no = 2 if surface == 'sphere' else 3
        np.savetxt('data/dat/surface_integral/sbi_case_{}_quads.dat'.format(case_no), np.asarray(mapped_quad_points).reshape(-1, DIM))
        np.savetxt('data/dat/surface_integral/sbi_case_{}_weights.dat'.format(case_no), np.asarray(weights).reshape(-1))

        # np.savetxt('data/dat/{}/sbi_case_{}_mesh_index_{}_quad_level_{}_quads.dat'.format(name, 
        #     case_no, mesh_index, quad_level), np.asarray(mapped_quad_points).reshape(-1, DIM))
        # np.savetxt('data/dat/{}/sbi_case_{}_mesh_index_{}_quad_level_{}_weights.dat'.format(name, 
        #     case_no, mesh_index, quad_level), np.asarray(weights).reshape(-1))


def test_function_0(points):
    return 1

def test_function_1(points):
    return 4 - 3 * points[:, 0]**2 + 2 * points[:, 1]**2 - points[:, 2]**2

def convergence_tests(test_function_number=0):
    name_tests = 'sbi_tests'
    name_convergence = 'sbi_convergence'
    case_no = 2 if surface == 'sphere' else 3
    quad_levels = np.arange(1, 4, 1)
    mesh_indices =  np.arange(0, 3, 1)
    test_function = test_function_0 if test_function_number == 0 else test_function_1
    ground_truth = 4 * np.pi if test_function_number == 0 else 40. / 3. * np.pi

    cache = False
    if cache:
        for mesh_index in mesh_indices:
            compute_qw(quad_levels, mesh_index, name_tests)

    mesh = []
    for mesh_index in mesh_indices:
        data = np.load('data/numpy/sbi/{}_cut_element_ids.npz'.format(surface), allow_pickle=True)
        total_ids = data['ids']
        total_refinement_levels = data['refinement_level']
        ids_cut = total_ids[mesh_index]
        refinement_level = total_refinement_levels[mesh_index]
        base = np.power(DIVISION, refinement_level)
        h = 2 * DOMAIN_SIZE / base * np.sqrt(3)
        mesh.append(h)

    errors = []
    for i, quad_level in enumerate(quad_levels):
        errors.append([])
        for j, mesh_index in enumerate(mesh_indices):
            mapped_quad_points = np.loadtxt('data/dat/{}/sbi_case_{}_mesh_index_{}_quad_level_{}_quads.dat'.format(name_tests, 
                case_no, mesh_index, quad_level))
            weights = np.loadtxt('data/dat/{}/sbi_case_{}_mesh_index_{}_quad_level_{}_weights.dat'.format(name_tests, 
                case_no, mesh_index, quad_level))
            values = test_function(mapped_quad_points)
            integral = np.sum(weights * values)
            relative_error = np.absolute((integral - ground_truth) / ground_truth)
            print("num quad points {}, quad_level {}, mesh_index {}, integral {}, ground_truth {}, relative error {}".format(len(weights), 
                 quad_level, mesh_index, integral, ground_truth, relative_error))
            errors[i].append(relative_error)
        convergence_array = np.concatenate((np.asarray(mesh).reshape(-1, 1), np.asarray(errors[i]).reshape(-1, 1)), axis=1)
        np.savetxt('data/dat/{}/case_{}_quad_level_{}.dat'.format(name_convergence, test_function_number, quad_level), convergence_array)

    fig = plt.figure()
    ax = fig.gca()
    for i, quad_level in enumerate(quad_levels):
        ax.plot(mesh, errors[i], linestyle='--', marker='o', label='# quad points per face {}x{}={}'.format(i + 1, i + 1, (i + 1)*(i + 1)))
 
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper left', prop={'size': 12})
    ax.tick_params(labelsize=14)
    ax.set_xlabel('mesh size', fontsize=14)
    ax.set_ylabel('relative error', fontsize=14)
    # fig.savefig(args.root_path + '/images/linear/L.png', bbox_inches='tight')

    print(np.log(errors[0][0]/errors[0][1]) / np.log(mesh[0]/mesh[1]))


def sbi_convergence_plot_single(test_function_number=0, size=0.65):
    label_name = '$\\\\mathcal{{E}}_{{\\\\rm{{rel}}}}$'
    gp.c('set terminal epslatex')
    gp.c('set output "data/latex/sbi/test_function_{}.tex"'.format(test_function_number))

    gp.c('set size {}, {}'.format(size, size))
    # gp.c('set terminal qt ' + str(pore_number))

    # gp.c('set title "Convergence tests" font ",14"')
    gp.c('set xlabel "$h$"')
    gp.c('set ylabel "{}"'.format(label_name))
    # gp.c('set xtics font ",12"')
    # gp.c('set ytics font ",12"')
    gp.c('set style data linespoints')
    # gp.c('set key top left')
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

    quad_file_1 = 'data/dat/sbi_convergence/case_{}_quad_level_{}.dat'.format(test_function_number, 1)
    quad_file_2 = 'data/dat/sbi_convergence/case_{}_quad_level_{}.dat'.format(test_function_number, 2)
    quad_file_3 = 'data/dat/sbi_convergence/case_{}_quad_level_{}.dat'.format(test_function_number, 3)
   
    quad_array_1 = np.loadtxt(quad_file_1)
    quad_array_2 = np.loadtxt(quad_file_2)
    quad_array_3 = np.loadtxt(quad_file_3)

    h_ratio = np.log((quad_array_1[0, 0] / quad_array_1[-1, 0]))

    quad_ratio_1 = np.log((quad_array_1[0, 1] / quad_array_1[-1, 1]))
    quad_ratio_2 = np.log((quad_array_2[0, 1] / quad_array_2[-1, 1]))
    quad_ratio_3 = np.log((quad_array_3[0, 1] / quad_array_3[-1, 1]))

    quad_ratio_1 = "Q 1x1 (OC={:.3f})".format(quad_ratio_1 / h_ratio)
    quad_ratio_2 = "Q 2x2 (OC={:.3f})".format(quad_ratio_2 / h_ratio)
    quad_ratio_3 = "Q 3x3 (OC={:.3f})".format(quad_ratio_3 / h_ratio)

    plotting_command = 'plot "{}" u 1:2 title "{}" lc "red" pt 5 ps 2 lt 1 lw 4 , \
            "{}" u 1:2 title "{}" lc "green" pt 9 ps 2 lt 1 lw 4, \
            "{}" u 1:2 title "{}" lc "blue" pt 7 ps 2 lt 1 lw 4'.format(quad_file_1,
                                                                        quad_ratio_1,
                                                                        quad_file_2,
                                                                        quad_ratio_2,
                                                                        quad_file_3,
                                                                        quad_ratio_3)            

    gp.c(plotting_command)
    gp.c('set out')


def generate_sbi_convergence():
    sbi_convergence_plot_single(test_function_number=0)
    sbi_convergence_plot_single(test_function_number=1)


if __name__ == '__main__':
    # generate_cut_elements()
    # brute_force(np.power(DIVISION, 6))
    # main()
    # print(np.sum(np.loadtxt('data/dat/surface_integral/case_{}_weights.dat'.format(2))))

    # compute_qw()
    # convergence_tests()
    generate_sbi_convergence()
    plt.show()