import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
from .surface_integral import to_id_xyz, get_vertices, is_cut, level_set, grad_level_set


from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt


np.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
np.set_printoptions(precision=5)
np.random.seed(1)

DIM = 3
DOMAIN_SIZE = 2
DIVISION = 2
surface = 'sphere'
NUM_DIRECTIONS = 2


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


def compute_qw(quad_levels, mesh_index, name):
    data = np.load('data/numpy/{}_cut_element_ids.npz'.format(surface), allow_pickle=True)
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

        # np.savetxt('data/dat/surface_integral/sbi_case_{}_quads.dat'.format(case_no), np.asarray(mapped_quad_points).reshape(-1, DIM))
        # np.savetxt('data/dat/surface_integral/sbi_case_{}_weights.dat'.format(case_no), np.asarray(weights).reshape(-1))

        np.savetxt('data/dat/{}/sbi_case_{}_mesh_index_{}_quad_level_{}_quads.dat'.format(name, 
            case_no, mesh_index, quad_level), np.asarray(mapped_quad_points).reshape(-1, DIM))
        np.savetxt('data/dat/{}/sbi_case_{}_mesh_index_{}_quad_level_{}_weights.dat'.format(name, 
            case_no, mesh_index, quad_level), np.asarray(weights).reshape(-1))


def test_function(points):
    return 4 - 3 * points[:, 0]**2 + 2 * points[:, 1]**2 - points[:, 2]**2


def cache_compute_qw(name):
    case_no = 2 if surface == 'sphere' else 3
    quad_levels = np.arange(1, 4, 1)
    mesh_indices =  np.arange(0, 3, 1)

    mesh = []
    cache = False
    if cache:
        for mesh_index in mesh_indices:
            compute_qw(quad_levels, mesh_index, name)

    ground_truth = 40. / 3. * np.pi
    errors = []
    for i, quad_level in enumerate(quad_levels):
        errors.append([])
        for j, mesh_index in enumerate(mesh_indices):
            mapped_quad_points = np.loadtxt('data/dat/{}/sbi_case_{}_mesh_index_{}_quad_level_{}_quads.dat'.format(name, 
                case_no, mesh_index, quad_level))
            weights = np.loadtxt('data/dat/{}/sbi_case_{}_mesh_index_{}_quad_level_{}_weights.dat'.format(name, 
                case_no, mesh_index, quad_level))
            values = test_function(mapped_quad_points)
            integral = np.sum(weights * values)
            relative_error = np.absolute((integral - ground_truth) / ground_truth)
            print("num quad points {}, quad_level {}, mesh_index {}, integral {}, ground_truth {}, relative error {}".format(len(weights), 
                 quad_level, mesh_index, integral, ground_truth, relative_error))
            errors[i].append(relative_error)

    for mesh_index in mesh_indices:

        data = np.load('data/numpy/{}_cut_element_ids.npz'.format(surface), allow_pickle=True)
        total_ids = data['ids']
        total_refinement_levels = data['refinement_level']
        ids_cut = total_ids[mesh_index]
        refinement_level = total_refinement_levels[mesh_index]
        base = np.power(DIVISION, refinement_level)
        h = 2 * DOMAIN_SIZE / base
        mesh.append(h)

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

    print( np.log(errors[0][0]/errors[0][1]) / np.log(mesh[0]/mesh[1]) )


if __name__ == '__main__':
    # compute_qw(quad_level=3, mesh_index=2, name='surface_integral')
    cache_compute_qw(name='sbi_tests')
    plt.show()
