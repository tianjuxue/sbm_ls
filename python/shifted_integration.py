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
QUAD_LEVEL = 3
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

    # print(mapped_boundary_points)
    # print(mapped_q_point)
    # print(triangle_area(mapped_boundary_points[0], mapped_boundary_points[1], mapped_q_point))
   
    return mapped_q_point, weight


def process_face(face, base, h, ax):
    step = h / QUAD_LEVEL
    mapped_quad_points = []
    weights = []
    element_id, face_number = face
    id_xyz = to_id_xyz(element_id, base)
    d = face_number // NUM_DIRECTIONS
    r = face_number % NUM_DIRECTIONS
    shifted_quad_points = np.zeros((np.power(QUAD_LEVEL, 2), DIM))
    for i in range(QUAD_LEVEL):
        for j in range(QUAD_LEVEL):
            shifted_quad_points[i * QUAD_LEVEL + j, d] = -DOMAIN_SIZE + (id_xyz[d] + r) * h
            shifted_quad_points[i * QUAD_LEVEL + j, (d + 1) % DIM] = -DOMAIN_SIZE + id_xyz[(d + 1) % DIM]* h + step / 2. + i * step
            shifted_quad_points[i * QUAD_LEVEL + j, (d + 2) % DIM] = -DOMAIN_SIZE + id_xyz[(d + 2) % DIM]* h + step / 2. + j * step

    # colors = ['blue', 'red', 'yellow']
    # print(face)
    # print(shifted_quad_points)
    # print("\n")
    # ax.scatter(shifted_quad_points[:, 0], shifted_quad_points[:, 1], shifted_quad_points[:, 2], color=colors[d], s=0.1)

    for shifted_q_point in shifted_quad_points:
        mapped_quad_point, weight = estimate_weights(shifted_q_point, d, step)
        mapped_quad_points.append(mapped_quad_point)
        weights.append(weight)

    return mapped_quad_points, weights


def main():
    data = np.load('data/numpy/{}_cut_element_ids.npz'.format(surface), allow_pickle=True)
    total_ids = data['ids']
    total_refinement_levels = data['refinement_level']
    index = 1
    ids_cut = total_ids[index]
    refinement_level = total_refinement_levels[index]
    base = np.power(DIVISION, refinement_level)
    h = 2 * DOMAIN_SIZE / base
    print("refinement_level is {} with h being {}, number of elements cut is {}".format(refinement_level, h, len(ids_cut)))

    weights = []
    quad_points = []
    shifted_quad_points = []

    faces = []
    for ele in range(0, len(ids_cut)):
        element_id = ids_cut[ele]
        faces += neighbors(element_id, base, h)

    mapped_quad_points = []
    weights = []
    ground_truth = 4 * np.pi

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, f in enumerate(faces):
        mapped_quad_points_f, weights_f = process_face(faces[i], base, h, ax)
        mapped_quad_points += mapped_quad_points_f
        weights += weights_f
        print("Progress {:.5f}%, weights {:.5f}, and gt is {:.5f}".format((i + 1)/len(faces)*100, np.sum(np.array(weights)), ground_truth))

    print(np.sum(np.array(weights)))

    case_no = 2 if surface == 'sphere' else 3
    np.savetxt('data/dat/surface_integral/sbi_case_{}_quads.dat'.format(case_no), np.asarray(mapped_quad_points).reshape(-1, DIM))
    np.savetxt('data/dat/surface_integral/sbi_case_{}_weights.dat'.format(case_no), np.asarray(weights).reshape(-1))


if __name__ == '__main__':
    # generate_cut_elements()
    # brute_force(np.power(DIVISION, 6))
    main()
