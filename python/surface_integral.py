import sys
import numpy as np
import matplotlib.pyplot as plt
import copy

np.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
np.set_printoptions(precision=5)
np.random.seed(1)

DIM = 3
DOMAIN_SIZE = 2
REFERENCE_SIZE = 1
DIVISION = 2
surface = 'sphere'
ORDER = 1

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
    num_direction = 2
    step = 2 * REFERENCE_SIZE / density
    weight = np.power(2 * REFERENCE_SIZE, 2) / np.power(density, 2)
    points_collection = []
    normal_vectors = []
    for d in range(DIM):
        for r in range(num_direction):
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
                if is_cut(vertices):
                    ids_cut.append(to_id(id_x, id_y, id_z, base))

    return ids_cut


def is_cut(vertices):
    negative_flag = False
    positive_flag = False
    for vertice in vertices:
        value = level_set(vertice)
        if value > 0:
            positive_flag = True
        elif value < 0:
            negative_flag = True
    return negative_flag and positive_flag

def point_c_and_scale(element_id, base):
    id_x, id_y, id_z = to_id_xyz(element_id, base)
    h = 2 * DOMAIN_SIZE / base
    point_c =  np.array([-DOMAIN_SIZE + (id_x + 1./2.) * h, -DOMAIN_SIZE + (id_y + 1./2.) * h, -DOMAIN_SIZE + (id_z + 1./2.) * h ])
    scale = h / (2 * REFERENCE_SIZE)
    return point_c, scale


def generate_cut_elements():
    start_refinement_level = 6
    end_refinement_level = 9
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
                if is_cut(get_vertices(sub_id_x, sub_id_y, sub_id_z, h / DIVISION)):
                    ids_cut_new.append(sub_id)
        ids_cut = ids_cut_new
        total_ids.append(ids_cut)
        total_refinement_levels.append(refinement_level + 1)
        print("refinement_level {}, length of inds {}".format(refinement_level + 1, len(ids_cut)))

    np.savez('data/numpy/{}_cut_element_ids.npz'.format(surface), ids=total_ids, refinement_level=total_refinement_levels, allow_pickle=True)
    return total_ids, total_refinement_levels


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
        S = np.diag([np.power(1./sigma, 2) if sigma > 1e-1 else 0 for sigma in s])

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

    np.savetxt('data/dat/surface_integral/{}_quads.dat'.format(surface), np.asarray(quad_points_to_save).reshape(-1, DIM))
    np.savetxt('data/dat/surface_integral/{}_weights.dat'.format(surface), np.asarray(weights_to_save).reshape(-1))

    print("Error is {:.5f}".format(hmf_result - ground_truth))

if __name__ == '__main__':
    # generate_cut_elements()
    # brute_force(np.power(DIVISION, 6))
    # main()

    print(np.sum(np.loadtxt('data/dat/surface_integral/{}_weights.dat'.format(surface))))