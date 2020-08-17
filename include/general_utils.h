#ifndef GENERAL_UTILS_H
#define GENERAL_UTILS_H

using namespace dealii;


template <int dim>
class AdvectionVelocity
{
public:
  Tensor<1, dim> get_velocity(Point<dim> &point);
};


template <int dim>
Tensor<1, dim> AdvectionVelocity<dim>::get_velocity(Point<dim> &point)
{
  Tensor<1, dim> vel;
  vel[0] = 0.0025;
  vel[1] = 0;
  return vel;
}


template <int dim>
double cross_product_norm(const Tensor<1, dim> &t1, const Tensor<1, dim> &t2)
{
  return abs(t1[0] * t2[1] - t1[1] * t2[0]);
}


/* Construct map and return distance vector.
   Based on https://doi.org/10.1137/S106482750037617X */
template <int dim>
void sbm_map(std::vector<Point<dim>> &target_points,
             std::vector<Tensor<1, dim>> &normal_vectors,
             std::vector<Tensor<1, dim>> &distance_vectors,
             const std::vector<Point<dim>> &points,
             int length,
             hp::DoFHandler<dim> &dof_handler,
             Vector<double> &solution)
{
  for (int i = 0; i < length; ++i)
  {
    // std::cout << std::endl << "  Point is " << points[i] << std::endl;
    Point<dim> target_point;
    target_point = points[i];
    double phi;
    Tensor<1, dim> grad_phi;
    double tol = 1e-4;
    double res = 1.;
    double relax_param = 1.;
    Tensor<1, dim> delta1, delta2;
    int step = 0;
    int max_step = 100;
    while (res > tol && step < max_step)
    {
      phi = VectorTools::point_value(dof_handler, solution, target_point);
      grad_phi = VectorTools::point_gradient(dof_handler, solution, target_point);

      res = abs(phi) + cross_product_norm(grad_phi, (points[i] - target_point));
      // std::cout << "  res is " << res << std::endl;
      // std::cout << "  res1 is " << abs(phi) << std::endl;
      // std::cout << "  res2 is " << cross_product_norm(grad_phi, (points[i] - target_point)) << std::endl;

      delta1 = -phi * grad_phi / (grad_phi * grad_phi);
      delta2 = (points[i] - target_point) - ( (points[i] - target_point) * grad_phi / (grad_phi * grad_phi) ) * grad_phi;
      target_point = target_point + relax_param * (delta1 + delta2);

      step++;

      // std::cout << "  The point found: " << target_point << std::endl;
      // std::cout << "  It should be " << 0.5 * points[i] / points[i].norm() << std::endl;
    }

    if (res > tol)
    {
      while (res > tol)
      {
        relax_param = 0.1;
        tol = 1e-3;
        phi = VectorTools::point_value(dof_handler, solution, target_point);
        grad_phi = VectorTools::point_gradient(dof_handler, solution, target_point);
        res = abs(phi) + cross_product_norm(grad_phi, (points[i] - target_point));
        delta1 = -phi * grad_phi / (grad_phi * grad_phi);
        delta2 = (points[i] - target_point) - ( (points[i] - target_point) * grad_phi / (grad_phi * grad_phi) ) * grad_phi;
        target_point = target_point + relax_param * (delta1 + delta2);
        step++;
      }
      std::cout << "  Bad point converge at step " << step << std::endl;
    }

    // std::cout << "  Total step is " << step << std::endl;
    target_points[i] = target_point;
    normal_vectors[i] = -grad_phi / grad_phi.norm();
    distance_vectors[i] = target_point - points[i];
    // std::cout << "  The target point found: " << target_point << std::endl;
    // std::cout << "  It should be " << 0.5 * points[i] / points[i].norm() << std::endl;
  }
  std::cout << "  End of this call to sbm_map, target points[0]: " << target_points[0] << std::endl << std::endl;
}


template <int dim>
void sbm_map_manual(std::vector<Point<dim>> &target_points,
                    std::vector<Tensor<1, dim>> &normal_vectors,
                    std::vector<Tensor<1, dim>> &distance_vectors,
                    const std::vector<Point<dim>> &points,
                    int length,
                    hp::DoFHandler<dim> &dof_handler,
                    Vector<double> &solution)
{
  for (int i = 0; i < length; ++i)
  {
    target_points[i] = 0.5 * points[i] / points[i].norm();
    normal_vectors[i] = points[i] / points[i].norm();
    distance_vectors[i] = target_points[i] - points[i];
  }
}


template <int dim>
void compute_boundary_values(AdvectionVelocity<dim> &velocity,
                             std::vector<Point<dim>> &target_points,
                             std::vector<Tensor<1, dim>> &normal_vectors,
                             std::vector<double> &boundary_values,
                             int length)
{
  for (int i = 0; i < length; ++i)
  {
    boundary_values[i] = velocity.get_velocity(target_points[i]) * normal_vectors[i];
  }
}


// template<int dim>
void vec2num_values(std::vector<Vector<double> > &vec,
                    std::vector<double >  &num, int length)
{

  int dim = 2;
  int temp = 2 * dim;
  for (int i = 0; i < length; ++i)
  {
    num[i] = vec[i][temp];
  }
}

// From vector<Tensor<1, dim>>, extract Tensor<2, dim>
template <int dim>
void vec2num_grads(std::vector< std::vector< Tensor<1, dim> >> &vec,
                   std::vector<Tensor<1, dim>> & ten, int length)
{
  int temp = 2 * dim;
  for (int i = 0; i < length; ++i)
  {
    for (int j = 0; j < dim; ++j)
    {
      ten[i][j] = vec[i][temp][j];
    }
  }
}


template <int dim>
bool is_inside(hp::DoFHandler<dim> &dof_handler, Vector<double> &solution, const Point<dim> &point)
{
  return VectorTools::point_value(dof_handler, solution, point) > 0 ? true : false;
}

template <int dim>
bool is_inside_manual(hp::DoFHandler<dim> &dof_handler, Vector<double> &solution, const Point<dim> &point)
{
  return point.norm() < 0.5 ? true : false;
}


template <int dim>
void set_support_points(hp::DoFHandler<dim> &dof_handler, std::vector<Point<dim>> &support_points)
{
  hp::MappingCollection<dim> mapping_collection;
  mapping_collection.push_back(MappingQ1<dim>());
  DoFTools::map_dofs_to_support_points(mapping_collection, dof_handler, support_points);
}



template <int dim>
void initialize_distance_field(hp::DoFHandler<dim> &dof_handler, Vector<double> &solution, double radius)
{
  std::vector<Point<dim>> support_points(dof_handler.n_dofs());
  set_support_points(dof_handler, support_points);
  for (unsigned int i = 0; i < dof_handler.n_dofs(); i++)
  {
    solution(i) = radius - support_points[i].norm();
  }
}


double min_multiple(std::vector<double> &v)
{
  double temp = v[0];
  for (unsigned int i = 0; i < v.size(); i++) {
    if (temp > v[i])
    {
      temp = v[i];
    }
  }
  return temp;
}


template <int dim>
double signed_distance_square(Point<dim> &point, double side_length)
{
  double half_side = side_length / 2.;
  double upper_left_corner_dist  = sqrt(pow(point[0] + half_side, 2) + pow(point[1] - half_side, 2));
  double upper_right_corner_dist = sqrt(pow(point[0] - half_side, 2) + pow(point[1] - half_side, 2));
  double lower_left_corner_dist  = sqrt(pow(point[0] + half_side, 2) + pow(point[1] + half_side, 2));
  double lower_right_corner_dist = sqrt(pow(point[0] - half_side, 2) + pow(point[1] + half_side, 2));
  double left_side_dist  = abs(point[0] + half_side);
  double right_side_dist = abs(point[0] - half_side);
  double upper_side_dist = abs(point[1] - half_side);
  double lower_side_dist = abs(point[1] + half_side);
  std::vector<double> v_corners{upper_left_corner_dist, upper_right_corner_dist, lower_left_corner_dist, lower_right_corner_dist};
  std::vector<double> v_sides{left_side_dist, right_side_dist, upper_side_dist, lower_side_dist};

  double dist;
  if ((point[0] > half_side  & point[1] > half_side)  ||
      (point[0] > half_side  & point[1] < -half_side) ||
      (point[0] < -half_side & point[1] > half_side)  ||
      (point[0] < -half_side & point[1] < -half_side))
    dist = min_multiple(v_corners);
  else if (point[0] > half_side)
    dist = right_side_dist;
  else if (point[0] < -half_side)
    dist = left_side_dist;
  else if (point[1] > half_side)
    dist = upper_side_dist;
  else if (point[1] < -half_side)
    dist = lower_side_dist;
  else
    dist = min_multiple(v_sides);

  double sign = point[0] < half_side & point[0] > -half_side & point[1] < half_side & point[1] > -half_side ? 1. : -1.;

  return sign * dist;
}


template <int dim>
void initialize_distance_field_square(hp::DoFHandler<dim> &dof_handler, Vector<double> &solution, double side_length)
{
  std::vector<Point<dim>> support_points(dof_handler.n_dofs());
  set_support_points(dof_handler, support_points);
  for (unsigned int i = 0; i < dof_handler.n_dofs(); i++)
  {
    solution(i) = signed_distance_square(support_points[i], side_length);
  }
}


#endif