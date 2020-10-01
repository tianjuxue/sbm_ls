#ifndef GENERAL_UTILS_H
#define GENERAL_UTILS_H

using namespace dealii;


template <int dim>
class AdvectionVelocity
{
public:
  Tensor<1, dim> get_velocity(Point<dim> &point, double time);
};


template <int dim>
Tensor<1, dim> AdvectionVelocity<dim>::get_velocity(Point<dim> &point, double time)
{
  Tensor<1, dim> vel;
  vel[0] = 1.;
  // vel[0] = 0.;
  vel[1] = 0;

  double T = 2;
  vel[0] = -2 * sin(M_PI * point[0]) * sin(M_PI * point[0]) * cos(M_PI * point[1]) * sin(M_PI * point[1]) * cos(M_PI * time / T);
  vel[1] = 2 * cos(M_PI * point[0]) * sin(M_PI * point[0]) * sin(M_PI * point[1]) * sin(M_PI * point[1]) * cos(M_PI * time / T);

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
    double tol = 1e-5;
    double res = 1.;
    double relax_param = 1.;
    Tensor<1, dim> delta1, delta2;
    int step = 0;
    int max_step = 100;

    phi = VectorTools::point_value(dof_handler, solution, target_point);
    grad_phi = VectorTools::point_gradient(dof_handler, solution, target_point);

    while (res > tol && step < max_step)
    {
      delta1 = -phi * grad_phi / (grad_phi * grad_phi);
      delta2 = (points[i] - target_point) - ( (points[i] - target_point) * grad_phi / (grad_phi * grad_phi) ) * grad_phi;
      target_point = target_point + relax_param * (delta1 + delta2);

      // Bound the point, to change
      target_point[0] = target_point[0] > 1 ? 0.5 : target_point[0];
      target_point[0] = target_point[0] < 0 ? 0.5 : target_point[0];
      target_point[1] = target_point[1] > 1 ? 0.5 : target_point[1];
      target_point[1] = target_point[1] < 0 ? 0.5 : target_point[1];

      phi = VectorTools::point_value(dof_handler, solution, target_point);
      grad_phi = VectorTools::point_gradient(dof_handler, solution, target_point);
      res = abs(phi) + cross_product_norm(grad_phi, (points[i] - target_point));
      step++;

      // std::cout << "  res is " << res << std::endl;
      // std::cout << "  res1 is " << abs(phi) << std::endl;
      // std::cout << "  res2 is " << cross_product_norm(grad_phi, (points[i] - target_point)) << std::endl;
      // std::cout << "  The point found: " << target_point << std::endl;
    }

    if (res > tol)
    {
      tol = 1e-5;
      relax_param = 0.1;
      while (abs(phi) > tol)
      {
        delta1 = -phi * grad_phi / (grad_phi * grad_phi);
        // delta2 = (points[i] - target_point) - ( (points[i] - target_point) * grad_phi / (grad_phi * grad_phi) ) * grad_phi;
        // target_point = target_point + relax_param * (delta1 + delta2);
        target_point = target_point + relax_param * (delta1);

        phi = VectorTools::point_value(dof_handler, solution, target_point);
        grad_phi = VectorTools::point_gradient(dof_handler, solution, target_point);
        res = abs(phi);
        step++;
      }
      std::cout << "  End of bad point converge at step " << step << " mapped point " << target_point << " phi value " << phi << std::endl;

    }

    // std::cout << "  Total step is " << step << std::endl;
    target_points[i] = target_point;
    normal_vectors[i] = -grad_phi / grad_phi.norm();
    distance_vectors[i] = target_point - points[i];
    // std::cout << "  The target point found: " << target_point << std::endl;
    // std::cout << "  It should be " << 0.5 * points[i] / points[i].norm() << std::endl;

    std::cout << "  End of this call to sbm_map, surrogate points[i]: " << points[i]
              << "  phi value "  << VectorTools::point_value(dof_handler, solution, target_point)
              << "  mapped points " << target_point << std::endl;

  }

  std::cout << std::endl;

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
void lagrangian_shift(AdvectionVelocity<dim> &velocity,
                      std::vector<Point<dim>> &target_points,
                      std::vector<Tensor<1, dim>> &distance_vectors,
                      std::vector<double> &boundary_values,
                      double dt,
                      double time,
                      int length)
{
  for (int i = 0; i < length; ++i)
  {
    boundary_values[i] = 0;
    Tensor<1, dim> shift = velocity.get_velocity(target_points[i], time) * dt;
    target_points[i] += shift;
    distance_vectors[i] += shift;
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
void initialize_distance_field_circle(hp::DoFHandler<dim> &dof_handler, Vector<double> &solution, const Point<dim> &center, double radius)
{

  std::vector<Point<dim>> support_points(dof_handler.n_dofs());
  set_support_points(dof_handler, support_points);
  for (unsigned int i = 0; i < dof_handler.n_dofs(); i++)
  {
    Tensor<1, dim> rel_pos = support_points[i] - center;
    solution(i) = radius - rel_pos.norm();
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
double signed_distance_square(Point<dim> &point, const Point<dim> &center, double side_length)
{
  Tensor<1, dim> rel_pos = point - center;

  double half_side = side_length / 2.;
  double upper_left_corner_dist  = sqrt(pow(rel_pos[0] + half_side, 2) + pow(rel_pos[1] - half_side, 2));
  double upper_right_corner_dist = sqrt(pow(rel_pos[0] - half_side, 2) + pow(rel_pos[1] - half_side, 2));
  double lower_left_corner_dist  = sqrt(pow(rel_pos[0] + half_side, 2) + pow(rel_pos[1] + half_side, 2));
  double lower_right_corner_dist = sqrt(pow(rel_pos[0] - half_side, 2) + pow(rel_pos[1] + half_side, 2));
  double left_side_dist  = abs(rel_pos[0] + half_side);
  double right_side_dist = abs(rel_pos[0] - half_side);
  double upper_side_dist = abs(rel_pos[1] - half_side);
  double lower_side_dist = abs(rel_pos[1] + half_side);
  std::vector<double> v_corners{upper_left_corner_dist, upper_right_corner_dist, lower_left_corner_dist, lower_right_corner_dist};
  std::vector<double> v_sides{left_side_dist, right_side_dist, upper_side_dist, lower_side_dist};

  double dist;
  if ((rel_pos[0] > half_side  & rel_pos[1] > half_side)  ||
      (rel_pos[0] > half_side  & rel_pos[1] < -half_side) ||
      (rel_pos[0] < -half_side & rel_pos[1] > half_side)  ||
      (rel_pos[0] < -half_side & rel_pos[1] < -half_side))
    dist = min_multiple(v_corners);
  else if (rel_pos[0] > half_side)
    dist = right_side_dist;
  else if (rel_pos[0] < -half_side)
    dist = left_side_dist;
  else if (rel_pos[1] > half_side)
    dist = upper_side_dist;
  else if (rel_pos[1] < -half_side)
    dist = lower_side_dist;
  else
    dist = min_multiple(v_sides);

  double sign = rel_pos[0] < half_side & rel_pos[0] > -half_side & rel_pos[1] < half_side & rel_pos[1] > -half_side ? 1. : -1.;

  return sign * dist;
}


template <int dim>
void initialize_distance_field_square(hp::DoFHandler<dim> &dof_handler, Vector<double> &solution, const Point<dim> &center, double side_length)
{
  std::vector<Point<dim>> support_points(dof_handler.n_dofs());
  set_support_points(dof_handler, support_points);
  for (unsigned int i = 0; i < dof_handler.n_dofs(); i++)
  {
    solution(i) = signed_distance_square(support_points[i], center, side_length);
  }
}


template <int dim>
void initialize_distance_field_linear(hp::DoFHandler<dim> &dof_handler, Vector<double> &solution)
{
  std::vector<Point<dim>> support_points(dof_handler.n_dofs());
  set_support_points(dof_handler, support_points);
  for (unsigned int i = 0; i < dof_handler.n_dofs(); i++)
  {
    solution(i) = support_points[i][0] + support_points[i][1];
  }
}


template <int dim>
void initialize_distance_field_quadratic(hp::DoFHandler<dim> &dof_handler, Vector<double> &solution, const Point<dim> &center)
{
  std::vector<Point<dim>> support_points(dof_handler.n_dofs());
  set_support_points(dof_handler, support_points);
  for (unsigned int i = 0; i < dof_handler.n_dofs(); i++)
  {
    Tensor<1, dim> rel_pos = support_points[i] - center;
    solution(i) = 0.25 - rel_pos[0] * rel_pos[0] - rel_pos[1] * rel_pos[1];
  }
}


template <int dim>
void reinitialize_distance_field(hp::DoFHandler<dim> &dof_handler,
                                 Vector<double> &old_solution,
                                 Vector<double> &solution)
{
  unsigned int length = dof_handler.n_dofs();
  std::vector<Point<dim>> support_points(length);
  set_support_points(dof_handler, support_points);

  for (unsigned int i = 0; i < length; i++)
  {
    if (old_solution(i) > 0.01)
    {
      unsigned int unit = 1;
      std::vector<Point<dim>> target_points(unit);
      std::vector<Tensor<1, dim> > normal_vectors(unit);
      std::vector<Tensor<1, dim> > distance_vectors(unit);
      std::vector<Point<dim>> points;
      points.push_back(support_points[i]);

      sbm_map(target_points, normal_vectors, distance_vectors, points, unit, dof_handler, old_solution);
      solution(i) = normal_vectors[0] * distance_vectors[0];
    }
  }
}


template <int dim>
void reinitialize_distance_field_poisson(hp::DoFHandler<dim> &dof_handler,
    Vector<double> &old_solution,
    Vector<double> &solution)
{
  unsigned int length = dof_handler.n_dofs();
  std::vector<Point<dim>> support_points(length);
  set_support_points(dof_handler, support_points);

  for (unsigned int i = 0; i < length; i++)
  {
    if (old_solution(i) < 0.1)
    {
      solution(i) = old_solution(i);
    }
  }

}

#endif