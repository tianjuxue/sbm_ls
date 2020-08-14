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
  vel[0] = 0.0;
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
             hp::DoFHandler<dim> &dof_handler_all,
             Vector<double> &solution_all)
{
  for (int i = 0; i < length; ++i)
  {
    std::cout << std::endl << "  Point is " << points[i] << std::endl;
    Point<dim> target_point;
    target_point = points[i];
    double phi;
    Tensor<1, dim> grad_phi;
    double tol = 1e-10;
    double res = 1.;
    double relax_param = 1.;
    Tensor<1, dim> delta1, delta2;
    int step = 0;
    while (res > tol)
    {
      phi = VectorTools::point_value(dof_handler_all, solution_all, target_point);
      grad_phi = VectorTools::point_gradient(dof_handler_all, solution_all, target_point);
      res = abs(phi) + cross_product_norm(grad_phi, (points[i] - target_point));
      delta1 = -phi * grad_phi / (grad_phi * grad_phi);
      delta2 = (points[i] - target_point) - ( (points[i] - target_point) * grad_phi / (grad_phi * grad_phi) ) * grad_phi;
      target_point = target_point + relax_param * (delta1 + delta2);
      step++;
      std::cout << "  res is " << res << std::endl;
      std::cout << "  res1 is " << abs(phi) << std::endl;
      std::cout << "  res2 is " << cross_product_norm(grad_phi, (points[i] - target_point)) << std::endl;
      std::cout << "  The point found: " << target_point << std::endl;
      std::cout << "  It should be " << 0.5 * points[i] / points[i].norm() << std::endl;
    }
    std::cout << "  Total step is " << step << std::endl;
    target_points[i] = target_point;
    normal_vectors[i] = -grad_phi / grad_phi.norm();
    distance_vectors[i] = target_point - points[i];
    std::cout << "  The target point found: " << target_point << std::endl;
    std::cout << "  It should be " << 0.5 * points[i] / points[i].norm() << std::endl;
  }
  std::cout << "  End of this call to sbm_map" << std::endl;
}


template <int dim>
void sbm_map_manual(std::vector<Point<dim>> &target_points,
                    std::vector<Tensor<1, dim>> &normal_vectors,
                    std::vector<Tensor<1, dim>> &distance_vectors,
                    const std::vector<Point<dim>> &points,
                    int length,
                    hp::DoFHandler<dim> &dof_handler_all,
                    Vector<double> &solution_all)
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
bool is_inside(hp::DoFHandler<dim> &dof_handler_all, Vector<double> &solution_all, Point<dim> &point)
{
  return VectorTools::point_value(dof_handler_all, solution_all, point) > 0 ? true : false;
}

template <int dim>
bool is_inside_manual(hp::DoFHandler<dim> &dof_handler_all, Vector<double> &solution_all, Point<dim> &point)
{
  return point.norm() < 0.5 ? true : false;
}


template <int dim>
void set_support_points(hp::DoFHandler<dim> &dof_handler_all, std::vector<Point<dim>> &support_points)
{
  hp::MappingCollection<dim> mapping_collection;
  mapping_collection.push_back(MappingQ1<dim>());
  DoFTools::map_dofs_to_support_points(mapping_collection, dof_handler_all, support_points);
}


template <int dim>
void union_distance_fields(hp::DoFHandler<dim> &dof_handler_all, Vector<double> &solution_all,
                           hp::DoFHandler<dim> &dof_handler_in, Vector<double> &solution_in,
                           hp::DoFHandler<dim> &dof_handler_out, Vector<double> &solution_out)
{

  std::vector<Point<dim>> support_points(dof_handler_all.n_dofs());
  set_support_points(dof_handler_all, support_points);

  for (unsigned int i = 0; i < dof_handler_all.n_dofs(); i++)
  {
    Vector<double> old_solution_all  = solution_all;
    if (is_inside(dof_handler_all, old_solution_all, support_points[i]))
      solution_all(i) = VectorTools::point_value(dof_handler_in, solution_in, support_points[i]);
    else
      solution_all(i) = VectorTools::point_value(dof_handler_out, solution_out, support_points[i]);

  }
}


template <int dim>
void initialize_distance_field(hp::DoFHandler<dim> &dof_handler_all, Vector<double> &solution_all, double radius)
{
  // double radius = 0.5;
  std::vector<Point<dim>> support_points(dof_handler_all.n_dofs());
  set_support_points(dof_handler_all, support_points);
  for (unsigned int i = 0; i < dof_handler_all.n_dofs(); i++)
  {
    solution_all(i) = radius - support_points[i].norm();
  }
}


#endif