#ifndef GENERAL_UTILS_H
#define GENERAL_UTILS_H

using namespace dealii;

template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  BoundaryValues()
    : Function<dim>()
  {}
  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override;
};


template <int dim>
double BoundaryValues<dim>::value(const Point<dim> & p,
                                  const unsigned int component) const
{
  return 1.;
}


template <int dim>
double pore_function_value(const Point<dim> &point, double c1, double c2)
{
  double x = point[0];
  double y = point[1];
  double theta = atan2(y, x);
  double r_square = pow(x, 2) + pow(y, 2);
  double value =  r_square - (1 + c1 * cos(4 * theta) + c2 * cos(8 * theta));
  return value;
}


template <int dim>
double star_function_value(const Point<dim> &point)
{
  double x = point[0];
  double y = point[1];
  double theta = atan2(y, x);
  double r_square = pow(x, 2) + pow(y, 2);
  double value =  r_square - (1 + 0.2 * sin(6 * theta));
  return value;
}


template <int dim>
double sphere_function_value(const Point<dim> &point)
{
  double x = point[0];
  double y = point[1];
  double z = point[2];
  double value  = pow(x, 2) + pow(y, 2) + pow(z, 2) - 1;
  return value;
}


template <int dim>
double torus_function_value(const Point<dim> &point)
{
  double x = point[0];
  double y = point[1];
  double z = point[2];
  double value  = 2 * y * (pow(y, 2) - 3 * pow(x, 2)) * (1 - pow(z, 2)) + pow((pow(x, 2) + pow(y, 2)), 2) - (9 * pow(z, 2) - 1) * (1 - pow(z, 2));
  return value;
}


// Mathematica code
// Grad[x^2 + y^2 - (1 + c1 * Cos[4*ArcTan[y/x]] +  c2 * Cos[8*ArcTan[y/x]]), {x, y}]
template <int dim>
Tensor<1, dim> pore_function_gradient(const Point<dim> &point, double c1, double c2)
{
  double x = point[0];
  double y = point[1];
  double theta = atan2(y, x);
  double r_square = pow(x, 2) + pow(y, 2);
  Tensor<1, dim> gradient;
  gradient[0] = -4 * c1 * y * sin(4 * theta) / r_square - 8 * c2 * y * sin(8 * theta) / r_square + 2 * x;
  gradient[1] = 4 * c1 * x * sin(4 * theta) / r_square + 8 * c2 * x * sin(8 * theta) / r_square  + 2 * y;
  return gradient;
}


// Mathematica code
// Grad[x^2 + y^2 - (1 + 0.2 * Sin[6*ArcTan[y/x]]), {x, y}]
template <int dim>
Tensor<1, dim> star_function_gradient(const Point<dim> &point)
{
  double x = point[0];
  double y = point[1];
  double theta = atan2(y, x);
  double r_square = pow(x, 2) + pow(y, 2);
  Tensor<1, dim> gradient;
  gradient[0] =  1.2 * y * cos(6 * theta) / r_square + 2 * x;
  gradient[1] =  -1.2 * x * cos(6 * theta) / r_square + 2 * y;
  return gradient;
}


template <int dim>
Tensor<1, dim> sphere_function_gradient(const Point<dim> &point)
{
  double x = point[0];
  double y = point[1];
  double z = point[2];
  Tensor<1, dim> gradient;
  gradient[0] = 2 * x;
  gradient[1] = 2 * y;
  gradient[2] = 2 * z;
  return gradient;
}

// Mathematica code
//  Grad[2*y*(y^2 - 3*x^2)*(1 - z^2) + (x^2 + y^2)^2 - (9*z^2 - 1)*(1 - z^2)), {x, y, z}]
template <int dim>
Tensor<1, dim> torus_function_gradient(const Point<dim> &point)
{
  double x = point[0];
  double y = point[1];
  double z = point[2];
  Tensor<1, dim> gradient;
  gradient[0] = 4 * x * (pow(x, 2) + pow(y, 2)) - 12 * x * y * (1 -  pow(z, 2));
  gradient[1] = 2 * (1 - pow(z, 2)) * (pow(y, 2) - 3 * pow(x, 2)) + 4 * y * (pow(x, 2) + pow(y, 2)) + 4 * pow(y, 2) * (1 - pow(z, 2));
  gradient[2] = -4 * y * z * (pow(y, 2) - 3 * pow(x, 2)) - 18 * (1 - pow(z, 2)) * z + 2 * (9 * pow(z, 2) - 1) * z;
  return gradient;
}


template <int dim>
void pore_function(std::vector<double> &u, const std::vector<Point<dim>> &points, int length, double c1, double c2)
{
  for (int i = 0; i < length; ++i)
  {
    u[i] = pore_function_value(points[i], c1, c2);
  }
}


template <int dim>
void star_function(std::vector<double> &u, const std::vector<Point<dim>> &points, int length)
{
  for (int i = 0; i < length; ++i)
  {
    u[i] = star_function_value(points[i]);
  }
}


template <int dim>
void sphere_function(std::vector<double> &u, const std::vector<Point<dim>> &points, int length)
{
  for (int i = 0; i < length; ++i)
  {
    u[i] = sphere_function_value(points[i]);
  }
}


template <int dim>
void torus_function(std::vector<double> &u, const std::vector<Point<dim>> &points, int length)
{
  for (int i = 0; i < length; ++i)
  {
    u[i] = torus_function_value(points[i]);
  }
}


template <int dim>
double cross_product_norm(const Tensor<1, dim> &t1, const Tensor<1, dim> &t2)
{
  if (dim == 2)
    return abs(t1[0] * t2[1] - t1[1] * t2[0]);
  else
    return sqrt(pow(t1[1] * t2[2] - t1[2] * t2[1], 2) + pow(t1[0] * t2[2] - t1[2] * t2[0], 2) + pow(t1[0] * t2[1] - t1[1] * t2[0], 2));
}


/* Construct map and return distance vector.
   Based on https://doi.org/10.1137/S106482750037617X */
template <int dim>
void sbm_map_newton(std::vector<Point<dim>> &target_points,
                    std::vector<Tensor<1, dim>> &normal_vectors,
                    std::vector<Tensor<1, dim>> &distance_vectors,
                    const std::vector<Point<dim>> &points,
                    int length,
                    std::function<double (const Point<dim> &)> &function_value,
                    std::function<Tensor<1, dim>(const Point<dim> &)> &function_gradient)
{
  for (int i = 0; i < length; ++i)
  {
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

    phi = function_value(target_point);
    grad_phi = function_gradient(target_point);

    while (res > tol && step < max_step)
    {
      delta1 = -phi * grad_phi / (grad_phi * grad_phi);
      delta2 = (points[i] - target_point) - ( (points[i] - target_point) * grad_phi / (grad_phi * grad_phi) ) * grad_phi;
      target_point = target_point + relax_param * (delta1 + delta2);

      // Bound the point
      target_point[0] = target_point[0] > DOMAIN_SIZE ? 0. : target_point[0];
      target_point[0] = target_point[0] < -DOMAIN_SIZE ? 0. : target_point[0];
      target_point[1] = target_point[1] > DOMAIN_SIZE ? 0. : target_point[1];
      target_point[1] = target_point[1] < -DOMAIN_SIZE ? 0. : target_point[1];

      phi = function_value(target_point);
      grad_phi = function_gradient(target_point);
      res = abs(phi) + cross_product_norm(grad_phi, (points[i] - target_point));
      step++;
    }

    if (res > tol)
    {
      tol = 1e-5;
      relax_param = 0.1;
      while (abs(phi) > tol)
      {
        delta1 = -phi * grad_phi / (grad_phi * grad_phi);
        target_point = target_point + relax_param * (delta1);
        phi = function_value(target_point);
        grad_phi = function_gradient(target_point);
        res = abs(phi);
        step++;
      }
      // std::cout << "  End of bad point converge at step " << step << " mapped point " << target_point << " phi value " << phi << std::endl;
    }

    target_points[i] = target_point;
    normal_vectors[i] = -grad_phi / grad_phi.norm();
    distance_vectors[i] = target_point - points[i];
    // std::cout << "  End of this call to sbm_map, surrogate points[i]: " << points[i]
    //           << "  phi value "  << function_value(target_point)
    //           << "  mapped points " << target_point << std::endl;
  }
  // std::cout << std::endl;
}


template <int dim>
void sbm_map_binary_search(std::vector<Point<dim>> &target_points,
                           std::vector<Tensor<1, dim>> &normal_vectors,
                           std::vector<Tensor<1, dim>> &distance_vectors,
                           const std::vector<Point<dim>> &points,
                           int length,
                           double h,
                           std::function<double (const Point<dim> &)> &function_value)
{
  for (int i = 0; i < length; ++i)
  {
    Point<dim> begin_point = points[i];
    Tensor<1, dim> normal_vector = normal_vectors[i];
    Point<dim> end_point = begin_point;
    double phi;
    do
    {
      end_point += h * normal_vector;
      phi = function_value(end_point);
    }
    while (phi < 0);

    double tol = 1e-6;
    double res;
    Point<dim> middle_point = begin_point;

    do
    {
      res = function_value(middle_point);
      if (res < 0)
        begin_point = middle_point;
      else
        end_point = middle_point;
      middle_point = (begin_point + end_point) / 2;
    }
    while (abs(res) > tol);

    target_points[i] = middle_point;
    distance_vectors[i] = target_points[i] - points[i];

    // std::cout << "  End of this call to sbm_map, surrogate points[i]: " << points[i]
    //           << "  phi value "  << function_value(middle_point)
    //           << "  mapped points " << middle_point << std::endl;
  }
  // std::cout << std::endl;
}



// Potentially useful functions
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
void set_support_points(hp::DoFHandler<dim> &dof_handler, std::vector<Point<dim>> &support_points)
{
  hp::MappingCollection<dim> mapping_collection;
  mapping_collection.push_back(MappingQ1<dim>());
  DoFTools::map_dofs_to_support_points(mapping_collection, dof_handler, support_points);
}


template <int dim>
void initialize_pore(hp::DoFHandler<dim> &dof_handler, Vector<double> &solution, double c1, double c2)
{
  std::vector<Point<dim>> support_points(dof_handler.n_dofs());
  set_support_points(dof_handler, support_points);
  for (unsigned int i = 0; i < dof_handler.n_dofs(); i++)
  {
    solution(i) = pore_function_value(support_points[i], c1, c2);
  }
}


template <int dim>
void initialize_torus(hp::DoFHandler<dim> &dof_handler, Vector<double> &solution)
{
  std::vector<Point<dim>> support_points(dof_handler.n_dofs());
  set_support_points(dof_handler, support_points);
  for (unsigned int i = 0; i < dof_handler.n_dofs(); i++)
  {
    solution(i) = torus_function_value(support_points[i]);
  }
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
    // solution(i) *= pow(support_points[i][0] - 0.25, 2) +  pow(support_points[i][1] - 0.25, 2) + 0.1;
  }
}


#endif