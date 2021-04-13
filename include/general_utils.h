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

// Mathematica code
// ContourPlot[ x^2+ y^2 - (1 -0.2*Cos[4*ArcTan[y/x]] -0.2*Cos[8*ArcTan[y/x]]) , {x,-2,2}, {y,-2,2}]
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


template <int dim>
double square_function_value(const Point<dim> &point)
{
  double length = 0.8;
  double x = point[0];
  double y = point[1];
  double value;
  if (y >= -x && y <= x && y >= -length && y <= length)
    value = x - length;
  if (y >= -x && y >= x && x >= -length && x <= length)
    value = y - length;
  if (y <= -x && y >= x && y >= -length && y <= length)
    value = -length - x;
  if (y <= -x && y <= x && x >= -length && x <= length)
    value = -length - y;
  if (x >= length && y >= length)
    value = sqrt(pow(x - length, 2) + pow(y - length, 2));
  if (x <= -length && y >= length)
    value = sqrt(pow(x + length, 2) + pow(y - length, 2));
  if (x <= -length && y <= -length)
    value = sqrt(pow(x + length, 2) + pow(y + length, 2));
  if (x >= length && y <= -length)
    value = sqrt(pow(x - length, 2) + pow(y + length, 2));
  return 2 * value;
}


template <int dim>
double peanut_phi_function_value(const Point<dim> &point)
{
  double r = 1.;
  double a = 0.7;
  double b = sqrt(pow(r, 2.) - pow(a, 2.));
  double x = point[0];
  double y = point[1];
  double value;
  if ((a - x) / sqrt((pow(a - x, 2.) + pow(y, 2.))) >= a / r && (a + x) / sqrt((pow(a + x, 2.) + pow(y, 2.))) >= a / r)
  {
    if (y >= 0)
      value = -sqrt(pow(x, 2.) +  pow(y - b, 2.));
    else
      value = -sqrt(pow(x, 2.) +  pow(y + b, 2.));
  }
  else
  {
    if (x >= 0)
      value = sqrt(pow(x - a, 2.) + pow(y, 2.)) - r;
    else
      value = sqrt(pow(x + a, 2.) + pow(y, 2.)) - r;
  }
  return value;
}


// quadratic 
template <int dim>
double peanut_q_function_value(const Point<dim> &point)
{
  double x = point[0];
  double y = point[1];
  return pow(x - 1, 2.) + pow(y - 1, 2.) + 0.1;
}


// sin
template <int dim>
double peanut_s_function_value(const Point<dim> &point)
{
  double k = 6;
  double x = point[0];
  double y = point[1];
  return 0.5 * sin(k * M_PI * x) * sin(k * M_PI * y) + 1;
}


template <int dim>
double peanut_function_value(const Point<dim> &point)
{
  return peanut_phi_function_value(point) * peanut_s_function_value(point);
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
Tensor<1, dim> square_function_gradient(const Point<dim> &point)
{
  double length = 0.8;
  double x = point[0];
  double y = point[1];
  Tensor<1, dim> gradient;
  if (y >= -x && y <= x && y >= -length && y <= length)
  {
    gradient[0] = 1;
    gradient[1] = 0;
  }
  if (y >= -x && y >= x && x >= -length && x <= length)
  {
    gradient[0] = 0;
    gradient[1] = 1;
  }
  if (y <= -x && y >= x && y >= -length && y <= length)
  {
    gradient[0] = -1;
    gradient[1] = 0;
  }
  if (y <= -x && y <= x && x >= -length && x <= length)
  {
    gradient[0] = 0;
    gradient[1] = -1;
  }
  if (x >= length && y >= length)
  {
    gradient[0] = x - length;
    gradient[1] = y - length;
  }
  if (x <= -length && y >= length)
  {
    gradient[0] = x + length;
    gradient[1] = y - length;
  }
  if (x <= -length && y <= -length)
  {
    gradient[0] = x + length;
    gradient[1] = y + length;
  }
  if (x >= length && y <= -length)
  {
    gradient[0] = x - length;
    gradient[1] = y + length;
  }
  gradient /= gradient.norm();
  return 2 * gradient;
}


template <int dim>
Tensor<1, dim> peanut_phi_function_gradient(const Point<dim> &point)
{
  double r = 1.;
  double a = 0.7;
  double b = sqrt(pow(r, 2.) - pow(a, 2.));
  double x = point[0];
  double y = point[1];
  Tensor<1, dim> gradient;
  if ((a - x) / sqrt((pow(a - x, 2.) + pow(y, 2.))) >= a / r && (a + x) / sqrt((pow(a + x, 2.) + pow(y, 2.))) >= a / r)
  {
    if (y >= 0)
    {
      gradient[0] = -x;
      gradient[1] = b - y;
    }
    else
    {
      gradient[0] = -x;
      gradient[1] = -b - y;
    }
  }
  else
  {
    if (x >= 0)
    {
      gradient[0] = x - a;
      gradient[1] = y;
    }
    else
    {
      gradient[0] = x + a;
      gradient[1] = y;
    }
  }
  gradient /= gradient.norm();
  return gradient;
}


template <int dim>
Tensor<1, dim> peanut_q_function_gradient(const Point<dim> &point)
{
  double x = point[0];
  double y = point[1];
  Tensor<1, dim> gradient;
  gradient[0] = 2 * (x - 1);
  gradient[1] = 2 * (y - 1);
  return gradient;
}


template <int dim>
Tensor<1, dim> peanut_s_function_gradient(const Point<dim> &point)
{
  double k = 6;
  double x = point[0];
  double y = point[1];
  Tensor<1, dim> gradient;
  gradient[0] = k * M_PI / 2. * cos(k * M_PI * x) * sin (k * M_PI * y);
  gradient[1] = k * M_PI / 2. * sin(k * M_PI * x) * cos (k * M_PI * y);
  return gradient;
}


template <int dim>
Tensor<1, dim> peanut_function_gradient(const Point<dim> &point)
{
  return peanut_s_function_value(point) * peanut_phi_function_gradient(point) +
         peanut_phi_function_value(point) * peanut_s_function_gradient(point);
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
void square_function(std::vector<double> &u, const std::vector<Point<dim>> &points, int length)
{
  for (int i = 0; i < length; ++i)
  {
    u[i] = square_function_value(points[i]);
  }
}


template <int dim>
void peanut_function(std::vector<double> &u, const std::vector<Point<dim>> &points, int length)
{
  for (int i = 0; i < length; ++i)
  {
    u[i] = peanut_function_value(points[i]);
  }
}


template <int dim>
void circle_exact_solution_value(std::vector<double> &u_e, const std::vector<Point<dim>> &points, int length)
{
  for (int i = 0; i < length; ++i)
  {
    u_e[i] =  sqrt(points[i].square()) - 1;
  }
}


template <int dim>
void circle_exact_solution_gradient(std::vector<Tensor<1, dim>> &g_e, const std::vector<Point<dim>> &points, int length)
{
  for (int i = 0; i < length; ++i)
  {
    Tensor<1, dim> gradient;
    for (int d = 0; d < dim; ++d)
    {
      gradient[d] = points[i][d] / sqrt(points[i].square());
    }
    g_e[i] =  gradient;
  }
}


template <int dim>
void peanut_exact_solution_value(std::vector<double> &u_e, const std::vector<Point<dim>> &points, int length)
{
  for (int i = 0; i < length; ++i)
  {
    u_e[i] = peanut_phi_function_value(points[i]);
  }
}


template <int dim>
void peanut_exact_solution_gradient(std::vector<Tensor<1, dim>> &g_e, const std::vector<Point<dim>> &points, int length)
{
  for (int i = 0; i < length; ++i)
  {
    Tensor<1, dim> gradient;
    g_e[i] =  peanut_phi_function_gradient(points[i]);
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
      std::cout << "  End of bad point " << points[i] << " converge at step " << step
                << " mapped point " << target_point << " phi value " << phi << std::endl;
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
    assert(function_value(points[i]) < 0 && "Error! Binary search should start inside of domain!");

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


// These functions are deprecated since support points don't work well with FE_Nothing
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