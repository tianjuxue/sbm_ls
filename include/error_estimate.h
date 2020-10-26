#ifndef ERROR_ESTIMATE_H
#define ERROR_ESTIMATE_H

using namespace dealii;


// Only a circle in the 2D case or a sphere in the 3D case has exact solutions
template <int dim>
void exact_solution(std::vector<double> &u_e, const std::vector<Point<dim>> &points, int length)
{
  for (int i = 0; i < length; ++i)
  {
    u_e[i] =  1 - sqrt(points[i].square());
  }
}


template <int dim>
double compute_L2_error(hp::DoFHandler<dim> &dof_handler,
                        Vector<double> &solution,
                        hp::FECollection<dim> &fe_collection,
                        hp::QCollection<dim> &q_collection,
                        unsigned int domain_flag)
{
  double l2_error_square = 0;
  hp::FEValues<dim> fe_values_hp (fe_collection, q_collection,
                                  update_values    |  update_gradients |
                                  update_quadrature_points  |  update_JxW_values);

  typename hp::DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  for (; cell != endc; ++cell)
  {
    if (cell->active_fe_index() == 0 || domain_flag == GLOBAL)
    {
      fe_values_hp.reinit (cell);
      const FEValues<dim> &fe_values = fe_values_hp.get_present_fe_values();
      const unsigned int &n_q_points = fe_values.n_quadrature_points;

      std::vector<double> solution_values(n_q_points);
      fe_values.get_function_values (solution, solution_values);
      std::vector<double> u_exact(n_q_points);
      exact_solution(u_exact, fe_values.get_quadrature_points(), n_q_points);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        double local_l2_error =  solution_values[q] - u_exact[q];
        l2_error_square += local_l2_error * local_l2_error * fe_values.JxW(q);
      }
    }
  }
  const double L2_error = std::sqrt(l2_error_square);
  return L2_error;
}



template <int dim>
double compute_Linfty_error(hp::DoFHandler<dim> &dof_handler,
                            Vector<double> &solution,
                            hp::FECollection<dim> &fe_collection,
                            unsigned int domain_flag)
{

  hp::QCollection<dim> q_collection_custom;
  const QTrapez<1>     q_trapez;
  const QIterated<dim> q_iterated(q_trapez, 3); // fe->degree * 2 + 1
  q_collection_custom.push_back(q_iterated);
  q_collection_custom.push_back(q_iterated);

  double linfty_error = 0;
  hp::FEValues<dim> fe_values_hp (fe_collection, q_collection_custom,
                                  update_values    |  update_gradients |
                                  update_quadrature_points  |  update_JxW_values);

  typename hp::DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  for (; cell != endc; ++cell)
  {
    if (cell->active_fe_index() == 0 || domain_flag == GLOBAL)
    {
      fe_values_hp.reinit (cell);
      const FEValues<dim> &fe_values = fe_values_hp.get_present_fe_values();
      const unsigned int &n_q_points = fe_values.n_quadrature_points;

      std::vector<double> solution_values(n_q_points);
      fe_values.get_function_values (solution, solution_values);
      std::vector<double> u_exact(n_q_points);
      exact_solution(u_exact, fe_values.get_quadrature_points(), n_q_points);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        linfty_error = std::max(linfty_error, abs(solution_values[q] - u_exact[q]));
      }
    }
  }
  return linfty_error;
}


template <int dim>
double compute_SD_error(hp::DoFHandler<dim> &dof_handler,
                        Vector<double> &solution,
                        hp::FECollection<dim> &fe_collection,
                        hp::QCollection<dim> &q_collection,
                        unsigned int domain_flag)
{

  double SD_error_square = 0;
  hp::FEValues<dim> fe_values_hp (fe_collection, q_collection,
                                  update_values    |  update_gradients |
                                  update_quadrature_points  |  update_JxW_values);

  typename hp::DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  for (; cell != endc; ++cell)
  {
    if (cell->active_fe_index() == 0 || domain_flag == GLOBAL)
    {
      fe_values_hp.reinit (cell);
      const FEValues<dim> &fe_values = fe_values_hp.get_present_fe_values();
      const unsigned int &n_q_points = fe_values.n_quadrature_points;
      std::vector<Tensor<1, dim>> solution_gradients(n_q_points);
      fe_values.get_function_gradients(solution, solution_gradients);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        double grad_norm = solution_gradients[q].norm();
        double local_SD_error =  grad_norm - 1;
        SD_error_square += local_SD_error * local_SD_error * fe_values.JxW(q);
      }
    }
  }
  const double SD_error = std::sqrt(SD_error_square);
  return SD_error;
}


template <int dim>
double compute_interface_error_pore(hp::DoFHandler<dim> &dof_handler,
                                    Vector<double> &solution,
                                    double c1,
                                    double c2)
{
  Functions::FEFieldFunction<dim, hp::DoFHandler<dim>, Vector<double>> fe_field_function(dof_handler, solution);
  unsigned int num_intervals = 100000;
  std::vector<Point<dim>> quad_points;
  for (unsigned int i = 0; i < num_intervals + 1; ++i)
  {
    double quad_theta = i * 2 * M_PI / num_intervals;
    double radius = sqrt(1 + c1 * cos(4 * quad_theta) + c2 * cos(8 * quad_theta));
    Point<dim> quad_point;
    quad_point[0] = radius * cos(quad_theta);
    quad_point[1] = radius * sin(quad_theta);
    quad_points.push_back(quad_point);
  }
  std::vector<double> quad_values(num_intervals + 1, 0.);
  fe_field_function.value_list(quad_points, quad_values);
  double line_integral = 0;
  for (unsigned int i = 0; i < num_intervals; ++i)
  {
    double value_start = pow(quad_values[i], 2);
    double value_end = pow(quad_values[i + 1], 2);
    double interval_measure = (quad_points[i + 1] - quad_points[i]).norm();
    line_integral += (value_start + value_end) * interval_measure / 2; // Trapezoid rule
  }
  return line_integral;
}


// Reference: http://math.mit.edu/~jorloff/suppnotes/suppnotes02/v9.pdf
template <int dim>
double compute_interface_error_sphere(hp::DoFHandler<dim> &dof_handler, Vector<double> &solution)
{
  Functions::FEFieldFunction<dim, hp::DoFHandler<dim>, Vector<double>> fe_field_function(dof_handler, solution);

  unsigned int num_intervals_theta = 2000;
  unsigned int num_intervals_phi = 1000;

  std::vector<Point<dim>> quad_points;
  std::vector<double> jacobians;
  for (unsigned int i = 0; i < num_intervals_theta; ++i)
  {
    double quad_theta = (i + 1. / 2.) * 2 * M_PI / num_intervals_theta;
    for (unsigned j = 0; j < num_intervals_phi; ++j)
    {
      double quad_phi = (j + 1. / 2.) * M_PI / num_intervals_phi;
      Point<dim> quad_point;
      quad_point[0] = RADIUS * sin(quad_phi) * cos(quad_theta);
      quad_point[1] = RADIUS * sin(quad_phi) * sin(quad_theta);
      quad_point[2] = RADIUS * cos(quad_phi);
      quad_points.push_back(quad_point);
      jacobians.push_back(pow(RADIUS, 2) * sin(quad_phi));
    }
  }

  std::vector<double> quad_values(num_intervals_theta * num_intervals_phi, 0.);
  std::cout << "  Start computing quad point values..." << std::endl;
  fe_field_function.value_list(quad_points, quad_values);
  std::cout << "  Finish computing quad point values" << std::endl;
  double surface_integral = 0;
  double weight = 2 * M_PI / num_intervals_theta * M_PI / num_intervals_phi;
  double area = 0;
  for (unsigned int i = 0; i < num_intervals_theta; ++i)
  {
    for (unsigned j = 0; j < num_intervals_phi; ++j)
    {
      double quad_value = quad_values[i * num_intervals_phi + j];
      double jacobian = jacobians[i * num_intervals_phi + j];
      surface_integral += pow(quad_value, 2) * jacobian * weight;
      area += jacobian * weight;
    }
  }

  std::cout << "  Area is " << area << " it should be " << 4 * M_PI * pow(RADIUS, 2) << std::endl;
  std::cout << "  surface_integral is " << surface_integral << std::endl;
  return surface_integral;
}


// Use pre-defined quadratures and weights to compute error
// Moment fitting method to compute interface error
// Reference: https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.4569
// Or shifted bounday integration scheme
template <int dim>
double compute_interface_error_qw(hp::DoFHandler<dim> &dof_handler, Vector<double> &solution,
                                  std::string &quads_dat_filename, std::string &weights_dat_filename)
{
  std::cout << "  quads_dat_filename: " << quads_dat_filename << " and weights_dat_filename: " << weights_dat_filename << std::endl;
  std::ifstream input_quads_file(quads_dat_filename);
  std::ifstream input_weights_file(weights_dat_filename);
  std::vector<Point<dim>> quad_points;
  std::vector<double> weights;
  std::cout << "  Reading quads file and weights file... " << std::endl;
  if (input_quads_file.is_open() && input_weights_file.is_open())
  {
    double weight;
    Point<dim> quad_point;
    while (input_weights_file >> weight)
    {
      weights.push_back(weight);
      for (int i = 0; i < dim; ++i)
        input_quads_file >> quad_point[i];
      quad_points.push_back(quad_point);
    }
  }
  else
  {
    assert(0 && "File not open!");
  }
  input_weights_file.close();
  input_quads_file.close();
  std::cout << "  Read in " << weights.size() << " quad points and " <<   weights.size() << " weights" << std::endl;

  std::vector<double> quad_values(weights.size(), 0.);
  Functions::FEFieldFunction<dim, hp::DoFHandler<dim>, Vector<double>> fe_field_function(dof_handler, solution);
  std::cout << "  Start computing quad point values..." << std::endl;
  fe_field_function.value_list(quad_points, quad_values);
  std::cout << "  Finish computing quad point values" << std::endl;

  double surface_integral = 0;
  double area = 0;
  for (unsigned i = 0; i < weights.size(); ++i)
  {
    surface_integral += pow(quad_values[i], 2) * weights[i];
    area += weights[i];
  }

  std::cout << "  Area is " << area << " it should be " << 4 * M_PI * pow(RADIUS, 2) << std::endl;
  std::cout << "  surface_integral is " << surface_integral << std::endl;
  return surface_integral;
}

#endif