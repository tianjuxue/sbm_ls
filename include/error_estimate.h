#ifndef ERROR_ESTIMATE_H
#define ERROR_ESTIMATE_H

using namespace dealii;


// The function leaves information for python finite difference script
// so that the same narrow band setup can be established in python script
template <int dim>
void write_narrow_band_solutions(hp::DoFHandler<dim> &dof_handler,
                                 Vector<double> &solution,
                                 hp::FECollection<dim> &fe_collection,
                                 unsigned int case_flag,
                                 unsigned int total_refinement,
                                 int picard_step)
{

  std::string dat_filename = "../data/dat/finite_difference/case_" +  Utilities::int_to_string(case_flag, 1) + "-sin"
                             "/refinement_" +  Utilities::int_to_string(total_refinement, 1) +
                             "/solution_step_" + Utilities::int_to_string(picard_step, 1) +
                             ".dat";
  std::ofstream dat_file;
  dat_file.open(dat_filename);

  hp::QCollection<dim> q_collection_custom;
  const QTrapez<1>     q_trapez;
  const QIterated<dim> q_iterated(q_trapez, 1); // (0, 0) (1, 0) (0, 1) (1, 1)
  q_collection_custom.push_back(q_iterated);
  q_collection_custom.push_back(q_iterated);

  hp::FEValues<dim> fe_values_hp (fe_collection, q_collection_custom,
                                  update_values    |  update_gradients |
                                  update_quadrature_points  |  update_JxW_values);

  typename hp::DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  for (; cell != endc; ++cell)
  {
    if (cell->active_fe_index() == 0)
    {
      fe_values_hp.reinit (cell);
      const FEValues<dim> &fe_values = fe_values_hp.get_present_fe_values();
      const unsigned int &n_q_points = fe_values.n_quadrature_points;
      const std::vector<Point<dim>> quad_points = fe_values.get_quadrature_points();

      std::vector<double> solution_values(n_q_points);
      fe_values.get_function_values(solution, solution_values);

      dat_file << cell->center() << " ";
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        dat_file << solution_values[q] << " ";
      }
      dat_file << std::endl;
    }
  }

}


template <int dim>
double compute_volume_error(hp::DoFHandler<dim> &dof_handler,
                            Vector<double> &solution,
                            hp::FECollection<dim> &fe_collection,
                            unsigned int domain_flag,
                            unsigned int case_flag)
{
  double volume = 0;

  hp::QCollection<dim> q_collection_custom;
  // const QTrapez<1>     q_trapez;
  // const QIterated<dim> q_iterated(q_trapez, 3); // fe->degree * 2 + 1

  const QMidpoint<1> q_mid;
  const QIterated<dim> q_iterated(q_mid, 10); // fe->degree * 2 + 1

  q_collection_custom.push_back(q_iterated);
  q_collection_custom.push_back(q_iterated);


  // const std::vector< Point< dim > > quad = q_iterated.get_points();
  // const std::vector<double> weights = q_iterated.get_weights();
  // for (unsigned int i = 0; i < quad.size(); ++i)
  // {
  //   std::cout << "quad point " << quad[i] << " weight " << weights[i] <<std::endl;
  // }

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
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        double step_value;
        if (solution_values[q] < 0)
          step_value = 1;
        else
          step_value = 0;
        volume += step_value * fe_values.JxW(q);
      }
    }
    else
    {
      if (cell->material_id() == FLAG_IN)
        volume += cell->measure();
    }
  }
  double exact_volume = 0.;

  if (case_flag == PORE_CASE)
  {
    // For all 9 pore cases, the exact volue is simply M_PI
    exact_volume = M_PI;
  }
  else if (case_flag == SPHERE_CASE)
  {
    exact_volume = 4. / 3. * M_PI;
  }
  else
  {
    std::cout << "Warning: Exact volume unknown and return 0.";
    return 0;
  }

  return abs((volume - exact_volume) / exact_volume);

}



template <int dim>
double compute_L2_error(hp::DoFHandler<dim> &dof_handler,
                        Vector<double> &solution,
                        hp::FECollection<dim> &fe_collection,
                        hp::QCollection<dim> &q_collection,
                        unsigned int domain_flag,
                        unsigned int case_flag)
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
    if (cell->active_fe_index() == 0)
    {
      fe_values_hp.reinit (cell);
      const FEValues<dim> &fe_values = fe_values_hp.get_present_fe_values();
      const unsigned int &n_q_points = fe_values.n_quadrature_points;

      std::vector<double> solution_values(n_q_points);
      fe_values.get_function_values (solution, solution_values);

      std::vector<double> u_exact(n_q_points);
      if (case_flag == PORE_CASE || case_flag == SPHERE_CASE)
        circle_exact_solution_value(u_exact, fe_values.get_quadrature_points(), n_q_points);
      else if (case_flag == PEANUT_CASE)
        peanut_exact_solution_value(u_exact, fe_values.get_quadrature_points(), n_q_points);
      else
        assert(0 && "Exact solutions of other cases may not exist!");

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        double local_l2_error = solution_values[q] - u_exact[q];
        l2_error_square += local_l2_error * local_l2_error * fe_values.JxW(q);
      }
    }
  }
  const double L2_error = std::sqrt(l2_error_square);
  return L2_error;
}


template <int dim>
double compute_H1_error(hp::DoFHandler<dim> &dof_handler,
                        Vector<double> &solution,
                        hp::FECollection<dim> &fe_collection,
                        hp::QCollection<dim> &q_collection,
                        unsigned int domain_flag,
                        unsigned int case_flag)
{
  double h1_error_square = 0;
  hp::FEValues<dim> fe_values_hp (fe_collection, q_collection,
                                  update_values    |  update_gradients |
                                  update_quadrature_points  |  update_JxW_values);

  typename hp::DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  for (; cell != endc; ++cell)
  {
    if (cell->active_fe_index() == 0)
    {
      fe_values_hp.reinit (cell);
      const FEValues<dim> &fe_values = fe_values_hp.get_present_fe_values();
      const unsigned int &n_q_points = fe_values.n_quadrature_points;

      std::vector<double> solution_values(n_q_points);
      fe_values.get_function_values (solution, solution_values);
      std::vector<double> u_exact(n_q_points);

      std::vector<Tensor<1, dim>> solution_gradients(n_q_points);
      fe_values.get_function_gradients(solution, solution_gradients);
      std::vector<Tensor<1, dim>> g_exact(n_q_points);

      if (case_flag == PORE_CASE || case_flag == SPHERE_CASE)
      {
        circle_exact_solution_value(u_exact, fe_values.get_quadrature_points(), n_q_points);
        circle_exact_solution_gradient(g_exact, fe_values.get_quadrature_points(), n_q_points);
      }
      else if (case_flag == PEANUT_CASE)
      {
        peanut_exact_solution_value(u_exact, fe_values.get_quadrature_points(), n_q_points);
        peanut_exact_solution_gradient(g_exact, fe_values.get_quadrature_points(), n_q_points);
      }
      else
        assert(0 && "Exact solutions of other cases may not exist!");

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        double local_value_error = solution_values[q] - u_exact[q];
        h1_error_square += local_value_error * local_value_error * fe_values.JxW(q);
        Tensor<1, dim> local_grad_error = solution_gradients[q] - g_exact[q];
        h1_error_square += local_grad_error * local_grad_error * fe_values.JxW(q);

      }
    }
  }
  const double H1_error = std::sqrt(h1_error_square);
  return H1_error;
}


template <int dim>
double compute_Linfty_error(hp::DoFHandler<dim> &dof_handler,
                            Vector<double> &solution,
                            hp::FECollection<dim> &fe_collection,
                            unsigned int domain_flag,
                            unsigned int case_flag)
{

  hp::QCollection<dim> q_collection_custom;
  const QTrapez<1>     q_trapez;
  const QIterated<dim> q_iterated(q_trapez, 3); // fe->degree * 2 + 1
  // const QIterated<dim> q_iterated(q_trapez, 1); // Quad points coincide with the four vertices
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
    if (cell->active_fe_index() == 0)
    {
      fe_values_hp.reinit (cell);
      const FEValues<dim> &fe_values = fe_values_hp.get_present_fe_values();
      const unsigned int &n_q_points = fe_values.n_quadrature_points;

      std::vector<double> solution_values(n_q_points);
      fe_values.get_function_values(solution, solution_values);

      std::vector<double> u_exact(n_q_points);
      if (case_flag == PORE_CASE || case_flag == SPHERE_CASE)
        circle_exact_solution_value(u_exact, fe_values.get_quadrature_points(), n_q_points);
      else if (case_flag == PEANUT_CASE)
        peanut_exact_solution_value(u_exact, fe_values.get_quadrature_points(), n_q_points);
      else
        assert(0 && "Exact solutions of other cases may not exist!");

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        // Point<dim> p = fe_values.get_quadrature_points()[q];
        // if (abs(solution_values[q]) < 1.2 * 4 / pow(2, 6))
        linfty_error = std::max(linfty_error, abs(solution_values[q] - u_exact[q]));

        // std::cout << solution_values[q] << " " << u_exact[q] << " " << abs(solution_values[q] - u_exact[q]) << std::endl;
        // if (linfty_error < abs(solution_values[v] - u_exact[v]))
        // {
        //   p[0] = vertices[v][0];
        //   p[1] = vertices[v][1];
        //   std::cout << "max point " << p << std::endl;
        // }
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
    if (cell->active_fe_index() == 0)
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
void arc_integral_quad_points_helper(double radius,
                                     double c1,
                                     double c2,
                                     double x_center,
                                     double y_center,
                                     double theta_start,
                                     double theta_end,
                                     std::vector<Point<dim>> &quad_points,
                                     unsigned int num_intervals)
{
  for (unsigned int i = 0; i < num_intervals + 1; ++i)
  {
    double quad_theta = i * (theta_end - theta_start) / num_intervals + theta_start;
    double pore_radius = radius * sqrt(1 + c1 * cos(4 * quad_theta) + c2 * cos(8 * quad_theta));
    Point<dim> quad_point;
    quad_point[0] = pore_radius * cos(quad_theta) + x_center;
    quad_point[1] = pore_radius * sin(quad_theta) + y_center;
    quad_points[i] = quad_point;
  }
}


template <int dim>
double arc_integral_evaluate_helper(hp::DoFHandler<dim> &dof_handler,
                                    Vector<double> &solution,
                                    unsigned int num_intervals,
                                    std::vector<Point<dim>> &quad_points)
{
  Functions::FEFieldFunction<dim, hp::DoFHandler<dim>, Vector<double>> fe_field_function(dof_handler, solution);
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


template <int dim>
double compute_interface_error_peanut(hp::DoFHandler<dim> &dof_handler,
                                      Vector<double> &solution)
{
  unsigned int num_intervals = 100000;

  std::vector<Point<dim>> quad_points_left(num_intervals + 1);
  arc_integral_quad_points_helper(1., 0., 0., -0.7, 0., acos(0.7 / 1.), 2 * M_PI - acos(0.7 / 1.), quad_points_left, num_intervals);
  double left_part = arc_integral_evaluate_helper(dof_handler, solution, num_intervals, quad_points_left);

  std::vector<Point<dim>> quad_points_right(num_intervals + 1);
  arc_integral_quad_points_helper(1., 0., 0., 0.7, 0., acos(0.7 / 1.) - M_PI, M_PI - acos(0.7 / 1.), quad_points_right, num_intervals);
  double right_part = arc_integral_evaluate_helper(dof_handler, solution, num_intervals, quad_points_right);

  return std::sqrt(left_part + right_part);

}


template <int dim>
double compute_interface_error_pore(hp::DoFHandler<dim> &dof_handler,
                                    Vector<double> &solution,
                                    double c1,
                                    double c2)
{
  unsigned int num_intervals = 100000;
  std::vector<Point<dim>> quad_points(num_intervals + 1);
  arc_integral_quad_points_helper(1., c1, c2, 0., 0., 0., 2 * M_PI, quad_points, num_intervals);
  double result = arc_integral_evaluate_helper(dof_handler, solution, num_intervals, quad_points);
  return std::sqrt(result);
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
  std::cout << "  sqrt of surface_integral is " << std::sqrt(surface_integral) << std::endl;
  return std::sqrt(surface_integral);
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
  std::cout << "  Read in " << weights.size() << " quad points and " << weights.size() << " weights" << std::endl;

  std::vector<double> quad_values(weights.size(), 0.);
  Functions::FEFieldFunction<dim, hp::DoFHandler<dim>, Vector<double>> fe_field_function(dof_handler, solution);
  std::cout << "  Start computing quad point values..." << std::endl;
  fe_field_function.value_list(quad_points, quad_values);
  std::cout << "  Finish computing quad point values" << std::endl;

  double surface_integral = 0;
  double area = 0;
  for (unsigned i = 0; i < weights.size(); ++i)
  {
    // if (i % 1000 == 0)
    // {
    //   std::cout << std::endl;
    //   std::cout << "  quad value " << quad_values[i] << std::endl;
    //   std::cout << "  analytical value " << torus_function_value(quad_points[i]) << std::endl;
    // }
    surface_integral += pow(quad_values[i], 2) * weights[i];
    area += weights[i];
  }

  std::cout << "  Area is " << area << " sphere is " << 4 * M_PI * pow(RADIUS, 2) << std::endl;
  std::cout << "  sqrt of surface_integral is " << std::sqrt(surface_integral) << std::endl;
  return std::sqrt(surface_integral);
}

#endif