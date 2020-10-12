#ifndef ERROR_ESTIMATE_H
#define ERROR_ESTIMATE_H

using namespace dealii;


template <int dim>
void exact_solution (std::vector<double> &u_e, const std::vector<Point<dim>> &points, int length)
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
                        hp::QCollection<dim> &q_collection)
{

  // hp::QCollection<dim>    q_collection_custom;
  // q_collection_custom.push_back(QGauss<dim>(2));
  // q_collection_custom.push_back(QGauss<dim>(2));

  double l2_error_square = 0;
  hp::FEValues<dim> fe_values_hp (fe_collection, q_collection,
                                  update_values    |  update_gradients |
                                  update_quadrature_points  |  update_JxW_values);

  typename hp::DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  for (; cell != endc; ++cell)
  {
    if (cell->material_id() == FLAG_IN_BAND || cell->material_id() == FLAG_OUT_BAND)
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
double compute_SD_error(hp::DoFHandler<dim> &dof_handler,
                        Vector<double> &solution,
                        hp::FECollection<dim> &fe_collection,
                        hp::QCollection<dim> &q_collection)
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
    if (cell->material_id() == FLAG_IN_BAND || cell->material_id() == FLAG_OUT_BAND)
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
double compute_interface_error(hp::DoFHandler<dim> &dof_handler,
                               Vector<double> &solution,
                               double c1,
                               double c2)
{
  Functions::FEFieldFunction<dim, hp::DoFHandler<dim>, Vector<double>> fe_field_function(dof_handler, solution);

  unsigned int num_intervals = 100000;
  std::vector<Point<dim>> quad_points(num_intervals + 1);
  std::vector<double> line_values(num_intervals + 1);
  for (unsigned int i = 0; i < num_intervals + 1; ++i)
  {
    double quad_theta = i * 2 * M_PI / num_intervals;
    double radius = sqrt(1 + c1 * cos(4 * quad_theta) + c2 * cos(8 * quad_theta));
    double x = radius * cos(quad_theta);
    double y = radius * sin(quad_theta);
    quad_points[i][0] = x;
    quad_points[i][1] = y;
  }

  fe_field_function.value_list(quad_points, line_values);
  double line_integral = 0;
  for (unsigned int i = 0; i < num_intervals; ++i)
  {
    double value_start = pow(line_values[i], 2);
    double value_end = pow(line_values[i + 1], 2);
    double interval_measure = (quad_points[i + 1] - quad_points[i]).norm();
    line_integral += (value_start + value_end) * interval_measure / 2; // Trapezoid rule
  }
  return line_integral;
}


#endif