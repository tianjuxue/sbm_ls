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
double compute_error(hp::DoFHandler<dim> &dof_handler,
                     Vector<double> &solution,
                     hp::FECollection<dim> &fe_collection,
                     hp::QCollection<dim> &q_collection,
                     unsigned int FLAG_IN_BAND,
                     unsigned int FLAG_OUT_BAND)
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


#endif