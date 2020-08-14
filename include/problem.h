#ifndef PROBLEM
#define PROBLEM

#include "general_utils.h"

template <int dim>
class NonlinearProblem
{
public:
  NonlinearProblem(Triangulation<dim> &triangulation_,
                   hp::DoFHandler<dim> &dof_handler_all_,
                   Vector<double> &solution_all_,
                   AdvectionVelocity<dim> &velocity_,
                   bool in_out_flag_);
  ~NonlinearProblem();
  void run();
  hp::DoFHandler<dim> dof_handler;
  Vector<double> solution;

private:
  void setup_system();
  void assemble_system();
  void assemble_system_initial_guess();
  void solve(double relaxation_parameter);
  void output_results(unsigned int cycle);
  double compute_residual();


  hp::FECollection<dim>   fe_collection;
  hp::QCollection<dim>    q_collection;
  hp::QCollection < dim - 1 >  q_collection_face;

  AffineConstraints<double> constraints;
  SparsityPattern           sparsity_pattern;
  SparseMatrix<double>      system_matrix;

  Vector<double>            newton_update;
  Vector<double>            system_rhs;

  Triangulation<dim> &triangulation;
  hp::DoFHandler<dim> & dof_handler_all;
  Vector<double> &solution_all;
  AdvectionVelocity<dim> &velocity;
  bool in_out_flag;

  double h;
  double alpha;
};


template <int dim>
NonlinearProblem<dim>::NonlinearProblem(Triangulation<dim> &triangulation_,
                                        hp::DoFHandler<dim> &dof_handler_all_,
                                        Vector<double> &solution_all_,
                                        AdvectionVelocity<dim> &velocity_,
                                        bool in_out_flag_)
  :
  dof_handler(triangulation_),
  triangulation(triangulation_),
  dof_handler_all(dof_handler_all_),
  solution_all(solution_all_),
  velocity(velocity_),
  in_out_flag(in_out_flag_),
  h(GridTools::minimal_cell_diameter(triangulation_)),
  alpha(10.) // Magic number, pick a number you like
{

  q_collection.push_back(QGauss<dim>(2));
  q_collection.push_back(QGauss<dim>(2));

  q_collection_face.push_back(QGauss < dim - 1 > (2));
  q_collection_face.push_back(QGauss < dim - 1 > (2));


  fe_collection.push_back(FE_Q<dim>(1));
  fe_collection.push_back(FE_Q<dim>(1));

  // fe_collection.push_back(FE_Nothing<dim>());
  // fe_collection.push_back(FE_Q<dim>(1));
}


template <int dim>
NonlinearProblem<dim>::~NonlinearProblem()
{
  dof_handler.clear();
}


template <int dim>
void NonlinearProblem<dim>::setup_system()
{

  int counter = 0;
  for (typename hp::DoFHandler<dim>::cell_iterator cell = dof_handler.begin_active();
       cell != dof_handler.end(); ++cell)
  {
    bool is_cell_active = true;
    for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
    {
      std::cout << "counter " << counter << std::endl;
      counter++;
      bool is_vertex_inside = is_inside_manual(dof_handler_all, solution_all, cell->vertex(v));
      is_cell_active = is_cell_active && (in_out_flag ? is_vertex_inside : !is_vertex_inside);
    }

    if (is_cell_active)
    {
      cell->set_active_fe_index(1);
      cell->set_material_id(0);
    }
    else
    {
      cell->set_active_fe_index(1);
      cell->set_material_id(1);
    }

  }

  dof_handler.distribute_dofs(fe_collection);

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler,
                                          constraints);
  constraints.close();


  solution.reinit(dof_handler.n_dofs());
  newton_update.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
}



template <int dim>
void NonlinearProblem<dim>::assemble_system()
{

  system_matrix = 0;
  system_rhs    = 0;

  hp::FEValues<dim> fe_values_hp (fe_collection, q_collection,
                                  update_values    |  update_gradients |
                                  update_quadrature_points  |  update_JxW_values);

  hp::FEFaceValues<dim> fe_values_face_hp (fe_collection, q_collection_face,
      update_values    |  update_gradients |
      update_quadrature_points  |  update_JxW_values |
      update_normal_vectors);


  FullMatrix<double>   local_matrix;
  Vector<double>       local_rhs;
  std::vector<types::global_dof_index> local_dof_indices;

  typename hp::DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  SymmetricTensor<2, dim> unit_tensor = unit_symmetric_tensor<dim>();

  for (; cell != endc; ++cell)
  {
    if (cell->active_fe_index() == 1)
    {

      fe_values_hp.reinit(cell);

      const FEValues<dim> &fe_values = fe_values_hp.get_present_fe_values();
      unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
      unsigned int n_q_points    = fe_values.n_quadrature_points;

      local_matrix.reinit(dofs_per_cell, dofs_per_cell);
      local_rhs.reinit(dofs_per_cell);
      local_dof_indices.resize(dofs_per_cell);

      local_matrix = 0;
      local_rhs = 0;
      cell->get_dof_indices(local_dof_indices);

      std::vector<double> solution_values(n_q_points);
      fe_values.get_function_values(solution, solution_values);
      std::vector<Tensor<1, dim>> solution_gradients(n_q_points);
      fe_values.get_function_gradients(solution, solution_gradients);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        double grad_norm = solution_gradients[q].norm();
        Tensor<2, dim> A_tensor = (1. - 1. / grad_norm) * unit_tensor +
                                  outer_product(solution_gradients[q], solution_gradients[q]) / pow(grad_norm, 3.);
        double a_scalar = 1. - 1. / grad_norm;

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            local_matrix(i, j) += fe_values.shape_grad(i, q) * A_tensor * fe_values.shape_grad(j, q) * fe_values.JxW(q);
          }
          local_rhs(i) -= a_scalar * fe_values.shape_grad(i, q) * solution_gradients[q] * fe_values.JxW(q);
        }
      }

      for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
      {
        if (cell->face(face_no)->at_boundary()) /* Exterior boundary */
        {
          fe_values_face_hp.reinit(cell, face_no);
          const FEFaceValues<dim> &fe_values_face = fe_values_face_hp.get_present_fe_values();
          unsigned int n_face_q_points = fe_values_face.n_quadrature_points;

          std::vector<double> solution_values_face(n_face_q_points);
          fe_values_face.get_function_values(solution, solution_values_face);
          std::vector<Tensor<1, dim>> solution_gradients_face(n_face_q_points);
          fe_values_face.get_function_gradients(solution, solution_gradients_face);

          for (unsigned int q = 0; q < n_face_q_points; ++q)
          {
            double grad_norm_face = solution_gradients_face[q].norm();
            Tensor<2, dim> A_tensor_face = (1. - 1. / grad_norm_face) * unit_tensor +
                                           outer_product(solution_gradients_face[q], solution_gradients_face[q]) / pow(grad_norm_face, 3.);
            double a_scalar_face = 1. - 1. / grad_norm_face;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              // for (unsigned int j = 0; j < dofs_per_cell; ++j)
              // {
              //   local_matrix(i, j) -= fe_values_face.normal_vector(q) * A_tensor_face *
              //                         fe_values_face.shape_grad(j, q) * fe_values_face.shape_value(i, q) *
              //                         fe_values_face.JxW(q);
              //   local_matrix(i, j) += fe_values_face.normal_vector(q) * a_scalar_face *
              //                         fe_values_face.shape_grad(j, q) * fe_values_face.shape_value(i, q) *
              //                         fe_values_face.JxW(q);
              // }
              // double neumann_boundary_value = 0;
              // local_rhs(i) += a_scalar_face * neumann_boundary_value * fe_values_face.shape_value(i, q) *
              //                 fe_values_face.JxW(q);
              // local_rhs(i) += fe_values_face.normal_vector(q) * a_scalar_face *
              //                 solution_gradients_face[q] * fe_values_face.shape_value(i, q) *
              //                 fe_values_face.JxW(q);

            }
          }
        }
        // else if (cell->neighbor(face_no)->active_fe_index() == 0) /* Interior boundary */
        else if (cell->material_id() == 0 && cell->neighbor(face_no)->material_id() == 1)
        {
          fe_values_face_hp.reinit(cell, face_no);
          const FEFaceValues<dim> &fe_values_face = fe_values_face_hp.get_present_fe_values();
          unsigned int n_face_q_points = fe_values_face.n_quadrature_points;

          std::vector<Point<dim>> target_points(n_face_q_points);
          std::vector<Tensor<1, dim> > normal_vectors(n_face_q_points);
          std::vector<Tensor<1, dim> > distance_vectors(n_face_q_points);
          std::vector<double> boundary_values(n_face_q_points);
          sbm_map_manual(target_points, normal_vectors, distance_vectors, fe_values_face.get_quadrature_points(), n_face_q_points, dof_handler_all, solution_all);
          compute_boundary_values(velocity, target_points, normal_vectors, boundary_values, n_face_q_points);

          std::vector<double> solution_values_face(n_face_q_points);
          fe_values_face.get_function_values(solution, solution_values_face);
          std::vector<Tensor<1, dim>> solution_gradients_face(n_face_q_points);
          fe_values_face.get_function_gradients(solution, solution_gradients_face);

          for (unsigned int q = 0; q < n_face_q_points; ++q)
          {
            double grad_norm_face = solution_gradients_face[q].norm();
            Tensor<2, dim> A_tensor_face = (1. - 1. / grad_norm_face) * unit_tensor +
                                           outer_product(solution_gradients_face[q], solution_gradients_face[q]) / pow(grad_norm_face, 3.);
            double a_scalar_face = 1. - 1. / grad_norm_face;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                local_matrix(i, j) -= fe_values_face.normal_vector(q) * A_tensor_face *
                                      fe_values_face.shape_grad(j, q) * fe_values_face.shape_value(i, q) *
                                      fe_values_face.JxW(q);

                local_matrix(i, j) -= fe_values_face.normal_vector(q) * A_tensor_face * fe_values_face.shape_grad(i, q) *
                                      (fe_values_face.shape_value(j, q) + fe_values_face.shape_grad(j, q) * distance_vectors[q]) *
                                      fe_values_face.JxW(q);

                local_matrix(i, j) += alpha / h * (fe_values_face.shape_value(i, q) + fe_values_face.shape_grad(i, q) * distance_vectors[q]) *
                                      (fe_values_face.shape_value(j, q) + fe_values_face.shape_grad(j, q) * distance_vectors[q]) *
                                      fe_values_face.JxW(q);
              }
              local_rhs(i) += fe_values_face.normal_vector(q) * a_scalar_face *
                              solution_gradients_face[q] * fe_values_face.shape_value(i, q) *
                              fe_values_face.JxW(q);

              local_rhs(i) += fe_values_face.normal_vector(q) * A_tensor_face * fe_values_face.shape_grad(i, q) *
                              (solution_values_face[q] + solution_gradients_face[q] * distance_vectors[q] - boundary_values[q]) *
                              fe_values_face.JxW(q);

              local_rhs(i) -= alpha / h * (fe_values_face.shape_value(i, q) + fe_values_face.shape_grad(i, q) * distance_vectors[q]) *
                              (solution_values_face[q] + solution_gradients_face[q] * distance_vectors[q] - boundary_values[q]) *
                              fe_values_face.JxW(q);
            }
          }
        }
      }
      constraints.distribute_local_to_global(local_matrix,
                                             local_rhs,
                                             local_dof_indices,
                                             system_matrix,
                                             system_rhs);
    }
  }
}



template <int dim>
double NonlinearProblem<dim>::compute_residual()
{
  Vector<double> residual(dof_handler.n_dofs());

  hp::FEValues<dim> fe_values_hp (fe_collection, q_collection,
                                  update_values    |  update_gradients |
                                  update_quadrature_points  |  update_JxW_values);

  hp::FEFaceValues<dim> fe_values_face_hp (fe_collection, q_collection_face,
      update_values    |  update_gradients |
      update_quadrature_points  |  update_JxW_values |
      update_normal_vectors);

  Vector<double>       local_rhs;
  std::vector<types::global_dof_index> local_dof_indices;

  typename hp::DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  SymmetricTensor<2, dim> unit_tensor = unit_symmetric_tensor<dim>();

  for (; cell != endc; ++cell)
  {
    if (cell->active_fe_index() == 1)
    {

      fe_values_hp.reinit(cell);

      const FEValues<dim> &fe_values = fe_values_hp.get_present_fe_values();
      unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
      unsigned int n_q_points    = fe_values.n_quadrature_points;

      local_rhs.reinit(dofs_per_cell);
      local_dof_indices.resize(dofs_per_cell);

      local_rhs = 0;
      cell->get_dof_indices(local_dof_indices);

      std::vector<double> solution_values(n_q_points);
      fe_values.get_function_values(solution, solution_values);
      std::vector<Tensor<1, dim>> solution_gradients(n_q_points);
      fe_values.get_function_gradients(solution, solution_gradients);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        double grad_norm = solution_gradients[q].norm();
        double a_scalar = (1. - 1. / grad_norm);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          local_rhs(i) += a_scalar * fe_values.shape_grad(i, q) * solution_gradients[q] * fe_values.JxW(q);
        }
      }

      for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
      {

        if (cell->face(face_no)->at_boundary()) /* Exterior boundary */
        {
          // std::cout << "  Warning: Exterior boundaries" << std::endl;
          fe_values_face_hp.reinit(cell, face_no);
          const FEFaceValues<dim> &fe_values_face = fe_values_face_hp.get_present_fe_values();
          unsigned int n_face_q_points = fe_values_face.n_quadrature_points;

          std::vector<double> solution_values_face(n_face_q_points);
          fe_values_face.get_function_values(solution, solution_values_face);
          std::vector<Tensor<1, dim>> solution_gradients_face(n_face_q_points);
          fe_values_face.get_function_gradients(solution, solution_gradients_face);

          for (unsigned int q = 0; q < n_face_q_points; ++q)
          {
            double grad_norm_face = solution_gradients_face[q].norm();
            double a_scalar_face = (1. - 1. / grad_norm_face);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              // double neumann_boundary_value = 0;
              // local_rhs(i) -= a_scalar_face * neumann_boundary_value * fe_values_face.shape_value(i, q) *
              //                 fe_values_face.JxW(q);
            }
          }
        }
        // else if (cell->neighbor(face_no)->active_fe_index() == 0) /* Interior boundary */
        else if (cell->material_id() == 0 && cell->neighbor(face_no)->material_id() == 1)
        {
          fe_values_face_hp.reinit(cell, face_no);
          const FEFaceValues<dim> &fe_values_face = fe_values_face_hp.get_present_fe_values();
          unsigned int n_face_q_points = fe_values_face.n_quadrature_points;

          std::vector<Point<dim>> target_points(n_face_q_points);
          std::vector<Tensor<1, dim> > normal_vectors(n_face_q_points);
          std::vector<Tensor<1, dim> > distance_vectors(n_face_q_points);
          std::vector<double> boundary_values(n_face_q_points);
          sbm_map_manual(target_points, normal_vectors, distance_vectors, fe_values_face.get_quadrature_points(), n_face_q_points, dof_handler_all, solution_all);
          compute_boundary_values(velocity, target_points, normal_vectors, boundary_values, n_face_q_points);

          std::vector<double> solution_values_face(n_face_q_points);
          fe_values_face.get_function_values(solution, solution_values_face);
          std::vector<Tensor<1, dim>> solution_gradients_face(n_face_q_points);
          fe_values_face.get_function_gradients(solution, solution_gradients_face);

          for (unsigned int q = 0; q < n_face_q_points; ++q)
          {
            double grad_norm_face = solution_gradients_face[q].norm();
            Tensor<2, dim> A_tensor_face = (1. - 1. / grad_norm_face) * unit_tensor +
                                           outer_product(solution_gradients_face[q], solution_gradients_face[q]) /
                                           pow(grad_norm_face, 3.);
            double a_scalar_face = (1. - 1. / grad_norm_face);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              local_rhs(i) -= fe_values_face.normal_vector(q) * a_scalar_face *
                              solution_gradients_face[q] * fe_values_face.shape_value(i, q) *
                              fe_values_face.JxW(q);

              local_rhs(i) -= fe_values_face.normal_vector(q) * A_tensor_face * fe_values_face.shape_grad(i, q) *
                              (solution_values_face[q] + solution_gradients_face[q] * distance_vectors[q] - boundary_values[q]) *
                              fe_values_face.JxW(q);

              local_rhs(i) += alpha / h * (fe_values_face.shape_value(i, q) + fe_values_face.shape_grad(i, q) * distance_vectors[q]) *
                              (solution_values_face[q] + solution_gradients_face[q] * distance_vectors[q] - boundary_values[q]) *
                              fe_values_face.JxW(q);
            }
          }
        }
      }
      constraints.distribute_local_to_global(local_rhs, local_dof_indices, residual);
    }
  }

  return residual.l2_norm();

}


template <int dim>
void NonlinearProblem<dim>::assemble_system_initial_guess()
{

  system_matrix = 0;
  system_rhs    = 0;

  hp::FEValues<dim> fe_values_hp (fe_collection, q_collection,
                                  update_values    |  update_gradients |
                                  update_quadrature_points  |  update_JxW_values);


  hp::FEFaceValues<dim> fe_values_face_hp (fe_collection, q_collection_face,
      update_values    |  update_gradients |
      update_quadrature_points  |  update_JxW_values |
      update_normal_vectors);


  FullMatrix<double>   local_matrix;
  Vector<double>       local_rhs;
  std::vector<types::global_dof_index> local_dof_indices;

  typename hp::DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  // int cell_counter = 0;
  for (; cell != endc; ++cell)
  {
    if (cell->active_fe_index() == 1)
    {
      // std::cout << "  cell" << cell_counter++ << std::endl;

      fe_values_hp.reinit (cell);

      const FEValues<dim> &fe_values = fe_values_hp.get_present_fe_values();
      const unsigned int &dofs_per_cell = cell->get_fe().dofs_per_cell;
      const unsigned int &n_q_points    = fe_values.n_quadrature_points;

      local_matrix.reinit (dofs_per_cell, dofs_per_cell);
      local_rhs.reinit (dofs_per_cell);
      local_dof_indices.resize (dofs_per_cell);

      local_matrix = 0;
      local_rhs = 0;
      cell->get_dof_indices (local_dof_indices);

      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
      {
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            local_matrix(i, j) += (fe_values.shape_grad(i, q_index) *
                                   fe_values.shape_grad(j, q_index) *
                                   fe_values.JxW(q_index));
          }
          double source = 10.;
          local_rhs(i) += (fe_values.shape_value(i, q_index) * source * fe_values.JxW(q_index));
        }
      }

      for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
      {
        if (cell->face(face_no)->at_boundary()) /* Exterior boundary */
        {
          fe_values_face_hp.reinit(cell, face_no);
          const FEFaceValues<dim> &fe_values_face = fe_values_face_hp.get_present_fe_values();
          unsigned int n_face_q_points = fe_values_face.n_quadrature_points;

          for (unsigned int q = 0; q < n_face_q_points; ++q)
          {

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              double neumann_boundary_value = 1;
              local_rhs(i) += fe_values_face.shape_value(i, q) * neumann_boundary_value * fe_values_face.JxW(q);
            }
          }
        }
        // else if (cell->neighbor(face_no)->active_fe_index() == 0) /* Interior boundary */
        else if (cell->material_id() == 0 && cell->neighbor(face_no)->material_id() == 1)
        {

          fe_values_face_hp.reinit(cell, face_no);
          const FEFaceValues<dim> &fe_values_face = fe_values_face_hp.get_present_fe_values();
          unsigned int n_face_q_points = fe_values_face.n_quadrature_points;

          std::vector<Point<dim>> target_points(n_face_q_points);
          std::vector<Tensor<1, dim> > normal_vectors(n_face_q_points);
          std::vector<Tensor<1, dim> > distance_vectors(n_face_q_points);
          std::vector<double> boundary_values(n_face_q_points);
          sbm_map_manual(target_points, normal_vectors, distance_vectors, fe_values_face.get_quadrature_points(), n_face_q_points, dof_handler_all, solution_all);
          compute_boundary_values(velocity, target_points, normal_vectors, boundary_values, n_face_q_points);

          for (unsigned int q = 0; q < n_face_q_points; ++q)
          {

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {

                local_matrix(i, j) -= (fe_values_face.shape_value(i, q) +
                                       fe_values_face.shape_grad(i, q) * distance_vectors[q]) *
                                      fe_values_face.shape_grad(j, q) *
                                      fe_values_face.normal_vector(q) *
                                      fe_values_face.JxW(q);

                local_matrix(i, j) -= (fe_values_face.shape_value(j, q) +
                                       fe_values_face.shape_grad(j, q) * distance_vectors[q]) *
                                      fe_values_face.shape_grad(i, q) *
                                      fe_values_face.normal_vector(q) *
                                      fe_values_face.JxW(q);

                local_matrix(i, j) += fe_values_face.shape_grad(i, q) * distance_vectors[q] *
                                      fe_values_face.shape_grad(j, q) * fe_values_face.normal_vector(q) *
                                      fe_values_face.JxW(q);

                local_matrix(i, j) += alpha / h *
                                      (fe_values_face.shape_value(j, q) +
                                       fe_values_face.shape_grad(j, q) * distance_vectors[q]) *
                                      (fe_values_face.shape_value(i, q) +
                                       fe_values_face.shape_grad(i, q) * distance_vectors[q]) *
                                      fe_values_face.JxW(q);

              }

              local_rhs(i) -= fe_values_face.shape_grad(i, q) *
                              fe_values_face.normal_vector(q) *
                              boundary_values[q] * fe_values_face.JxW(q);

              local_rhs(i) += alpha / h *
                              (fe_values_face.shape_value(i, q) +
                               fe_values_face.shape_grad(i, q) * distance_vectors[q]) *
                              boundary_values[q] * fe_values_face.JxW(q);

            }
          }
        }

      }

      constraints.distribute_local_to_global (local_matrix,
                                              local_rhs,
                                              local_dof_indices,
                                              system_matrix,
                                              system_rhs);
    }

  }
}


template <int dim>
void NonlinearProblem<dim>::solve(double relaxation_parameter)
{
  std::cout << "Start to solve " << std::endl;

  SparseDirectUMFPACK  A_direct;
  A_direct.initialize(system_matrix);
  A_direct.vmult(newton_update, system_rhs);

  constraints.distribute(newton_update);

  solution.add(relaxation_parameter, newton_update);

  std::cout << "End of solve " << std::endl;
}


template <int dim>
void NonlinearProblem<dim>::output_results(unsigned int cycle)
{
  std::vector<std::string> solution_names;
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  data_component_interpretation;

  solution_names.push_back("u");
  data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

  DataOut<dim, hp::DoFHandler<dim>> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, solution_names,
                           DataOut<dim, hp::DoFHandler<dim>>::type_dof_data,
                           data_component_interpretation);
  data_out.build_patches();

  std::string filename = "solution-" + Utilities::int_to_string (cycle, 5) + ".vtk";
  std::ofstream output(filename.c_str());
  data_out.write_vtk(output);
}


template <int dim>
void NonlinearProblem<dim>::run()
{
  unsigned int newton_step = 0;
  double res = 0;
  bool first_step = true;
  while ( (first_step || (res > 1e-5)) && newton_step < 500)
  {
    std::cout << std::endl << "  Newton step " << newton_step << std::endl;
    if (first_step == true)
    {
      std::cout << "  Start to set up system" << std::endl;
      setup_system();
      std::cout << "  End of set up system" << std::endl;

      // std::cout << "  Start to assemble system" << std::endl;
      // assemble_system_initial_guess();
      // std::cout << "  End of assemble system" << std::endl;

      // std::cout << "  Start to solve..." << std::endl;
      // solve(1.);
      // std::cout << "  End of solve" << std::endl;

      // std::cout << "  Number of active cells: "
      //           << triangulation.n_active_cells()
      //           << std::endl;
      // std::cout << "  Number of degrees of freedom: "
      //           << dof_handler.n_dofs()
      //           << std::endl;
      // std::cout << "  Global Number of degrees of freedom: "
      //           << dof_handler_all.n_dofs()
      //           << std::endl;
      first_step = false;
      // output_results(newton_step);


      initialize_distance_field(dof_handler, solution, 0.4);

      // exit(0);

     }

    // std::cout << "  Start to assemble system" << std::endl;
    assemble_system();
    // std::cout << "  End of assemble system" << std::endl;

    // std::cout << "  Start to solve..." << std::endl;
    solve(0.2);
    // std::cout << "  End of solve" << std::endl;

    res = compute_residual();
    std::cout << "  Residual: " << res << std::endl;
    // std::cout << "  Delta phi norm: " << newton_update.l2_norm() << std::endl;

    output_results(newton_step);
    newton_step++;
  }
  exit(0);
}


#endif