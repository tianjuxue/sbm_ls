#ifndef PROBLEM
#define PROBLEM

#include "general_utils.h"

template <int dim>
class NonlinearProblem
{
public:
  NonlinearProblem(Triangulation<dim> &triangulation_,
                   AdvectionVelocity<dim> &velocity_);
  ~NonlinearProblem();
  void run(bool first_cycle);
  void output_results(unsigned int cycle);
  hp::DoFHandler<dim> dof_handler;
  Vector<double> solution;
  Vector<double> old_solution;
  int cycle_no;

private:
  void cache_interface();
  void setup_system(bool first_cycle);
  void assemble_system();
  double compute_residual();
  void solve(double relaxation_parameter);


  hp::FECollection<dim>   fe_collection;
  hp::QCollection<dim>    q_collection;
  hp::QCollection < dim - 1 >  q_collection_face;

  AffineConstraints<double> constraints;
  SparsityPattern           sparsity_pattern;
  SparseMatrix<double>      system_matrix;

  Vector<double>            newton_update;
  Vector<double>            system_rhs;

  Triangulation<dim> &triangulation;
  AdvectionVelocity<dim> &velocity;

  std::vector<std::vector<Tensor<1, dim>>> cache_distance_vectors;
  std::vector<std::vector<double>> cache_boundary_values;

  double h;
  double alpha;

};


template <int dim>
NonlinearProblem<dim>::NonlinearProblem(Triangulation<dim> &triangulation_,
                                        AdvectionVelocity<dim> &velocity_)
  :
  cycle_no(0),
  dof_handler(triangulation_),
  triangulation(triangulation_),
  velocity(velocity_),
  h(GridTools::minimal_cell_diameter(triangulation_)),
  alpha(10.)// Magic number, pick a number you like
{

  q_collection.push_back(QGauss<dim>(3));
  q_collection.push_back(QGauss<dim>(3));

  q_collection_face.push_back(QGauss < dim - 1 > (2));
  q_collection_face.push_back(QGauss < dim - 1 > (2));

  fe_collection.push_back(FE_Q<dim>(1));
  fe_collection.push_back(FE_Q<dim>(1));
}


template <int dim>
NonlinearProblem<dim>::~NonlinearProblem()
{
  dof_handler.clear();
}


template <int dim>
void NonlinearProblem<dim>::cache_interface()
{

  cache_distance_vectors.clear();
  cache_boundary_values.clear();


  std::ofstream raw_file("../data/raw_points" + Utilities::int_to_string(cycle_no, 3) + ".txt");
  std::ofstream map_file("../data/map_points" + Utilities::int_to_string(cycle_no, 3) + ".txt");
  std::ofstream bv_file("../data/bv" + Utilities::int_to_string(cycle_no, 3) + ".txt");

  hp::FEFaceValues<dim> fe_values_face_hp (fe_collection, q_collection_face,
      update_values    |  update_gradients |
      update_quadrature_points  |  update_JxW_values |
      update_normal_vectors);

  typename hp::DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  for (; cell != endc; ++cell)
  {
    for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
    {
      if (!cell->face(face_no)->at_boundary())
      {
        if (cell->material_id() == 0 && cell->neighbor(face_no)->material_id() == 1)
        {
          fe_values_face_hp.reinit(cell, face_no);
          const FEFaceValues<dim> &fe_values_face = fe_values_face_hp.get_present_fe_values();
          unsigned int n_face_q_points = fe_values_face.n_quadrature_points;

          std::vector<Point<dim>> target_points(n_face_q_points);
          std::vector<Tensor<1, dim> > normal_vectors(n_face_q_points);
          std::vector<Tensor<1, dim> > distance_vectors(n_face_q_points);
          std::vector<double> boundary_values(n_face_q_points);
          std::vector<Point<dim>> quadrature_points = fe_values_face.get_quadrature_points();
          sbm_map(target_points, normal_vectors, distance_vectors, quadrature_points, n_face_q_points, dof_handler, old_solution);
          compute_boundary_values(velocity, target_points, normal_vectors, boundary_values, n_face_q_points);
          cache_distance_vectors.push_back(distance_vectors);
          cache_boundary_values.push_back(boundary_values);

          for (unsigned int i = 0; i < n_face_q_points; ++i)
          {
            raw_file << std::fixed << std::setprecision(8) << quadrature_points[i][0] << " " << quadrature_points[i][1] << std::endl;
            map_file << std::fixed << std::setprecision(8) << target_points[i][0] << " " << target_points[i][1] << std::endl;
            bv_file << std::fixed << std::setprecision(8) << boundary_values[i] << std::endl;
          }

        }
      }
    }
  }
  map_file.close();
  raw_file.close();
  bv_file.close();
}


template <int dim>
void NonlinearProblem<dim>::setup_system(bool first_cycle)
{

  dof_handler.distribute_dofs(fe_collection);

  if (first_cycle)
  {
    std::cout << "  First cycle setup" << std::endl;
    solution.reinit(dof_handler.n_dofs());
    initialize_distance_field(dof_handler, solution, 0.5);
    old_solution = solution;
    output_results(0);
  }

  newton_update.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  int counter = 0;
  for (typename hp::DoFHandler<dim>::cell_iterator cell = dof_handler.begin_active();
       cell != dof_handler.end(); ++cell)
  {
    // cell->set_active_fe_index(1);
    if (is_inside(dof_handler, old_solution, cell->center()))
    {
      cell->set_material_id(0);
      counter++;
    }
    else
      cell->set_material_id(1);

  }
  std::cout << "  cell counter " << counter << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler,
                                          constraints);
  constraints.close();

  cache_interface();
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
  unsigned int cache_index = 0;

  for (; cell != endc; ++cell)
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

        for (unsigned int q = 0; q < n_face_q_points; ++q)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            double neumann_boundary_value = 0;
            local_rhs(i) += neumann_boundary_value * fe_values_face.shape_value(i, q) *
                            fe_values_face.JxW(q);
          }
        }
      }
      else if (cell->material_id() == 0 && cell->neighbor(face_no)->material_id() == 1)
      {
        fe_values_face_hp.reinit(cell, face_no);
        const FEFaceValues<dim> &fe_values_face = fe_values_face_hp.get_present_fe_values();
        unsigned int n_face_q_points = fe_values_face.n_quadrature_points;

        std::vector<Tensor<1, dim> > distance_vectors = cache_distance_vectors[cache_index];
        std::vector<double> boundary_values = cache_boundary_values[cache_index];
        cache_index++;

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

              // local_matrix(i, j) -= fe_values_face.normal_vector(q) * A_tensor_face * fe_values_face.shape_grad(i, q) *
              //                       (fe_values_face.shape_value(j, q) + fe_values_face.shape_grad(j, q) * distance_vectors[q]) *
              //                       fe_values_face.JxW(q);

              local_matrix(i, j) += alpha / h * (fe_values_face.shape_value(i, q) + fe_values_face.shape_grad(i, q) * distance_vectors[q]) *
                                    (fe_values_face.shape_value(j, q) + fe_values_face.shape_grad(j, q) * distance_vectors[q]) *
                                    fe_values_face.JxW(q);
            }
            local_rhs(i) += fe_values_face.normal_vector(q) * a_scalar_face *
                            solution_gradients_face[q] * fe_values_face.shape_value(i, q) *
                            fe_values_face.JxW(q);

            // local_rhs(i) += fe_values_face.normal_vector(q) * A_tensor_face * fe_values_face.shape_grad(i, q) *
            //                 (solution_values_face[q] + solution_gradients_face[q] * distance_vectors[q] - boundary_values[q]) *
            //                 fe_values_face.JxW(q);

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

  // SymmetricTensor<2, dim> unit_tensor = unit_symmetric_tensor<dim>();
  unsigned int cache_index = 0;

  for (; cell != endc; ++cell)
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
        fe_values_face_hp.reinit(cell, face_no);
        const FEFaceValues<dim> &fe_values_face = fe_values_face_hp.get_present_fe_values();
        unsigned int n_face_q_points = fe_values_face.n_quadrature_points;

        for (unsigned int q = 0; q < n_face_q_points; ++q)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            double neumann_boundary_value = 0;
            local_rhs(i) -= neumann_boundary_value * fe_values_face.shape_value(i, q) *
                            fe_values_face.JxW(q);
          }
        }
      }
      else if (cell->material_id() == 0 && cell->neighbor(face_no)->material_id() == 1)
      {
        fe_values_face_hp.reinit(cell, face_no);
        const FEFaceValues<dim> &fe_values_face = fe_values_face_hp.get_present_fe_values();
        unsigned int n_face_q_points = fe_values_face.n_quadrature_points;

        std::vector<Tensor<1, dim> > distance_vectors = cache_distance_vectors[cache_index];
        std::vector<double> boundary_values = cache_boundary_values[cache_index];
        cache_index++;

        std::vector<double> solution_values_face(n_face_q_points);
        fe_values_face.get_function_values(solution, solution_values_face);
        std::vector<Tensor<1, dim>> solution_gradients_face(n_face_q_points);
        fe_values_face.get_function_gradients(solution, solution_gradients_face);

        for (unsigned int q = 0; q < n_face_q_points; ++q)
        {
          double grad_norm_face = solution_gradients_face[q].norm();
          // Tensor<2, dim> A_tensor_face = (1. - 1. / grad_norm_face) * unit_tensor +
          //                                outer_product(solution_gradients_face[q], solution_gradients_face[q]) /
          //                                pow(grad_norm_face, 3.);
          double a_scalar_face = (1. - 1. / grad_norm_face);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            local_rhs(i) -= fe_values_face.normal_vector(q) * a_scalar_face *
                            solution_gradients_face[q] * fe_values_face.shape_value(i, q) *
                            fe_values_face.JxW(q);

            // local_rhs(i) -= fe_values_face.normal_vector(q) * A_tensor_face * fe_values_face.shape_grad(i, q) *
            //                 (solution_values_face[q] + solution_gradients_face[q] * distance_vectors[q] - boundary_values[q]) *
            //                 fe_values_face.JxW(q);

            local_rhs(i) += alpha / h * (fe_values_face.shape_value(i, q) + fe_values_face.shape_grad(i, q) * distance_vectors[q]) *
                            (solution_values_face[q] + solution_gradients_face[q] * distance_vectors[q] - boundary_values[q]) *
                            fe_values_face.JxW(q);
          }
        }
      }
    }
    constraints.distribute_local_to_global(local_rhs, local_dof_indices, residual);
  }

  return residual.l2_norm();
}




template <int dim>
void NonlinearProblem<dim>::solve(double relaxation_parameter)
{
  SparseDirectUMFPACK  A_direct;
  A_direct.initialize(system_matrix);
  A_direct.vmult(newton_update, system_rhs);

  constraints.distribute(newton_update);

  solution.add(relaxation_parameter, newton_update);
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

  std::string filename = "solution-" + Utilities::int_to_string(cycle, 5) + ".vtk";
  std::ofstream output(filename.c_str());
  data_out.write_vtk(output);
}


template <int dim>
void NonlinearProblem<dim>::run(bool first_cycle)
{
  unsigned int newton_step = 0;
  double res = 0;
  bool first_step = true;
  while ( (first_step || (res > 1e-8)) && newton_step < 500)
  {
    std::cout << std::endl << "  Newton step " << newton_step << std::endl;
    if (first_step)
    {
      std::cout << "  Start to set up system" << std::endl;
      setup_system(first_cycle);
      std::cout << "  End of set up system" << std::endl;

      std::cout << "  Number of active cells: "
                << triangulation.n_active_cells()
                << std::endl;
      std::cout << "  Number of degrees of freedom: "
                << dof_handler.n_dofs()
                << std::endl;

      first_step = false;

      // output_results(newton_step);
      // exit(0);

    }

    std::cout << "  Start to assemble system" << std::endl;
    assemble_system();
    std::cout << "  End of assemble system" << std::endl;

    std::cout << "  Start to solve..." << std::endl;
    solve(0.6);
    std::cout << "  End of solve" << std::endl;

    res = compute_residual();
    std::cout << "  Residual: " << res << std::endl;
    // std::cout << "  Delta phi norm: " << newton_update.l2_norm() << std::endl;

    // output_results(newton_step);
    newton_step++;
  }

  old_solution = solution;

}


#endif