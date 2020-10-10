#ifndef PROBLEM
#define PROBLEM

#include "general_utils.h"
#include "error_estimate.h"


template <int dim>
class NonlinearProblem
{
public:
  NonlinearProblem();
  ~NonlinearProblem();
  void run_picard();
  void output_results(unsigned int cycle);
  unsigned int cycle_no;


private:
  void cache_interface();
  void make_constraints();
  void setup_system();
  double compute_residual();
  void assemble_system_picard();
  void assemble_system_poisson();
  void assemble_system_projection();
  void solve_picard();

  Triangulation<dim> triangulation;
  hp::DoFHandler<dim> dof_handler;

  hp::FECollection<dim>   fe_collection;
  hp::QCollection<dim>    q_collection;
  hp::QCollection < dim - 1 >  q_collection_face;

  AffineConstraints<double> constraints;
  SparsityPattern           sparsity_pattern;
  SparseMatrix<double>      system_matrix;

  Vector<double>            newton_update;
  Vector<double>            system_rhs;

  Vector<double> solution;
  Vector<double> old_solution;

  std::vector<std::vector<Tensor<1, dim>>> cache_distance_vectors;
  std::vector<std::vector<double>> cache_boundary_values;
  std::vector<unsigned int> dof_flags;

  double h;
  double alpha;

  unsigned int FLAG_IN;
  unsigned int FLAG_OUT;
  unsigned int FLAG_IN_BAND;
  unsigned int FLAG_OUT_BAND;

  int solver_type;
  int DISTANCE_SOLVER;
  int POISSON_BAND_SOLVER;
  int DISTANCE_BAND_SOLVER;

  double c1;
  double c2;

  unsigned int refinement_level;

};


template <int dim>
NonlinearProblem<dim>::NonlinearProblem()
  :
  cycle_no(0),
  dof_handler(triangulation),
  alpha(1e1), // Magic number, may affect numerical instability
  FLAG_IN(0),
  FLAG_OUT(1),
  FLAG_IN_BAND(2),
  FLAG_OUT_BAND(3),
  solver_type(0),
  DISTANCE_SOLVER(0),
  POISSON_BAND_SOLVER(1),
  DISTANCE_BAND_SOLVER(2),
  c1(-0.),
  c2(0.),
  refinement_level(8)
{
  int fe_degree = 1;
  fe_collection.push_back(FE_Q<dim>(fe_degree));
  fe_collection.push_back(FE_Q<dim>(fe_degree));

  q_collection.push_back(QGauss<dim>(fe_degree + 1));
  q_collection.push_back(QGauss<dim>(fe_degree + 1));

  q_collection_face.push_back(QGauss < dim - 1 > (fe_degree + 1));
  q_collection_face.push_back(QGauss < dim - 1 > (fe_degree + 1));

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
        if (cell->material_id() == FLAG_IN_BAND && cell->neighbor(face_no)->material_id() == FLAG_OUT_BAND)
        {
          fe_values_face_hp.reinit(cell, face_no);
          const FEFaceValues<dim> &fe_values_face = fe_values_face_hp.get_present_fe_values();
          unsigned int n_face_q_points = fe_values_face.n_quadrature_points;

          std::vector<Point<dim>> target_points(n_face_q_points);
          std::vector<Tensor<1, dim> > normal_vectors(n_face_q_points);
          std::vector<Tensor<1, dim> > distance_vectors(n_face_q_points);
          std::vector<double> boundary_values(n_face_q_points);
          std::vector<Point<dim>> quadrature_points = fe_values_face.get_quadrature_points();

          for (unsigned int q = 0; q < n_face_q_points; ++q)
          {
            normal_vectors[q] = fe_values_face.normal_vector(q);
          }

          // sbm_map_newton(target_points, normal_vectors, distance_vectors, quadrature_points, n_face_q_points, dof_handler, old_solution);
          // sbm_map_binary_search(target_points, normal_vectors, distance_vectors, quadrature_points, n_face_q_points, dof_handler, old_solution, h);

          // sbm_map_newton(target_points, normal_vectors, distance_vectors, quadrature_points, n_face_q_points, c1, c2);
          sbm_map_binary_search(target_points, normal_vectors, distance_vectors, quadrature_points, n_face_q_points, c1, c2, h);

          for (unsigned int q = 0; q < n_face_q_points; ++q)
          {
            raw_file << std::fixed << std::setprecision(8) << quadrature_points[q][0] << " " << quadrature_points[q][1] << std::endl;
            map_file << std::fixed << std::setprecision(8) << target_points[q][0] << " " << target_points[q][1] << std::endl;
            bv_file << std::fixed << std::setprecision(8) << boundary_values[q] << std::endl;
          }
          cache_distance_vectors.push_back(distance_vectors);
          cache_boundary_values.push_back(boundary_values);

        }
      }
    }
  }
  map_file.close();
  raw_file.close();
  bv_file.close();
}



template <int dim>
void NonlinearProblem<dim>::make_constraints()
{
  dof_flags.clear();
  unsigned int tmp_flag = 100;
  dof_flags.insert(dof_flags.end(), dof_handler.n_dofs(), tmp_flag);

  if (solver_type == POISSON_BAND_SOLVER)
  {

    std::vector<types::global_dof_index> local_dof_indices;
    for (typename hp::DoFHandler<dim>::cell_iterator cell = dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
    {
      unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
      {
        if (cell->material_id() == FLAG_IN)
        {
          dof_flags[local_dof_indices[i]] = FLAG_IN;
        }
        else if ((cell->material_id() == FLAG_OUT))
        {
          dof_flags[local_dof_indices[i]] = FLAG_OUT;
        }
      }
    }

    constraints.clear();
    for (unsigned int i = 0; i < dof_flags.size(); ++i)
    {
      if (dof_flags[i] == FLAG_IN)
      {
        constraints.add_line(i);
        constraints.set_inhomogeneity(i, 1.);
      }
      else if (dof_flags[i] == FLAG_OUT)
      {
        constraints.add_line(i);
        constraints.set_inhomogeneity(i, -1);
      }
    }
    constraints.close();
  }
  else if (solver_type == DISTANCE_BAND_SOLVER)
  {
    std::vector<types::global_dof_index> local_dof_indices;
    for (typename hp::DoFHandler<dim>::cell_iterator cell = dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
    {
      unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
      {
        if (cell->material_id() == FLAG_IN_BAND)
        {
          dof_flags[local_dof_indices[i]] = FLAG_IN_BAND;
        }
        else if ((cell->material_id() == FLAG_OUT_BAND))
        {
          dof_flags[local_dof_indices[i]] = FLAG_OUT_BAND;
        }
      }
    }

    constraints.clear();
    for (unsigned int i = 0; i < dof_flags.size(); ++i)
    {
      if (dof_flags[i] == tmp_flag)
      {
        constraints.add_line(i);
        constraints.set_inhomogeneity(i, solution(i));
      }
    }
    constraints.close();
  }
  else if (solver_type == DISTANCE_SOLVER)
  {
    constraints.clear();
    // VectorTools::interpolate_boundary_values(dof_handler, 0, BoundaryValues<dim>(), constraints);
    constraints.close();
  }
}



template <int dim>
void NonlinearProblem<dim>::setup_system()
{

  GridGenerator::hyper_cube(triangulation, -2, 2);
  triangulation.refine_global(refinement_level);
  h = GridTools::minimal_cell_diameter(triangulation);

  std::cout << "  Mesh info: " << "h " << h << " square length " << 4 / pow(2, refinement_level) << std::endl;

  constraints.clear();
  constraints.close();

  dof_handler.distribute_dofs(fe_collection);

  if (cycle_no == 0)
  {
    std::cout << "  First cycle setup" << std::endl;
    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());

    initialize_distance_field_circle(dof_handler, solution, Point<dim>(0., 0.), 1);
    // initialize_distance_field_square(dof_handler, solution, Point<dim>(0.0, 0.0), 1.6);

    old_solution = solution;
    output_results(cycle_no);
  }

  newton_update.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());


  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);


  int counter_in = 0;
  for (typename hp::DoFHandler<dim>::cell_iterator cell = dof_handler.begin_active();
       cell != dof_handler.end(); ++cell)
  {
    bool is_surrogate_cell = true;
    for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
    {
      // is_surrogate_cell = is_surrogate_cell && old_solution(cell->vertex_dof_index(v, 0, 0)) > 0;
      is_surrogate_cell = is_surrogate_cell &&  pore_function_value(cell->vertex(v), c1, c2) > 0;
    }

    // if (old_solution(cell->vertex_dof_index(0, 0, 0)) > 0 &&
    //     old_solution(cell->vertex_dof_index(1, 0, 0)) > 0 &&
    //     old_solution(cell->vertex_dof_index(2, 0, 0)) > 0 &&
    //     old_solution(cell->vertex_dof_index(3, 0, 0)) > 0)
    if (is_surrogate_cell)
    {
      cell->set_material_id(FLAG_IN);
      counter_in++;
    }
    else
    {
      cell->set_material_id(FLAG_OUT);
    }
  }
  std::cout << "  Number of inner surrogate cells " << counter_in << std::endl;


  int counter_band = 0;
  unsigned int band_width = pow(2, refinement_level - 4);
  for (unsigned int i = 0; i < band_width; ++i)
  {
    for (typename hp::DoFHandler<dim>::cell_iterator cell = dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
    {
      cell->clear_user_flag();
      for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
      {
        if (!cell->face(face_no)->at_boundary())
        {
          if ( (cell->material_id() == FLAG_IN && cell->neighbor(face_no)->material_id() != FLAG_IN) ||
               (cell->material_id() == FLAG_OUT && cell->neighbor(face_no)->material_id() != FLAG_OUT) )
          {
            cell->set_user_flag();
          }
        }
      }
    }
    for (typename hp::DoFHandler<dim>::cell_iterator cell = dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
    {
      if (cell->user_flag_set())
      {
        counter_band++;
        if (cell->material_id() == FLAG_IN)
        {
          cell->set_material_id(FLAG_IN_BAND);
        }
        else if (cell->material_id() == FLAG_OUT)
        {
          cell->set_material_id(FLAG_OUT_BAND);
        }
      }
    }
  }

  std::cout << "  Number of band cells " << counter_band << std::endl;


}


template <int dim>
void NonlinearProblem<dim>::assemble_system_picard()
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

    if (cell->material_id() == FLAG_IN_BAND || cell->material_id() == FLAG_OUT_BAND || solver_type == DISTANCE_SOLVER)
    {
      std::vector<double> solution_values(n_q_points);
      fe_values.get_function_values(solution, solution_values);
      std::vector<Tensor<1, dim>> solution_gradients(n_q_points);
      fe_values.get_function_gradients(solution, solution_gradients);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        double grad_norm = solution_gradients[q].norm();
        Tensor<1, dim> part_d = solution_gradients[q] / grad_norm;

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            local_matrix(i, j) += fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) * fe_values.JxW(q);
          }
          local_rhs(i) += part_d * fe_values.shape_grad(i, q) * fe_values.JxW(q);
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
              double neumann_boundary_value = 0.;
              local_rhs(i) += neumann_boundary_value * fe_values_face.shape_value(i, q) *
                              fe_values_face.JxW(q);
            }
          }
        }
        else if (cell->material_id() == FLAG_IN_BAND && cell->neighbor(face_no)->material_id() == FLAG_OUT_BAND)
        {
          fe_values_face_hp.reinit(cell, face_no);
          const FEFaceValues<dim> &fe_values_face = fe_values_face_hp.get_present_fe_values();
          unsigned int n_face_q_points = fe_values_face.n_quadrature_points;

          std::vector<Tensor<1, dim> > distance_vectors = cache_distance_vectors[cache_index];
          std::vector<double> boundary_values = cache_boundary_values[cache_index];
          cache_index++;

          // std::vector<double> solution_values_face(n_face_q_points);
          // fe_values_face.get_function_values(solution, solution_values_face);
          // std::vector<Tensor<1, dim>> solution_gradients_face(n_face_q_points);
          // fe_values_face.get_function_gradients(solution, solution_gradients_face);

          for (unsigned int q = 0; q < n_face_q_points; ++q)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                local_matrix(i, j) += alpha / h * (fe_values_face.shape_value(i, q) + fe_values_face.shape_grad(i, q) * distance_vectors[q]) *
                                      (fe_values_face.shape_value(j, q) + fe_values_face.shape_grad(j, q) * distance_vectors[q]) *
                                      fe_values_face.JxW(q);
              }
              local_rhs(i) += alpha / h * (fe_values_face.shape_value(i, q) + fe_values_face.shape_grad(i, q) * distance_vectors[q]) *
                              boundary_values[q] * fe_values_face.JxW(q);
            }
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
void NonlinearProblem<dim>::assemble_system_poisson()
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

    if (cell->material_id() == FLAG_IN_BAND || cell->material_id() == FLAG_OUT_BAND)
    {
      std::vector<double> solution_values(n_q_points);
      fe_values.get_function_values(solution, solution_values);
      std::vector<Tensor<1, dim>> solution_gradients(n_q_points);
      fe_values.get_function_gradients(solution, solution_gradients);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            local_matrix(i, j) += fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) * fe_values.JxW(q);
          }
          local_rhs(i) += 0 * fe_values.shape_value(i, q) * fe_values.JxW(q);
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
void NonlinearProblem<dim>::assemble_system_projection()
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

    if (cell->material_id() == FLAG_IN_BAND || cell->material_id() == FLAG_OUT_BAND)
    {
      std::vector<double> solution_values(n_q_points);
      fe_values.get_function_values(solution, solution_values);
      std::vector<double> u_exact(n_q_points);
      exact_solution(u_exact, fe_values.get_quadrature_points(), n_q_points);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            local_matrix(i, j) += fe_values.shape_value(i, q) * fe_values.shape_value(j, q) * fe_values.JxW(q);
          }
          local_rhs(i) += u_exact[q] * fe_values.shape_value(i, q) * fe_values.JxW(q);
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
void NonlinearProblem<dim>::solve_picard()
{
  SparseDirectUMFPACK  A_direct;
  A_direct.initialize(system_matrix);
  A_direct.vmult(solution, system_rhs);
  constraints.distribute(solution);
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
void NonlinearProblem<dim>::run_picard()
{

  std::cout << "  Start to set up system" << std::endl;
  setup_system();
  cache_interface();
  std::cout << "  End of set up system" << std::endl;

  std::cout << "  Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl;

  std::cout << "  Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;


  solver_type = POISSON_BAND_SOLVER;
  make_constraints();
  std::cout << "  Start to assemble system FLAG_IN" << std::endl;
  assemble_system_poisson();
  std::cout << "  End of assemble system" << std::endl;

  std::cout << "  Start to solve..." << std::endl;
  solve_picard();
  std::cout << "  End of solve" << std::endl;
  output_results(1);


  solver_type = DISTANCE_BAND_SOLVER;
  make_constraints();
  std::cout << "  Number of lagrangian points: "
            << cache_boundary_values.size()
            << std::endl;

  unsigned int picard_step = 0;
  double res = 1e3;
  while (res > 1e-10 && picard_step < 1000)
  {
    std::cout << std::endl << "  Picard step " << picard_step << std::endl;

    std::cout << "  Start to assemble system" << std::endl;
    assemble_system_picard();
    std::cout << "  End of assemble system" << std::endl;

    std::cout << "  Start to solve..." << std::endl;
    solve_picard();
    std::cout << "  End of solve" << std::endl;

    Vector<double>  delta_solution = solution;
    delta_solution.sadd(-1, old_solution);
    res = delta_solution.l2_norm();
    std::cout << "  Delta phi norm: " << res  << std::endl;

    old_solution = solution;
    picard_step++;
    output_results(picard_step + 1);

    double L2_error = compute_error(dof_handler, solution, fe_collection, q_collection, FLAG_IN_BAND, FLAG_OUT_BAND);
    std::cout << "  L2 error is " << L2_error << std::endl;
  }

  // assemble_system_projection();
  // solve_picard();
  // double L2_error = compute_error(dof_handler, solution, fe_collection, q_collection, FLAG_IN_BAND, FLAG_OUT_BAND);
  // std::cout << "  L2 error for projection is " << L2_error << std::endl;
}


#endif
