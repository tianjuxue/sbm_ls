#ifndef PROBLEM
#define PROBLEM


template <int dim>
class NonlinearProblem
{
public:
  NonlinearProblem(unsigned int case_flag_ = PORE_CASE,
                   unsigned int domain_flag_ = NARROW_BAND,
                   unsigned int refinement_level_ = 5,
                   unsigned int refinement_increment_ = 1,
                   unsigned int band_width_ = 1,
                   unsigned int map_choice_ = MAP_NEWTON,
                   int pore_number_ = CIRCLE_PORE,
                   unsigned int constraint_type_ = TRIVIAL_CONSTRAINT);
  ~NonlinearProblem();
  void run();
  void error_analysis();
  void output_cycle_vtk(unsigned int cycle);
  void output_vtk(std::string &filename);
  void output_binary();
  double h;
  double L_infty_error;
  double L2_error;
  double H1_error;
  double SD_error;
  double interface_error_parametric;
  double interface_error_qw;
  double volume_error;

private:
  void cache_interface();
  void make_constraints();
  void narrow_band_helper(hp::DoFHandler<dim> &dof_handler, int level);
  void refine_to_same_level(hp::DoFHandler<dim> &dof_handler, int coarse_level, int fine_level);
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

  Vector<double> system_rhs;
  Vector<double> solution;
  Vector<double> old_solution;

  std::vector<std::vector<Tensor<1, dim>>> cache_distance_vectors;
  std::vector<std::vector<double>> cache_boundary_values;
  std::vector<unsigned int> dof_flags;

  double alpha;
  unsigned int constraint_type;
  double c1;
  double c2;
  unsigned int domain_flag;
  unsigned int case_flag;
  int pore_number;
  unsigned int map_choice;
  unsigned int refinement_level;
  unsigned int refinement_increment;
  unsigned int band_width;

  std::string vector_filename;

};


template <int dim>
NonlinearProblem<dim>::NonlinearProblem(unsigned int case_flag_,
                                        unsigned int domain_flag_,
                                        unsigned int refinement_level_,
                                        unsigned int refinement_increment_,
                                        unsigned int band_width_ ,
                                        unsigned int map_choice_,
                                        int pore_number_,
                                        unsigned int constraint_type_)
  :
  L_infty_error(0.),
  L2_error(0.),
  H1_error(0.),
  SD_error(0.),
  interface_error_parametric(0.),
  interface_error_qw(0.),
  volume_error(0.),
  dof_handler(triangulation),
  alpha(1e1), // Magic number, may affect numerical instability, 1e1 for 2D
  constraint_type(constraint_type_),
  c1(0.),
  c2(0.),
  domain_flag(domain_flag_),
  case_flag(case_flag_),
  pore_number(pore_number_),
  map_choice(map_choice_),
  refinement_level(refinement_level_),
  refinement_increment(refinement_increment_),
  band_width(band_width_)
{
  int fe_degree = 1;

  if (domain_flag == NARROW_BAND)
  {
    fe_collection.push_back(FE_Q<dim>(fe_degree));
    fe_collection.push_back(FE_Nothing<dim>());
  }
  else
  {
    fe_collection.push_back(FE_Q<dim>(fe_degree));
    fe_collection.push_back(FE_Q<dim>(fe_degree));
  }

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

  int cycle_no = 0;
  std::ofstream raw_file("../data/text/raw_points" + Utilities::int_to_string(cycle_no, 3) + ".txt");
  std::ofstream map_file("../data/text/map_points" + Utilities::int_to_string(cycle_no, 3) + ".txt");
  std::ofstream bv_file("../data/text/bv" + Utilities::int_to_string(cycle_no, 3) + ".txt");

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
        if ((cell->material_id() == FLAG_IN && cell->neighbor(face_no)->material_id() == FLAG_MID_BAND) ||
            (cell->material_id() == FLAG_IN_BAND && cell->neighbor(face_no)->material_id() == FLAG_MID_BAND))
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
            normal_vectors[q] = fe_values_face.normal_vector(q);

          std::function<double(const Point<dim> &)> function_value;
          std::function<Tensor<1, dim>(const Point<dim> &)> function_gradient;

          if (case_flag == PORE_CASE)
          {
            // Like functools.partial in Python
            function_value = std::bind(pore_function_value<dim>, std::placeholders::_1, c1, c2);
            function_gradient = std::bind(pore_function_gradient<dim>, std::placeholders::_1, c1, c2);
          }
          else if (case_flag == STAR_CASE)
          {
            function_value = star_function_value<dim>;
            function_gradient = star_function_gradient<dim>;
          }
          else if (case_flag == SPHERE_CASE)
          {
            function_value = sphere_function_value<dim>;
            function_gradient = sphere_function_gradient<dim>;
          }
          else if (case_flag == TORUS_CASE)
          {
            function_value = torus_function_value<dim>;
            function_gradient = torus_function_gradient<dim>;
          }
          else if (case_flag == PEANUT_CASE)
          {
            function_value = peanut_function_value<dim>;
            function_gradient = peanut_function_gradient<dim>;
          }
          else if (case_flag == FEM_CASE)
          {
            // Using FEFieldFunction can be faster than functions defined under VectorTools (e.g., point_value)
            // See https://www.dealii.org/current/doxygen/deal.II/namespaceVectorTools.html#acd358e9b110ccbf4a7f76796d206b9c7
            // phi = VectorTools::point_value(dof_handler, solution, target_point);
            // grad_phi = VectorTools::point_gradient(dof_handler, solution, target_point);
            Functions::FEFieldFunction<dim, hp::DoFHandler<dim>, Vector<double>> fe_field_function(dof_handler, solution);

            // The following will throw an error: "invalid use of non-static member function"
            // function_value = fe_field_function.value;
            // function_gradient = fe_field_function.gradient;

            // C++ lambda functions
            function_value = [&fe_field_function](const Point<dim> &point) {return fe_field_function.value(point);};
            function_gradient = [&fe_field_function](const Point<dim> &point) {return fe_field_function.gradient(point);};
          }
          else
          {
            assert(0 && "Other cases not implemented!");
          }

          if (map_choice == MAP_NEWTON)
            sbm_map_newton(target_points, normal_vectors, distance_vectors, quadrature_points, n_face_q_points, function_value, function_gradient);
          else
            sbm_map_binary_search(target_points, normal_vectors, distance_vectors, quadrature_points, n_face_q_points, h, function_value);

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
  assert( !(domain_flag == GLOBAL && constraint_type == POISSON_CONSTRAINT) &&
          "Error: when using laplace smoothing with POISSON_CONSTRAINT, domain_flag can't be set to GLOBAL!");
  dof_flags.clear();
  unsigned int tmp_flag = 100; // An arbitrary number
  dof_flags.insert(dof_flags.end(), dof_handler.n_dofs(), tmp_flag);

  if (constraint_type == POISSON_CONSTRAINT)
  {
    std::vector<types::global_dof_index> local_dof_indices;
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->active_fe_index() == 0)
      {
        for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
        {
          if (!cell->face(face_no)->at_boundary())
          {
            if (cell->neighbor(face_no)->active_fe_index() == 1)
            {
              unsigned int dofs_per_face = cell->get_fe().dofs_per_face;
              local_dof_indices.resize(dofs_per_face);
              // Second argument: const unsigned int fe_index = DoFHandlerType::default_fe_index
              cell->face(face_no)->get_dof_indices(local_dof_indices, 0);
              for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
              {
                if (cell->neighbor(face_no)->material_id() == FLAG_IN)
                  dof_flags[local_dof_indices[i]] = FLAG_IN;
                else if (cell->neighbor(face_no)->material_id() == FLAG_OUT)
                  dof_flags[local_dof_indices[i]] = FLAG_OUT;
                else
                  assert(0 && "Error: neighbor cell should only be either FLAG_IN or FLAG_OUT");
              }
            }
          }
        }
      }
    }
    constraints.clear();
    for (unsigned int i = 0; i < dof_flags.size(); ++i)
    {
      if (dof_flags[i] == FLAG_IN)
      {
        constraints.add_line(i);
        constraints.set_inhomogeneity(i, -1.);
      }
      else if (dof_flags[i] == FLAG_OUT)
      {
        constraints.add_line(i);
        constraints.set_inhomogeneity(i, 1);
      }
    }
    constraints.close();
  }
  else
  {
    constraints.clear();
    // VectorTools::interpolate_boundary_values(dof_handler, 0, BoundaryValues<dim>(), constraints);
    constraints.close();
  }
}


template <int dim>
void NonlinearProblem<dim>::narrow_band_helper(hp::DoFHandler<dim> &dof_handler, int level)
{
  int counter_band = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (cell->level() == level)
    {
      bool positive_flag = false;
      bool negative_flag = false;
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
      {
        double level_set_value;
        if (case_flag == PORE_CASE)
          level_set_value = pore_function_value(cell->vertex(v), c1, c2);
        else if (case_flag == STAR_CASE)
          level_set_value = star_function_value(cell->vertex(v));
        else if (case_flag == SPHERE_CASE)
          level_set_value = sphere_function_value(cell->vertex(v));
        else if (case_flag == TORUS_CASE)
          level_set_value = torus_function_value(cell->vertex(v));
        else if (case_flag == PEANUT_CASE)
          level_set_value = peanut_function_value(cell->vertex(v));
        else if (case_flag == FEM_CASE)
        {
          // To be implemented...
          // level_set_value = solution(cell->vertex_dof_index(v, 0, 0));
        }
        else
          assert(0 && "Other cases not implemented!");
        if (level_set_value >= 0)
          positive_flag = true;
        else
          negative_flag = true;
      }

      if (positive_flag == true && negative_flag == true)
      {
        cell->set_material_id(FLAG_MID_BAND);
        cell->set_active_fe_index(0);
        cell->set_refine_flag();
        counter_band++;
      }
      else if (positive_flag == true && negative_flag == false)
      {
        cell->set_material_id(FLAG_OUT);
        cell->set_active_fe_index(1);
      }
      else if (positive_flag == false && negative_flag == true)
      {
        cell->set_material_id(FLAG_IN);
        cell->set_active_fe_index(1);
      }
      else
        assert(0 && "Error!");
    }
    else
    {
      // A special note (Tianju): The following ensures that the coarser cells must not be active.
      // However, this line of code should be unncessary, logically.
      // In 2D cases, it works well without this line of code.
      // In 3D cases, it fails wihout it.
      // I suspect there are some internal bugs in dealii function execute_coarsening_and_refinement
      // where in 3D cases it changes my active_fe_index flag, which is terrible.
      cell->set_active_fe_index(1);
    }
  }

  std::cout << "   Number of middle_band cells " << counter_band << std::endl;

  for (unsigned int i = 0; i < band_width; ++i)
  {
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->level() == level)
      {
        cell->clear_user_flag();
        for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
        {
          if (!cell->face(face_no)->at_boundary() && cell->level() == cell->neighbor_level(face_no))
          {
            if ( (cell->material_id() == FLAG_IN && cell->neighbor(face_no)->material_id() != FLAG_IN) ||
                 (cell->material_id() == FLAG_OUT && cell->neighbor(face_no)->material_id() != FLAG_OUT) )
            {
              cell->set_user_flag();
            }
          }
        }
      }
    }
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->level() == level)
      {
        if (cell->user_flag_set())
        {
          if (cell->material_id() == FLAG_IN)
          {
            cell->set_material_id(FLAG_IN_BAND);
            cell->set_active_fe_index(0);
            cell->set_refine_flag();
          }
          else if (cell->material_id() == FLAG_OUT)
          {
            cell->set_material_id(FLAG_OUT_BAND);
            cell->set_active_fe_index(0);
            cell->set_refine_flag();
          }
          counter_band++;
        }
      }
    }
  }
  std::cout << "   Number of band cells " << counter_band << std::endl;

  // Perform some tests
  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (cell->level() != level && cell->active_fe_index() == 0)
      assert(0 && "Cell at coarser level, but active!");
    if (cell->level() == level)
    {
      if (cell->refine_flag_set() && cell->active_fe_index() == 1)
        assert (0 && "Cell refine flag set but not active");

      if (!cell->refine_flag_set() && cell->active_fe_index() == 0)
        assert(0 && "Cell refine flag not set but active");
    }
  }

}


// The purpose of doing this instead of just using a simple global refinement is to
// keep record of narrow band region in the context of global refinement.
// So this will enable us to answer the review's question about fair comparison
// between narrow band and global (using the same volume for integral when computing error)
template <int dim>
void NonlinearProblem<dim>::refine_to_same_level(hp::DoFHandler<dim> &dof_handler, int coarse_level, int fine_level)
{
  for (const auto &cell : dof_handler.active_cell_iterators())
    cell->clear_refine_flag();

  for (int level = coarse_level; level < fine_level; level++)
  {
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->level() == level)
        cell->set_refine_flag();
    }
    triangulation.execute_coarsening_and_refinement();
  }

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (cell->material_id() == FLAG_IN_BAND || cell->material_id() == FLAG_OUT_BAND || cell->material_id() == FLAG_MID_BAND)
      assert(cell->active_fe_index() == 0 && "narrow_band cells should have active_fe_index to be 0");
    else
      cell->set_active_fe_index(1);
  }
}


template <int dim>
void NonlinearProblem<dim>::setup_system()
{
  if (case_flag == PORE_CASE)
  {
    // pore_number can't be of "unsigned int" type, otherwise causing overflow
    c1 = ((pore_number / 3) - 1) * 0.2;
    c2 = ((pore_number % 3) - 1) * 0.2;
    std::cout << "  Pore inf: c1 = " << c1 << ", c2 = " << c2 << std::endl;
    vector_filename = "../data/vector/case_" + Utilities::int_to_string(case_flag, 1) +
                      "/narrow_band_" + Utilities::int_to_string(domain_flag, 1) +
                      "_refinement_level_" + Utilities::int_to_string(refinement_level, 1) +
                      "_" + Utilities::int_to_string(refinement_increment, 1) +
                      "_map_choice_" + Utilities::int_to_string(map_choice, 1) +
                      "_pore_" + Utilities::int_to_string(pore_number, 1) +
                      "_laplace_" + Utilities::int_to_string(constraint_type, 1);
  }
  else
  {
    vector_filename = "../data/vector/case_" + Utilities::int_to_string(case_flag, 1) +
                      "/narrow_band_" + Utilities::int_to_string(domain_flag, 1) +
                      "_refinement_level_" + Utilities::int_to_string(refinement_level, 1) +
                      "_" + Utilities::int_to_string(refinement_increment, 1) +
                      "_map_choice_" + Utilities::int_to_string(map_choice, 1) +
                      "_laplace_" + Utilities::int_to_string(constraint_type, 1);
  }

  std::cout << "  Strat grid" << std::endl;

  GridGenerator::hyper_cube(triangulation, -DOMAIN_SIZE, DOMAIN_SIZE);

  triangulation.refine_global(refinement_level);
  narrow_band_helper(dof_handler, refinement_level);
  for (unsigned int i = 0; i < refinement_increment; ++i)
  {
    triangulation.execute_coarsening_and_refinement();
    narrow_band_helper(dof_handler, refinement_level + i + 1);
  }

  if (domain_flag == GLOBAL)
    refine_to_same_level(dof_handler, refinement_level, refinement_level + refinement_increment);

  dof_handler.distribute_dofs(fe_collection);

  solution.reinit(dof_handler.n_dofs());
  old_solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);

  h = GridTools::minimal_cell_diameter(triangulation);
  std::cout << "  General info: " << vector_filename << std::endl;
  std::cout << "  Mesh info: " << "h " << h << ", which should be "
            << 4 / pow(2, refinement_level + refinement_increment) * sqrt(dim) << std::endl;

  std::cout << "  End grid" << std::endl;

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

    if (cell->active_fe_index() == 0 || domain_flag == GLOBAL)
    {
      std::vector<double> solution_values(n_q_points);
      fe_values.get_function_values(solution, solution_values);
      std::vector<Tensor<1, dim>> solution_gradients(n_q_points);
      fe_values.get_function_gradients(solution, solution_gradients);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        double grad_norm = solution_gradients[q].norm();
        // Tensor<1, dim> part_d =  grad_norm > 1e-3 ? solution_gradients[q] / grad_norm : solution_gradients[q] * (2 - grad_norm);
        Tensor<1, dim> part_d =  grad_norm > 1 ? solution_gradients[q] / grad_norm : solution_gradients[q] * (2 - grad_norm);
        // Tensor<1, dim> part_d = solution_gradients[q] / grad_norm;

        // std::cout << std::endl;
        // std::cout << "material_id " << cell->material_id() << std::endl;
        // std::cout << "solution_values[q] = " << solution_values[q] << std::endl;
        // std::cout << "solution_gradients[q] = " << solution_gradients[q] << std::endl;
        // std::cout << "quad point is " << fe_values.get_quadrature_points()[q] << std::endl;

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
        else if ((cell->material_id() == FLAG_IN && cell->neighbor(face_no)->material_id() == FLAG_MID_BAND) ||
                 (cell->material_id() == FLAG_IN_BAND && cell->neighbor(face_no)->material_id() == FLAG_MID_BAND))
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

    if (cell->active_fe_index() == 0)
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

    if (cell->active_fe_index() == 0 || domain_flag == GLOBAL)
    {
      std::vector<double> function_values(n_q_points);
      if (case_flag == PORE_CASE)
      {
        pore_function(function_values, fe_values.get_quadrature_points(), n_q_points, c1, c2);
      }
      else if (case_flag == STAR_CASE)
      {
        star_function(function_values, fe_values.get_quadrature_points(), n_q_points);
      }
      else if (case_flag == SPHERE_CASE)
      {
        sphere_function(function_values, fe_values.get_quadrature_points(), n_q_points);
      }
      else if (case_flag == TORUS_CASE)
      {
        torus_function(function_values, fe_values.get_quadrature_points(), n_q_points);
      }
      else if (case_flag == PEANUT_CASE)
      {
        peanut_function(function_values, fe_values.get_quadrature_points(), n_q_points);
      }
      else
      {
        assert(0 && "Other cases not implemented yet!");
      }

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            local_matrix(i, j) += fe_values.shape_value(i, q) * fe_values.shape_value(j, q) * fe_values.JxW(q);
          }
          local_rhs(i) += function_values[q] * fe_values.shape_value(i, q) * fe_values.JxW(q);
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

  // SolverControl            solver_control(1000, 1e-12);
  // SolverCG<Vector<double>> solver(solver_control);
  // PreconditionSSOR<SparseMatrix<double>> preconditioner;
  // preconditioner.initialize(system_matrix);
  // solver.solve(system_matrix, solution, system_rhs, preconditioner);
  // constraints.distribute(solution);
}


template <int dim>
void NonlinearProblem<dim>::output_vtk(std::string & filename)
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

  std::ofstream output(filename);
  data_out.write_vtk(output);
}


template <int dim>
void NonlinearProblem<dim>::output_cycle_vtk(unsigned int cycle)
{
  std::string filename = "solution-" + Utilities::int_to_string(cycle, 5) + ".vtk";
  output_vtk(filename);
}


template <int dim>
void NonlinearProblem<dim>::output_binary()
{
  std::ofstream output_solution_file(vector_filename);
  solution.block_write(output_solution_file);
  output_solution_file.close();
}


template <int dim>
void NonlinearProblem<dim>::error_analysis()
{
  // std::cout <<  std::endl <<  std::endl << "############################################################" << std::endl;

  std::cout << "  Start to set up system" << std::endl;
  setup_system();
  std::cout << "  End of set up system" << std::endl;

  std::ifstream input_solution_file(vector_filename);
  solution.block_read(input_solution_file);
  input_solution_file.close();

  std::cout << "  Start computing SD_error..." << std::endl;
  SD_error = compute_SD_error(dof_handler, solution, fe_collection, q_collection, domain_flag);
  std::cout << "  Finish computing SD_error: " << SD_error << std::endl;

  // std::cout << "  Start computing volume error..." << std::endl;
  // volume_error = compute_volume_error(dof_handler, solution, fe_collection, domain_flag, case_flag);
  // std::cout << "  Finish computing volume error: " << volume_error << std::endl;

  std::string quads_dat_filename = "../data/dat/surface_integral/sbi_case_" + Utilities::int_to_string(case_flag, 1) + "_quads.dat";
  std::string weights_dat_filename = "../data/dat/surface_integral/sbi_case_" + Utilities::int_to_string(case_flag, 1) + "_weights.dat";

  if (case_flag == PORE_CASE)
  {
    volume_error = compute_volume_error(dof_handler, solution, fe_collection, domain_flag, case_flag);
    interface_error_parametric = compute_interface_error_pore(dof_handler, solution, c1, c2);
    if (pore_number == CIRCLE_PORE)
    {
      L2_error = compute_L2_error(dof_handler, solution, fe_collection, q_collection, domain_flag, case_flag);
      L_infty_error = compute_Linfty_error(dof_handler, solution, fe_collection, domain_flag, case_flag);
      H1_error = compute_H1_error(dof_handler, solution, fe_collection, q_collection, domain_flag, case_flag);
      std::cout << "  L_infty_error " << L_infty_error << std::endl;
    }
  }
  else if (case_flag == STAR_CASE)
  {
    // TODO: something like
    // interface_error_parametric = compute_interface_error_pore(dof_handler, solution);
  }
  else if (case_flag == SPHERE_CASE)
  {
    volume_error = compute_volume_error(dof_handler, solution, fe_collection, domain_flag, case_flag);
    interface_error_parametric = compute_interface_error_sphere(dof_handler, solution);
    interface_error_qw = compute_interface_error_qw(dof_handler, solution, quads_dat_filename, weights_dat_filename);
    L2_error = compute_L2_error(dof_handler, solution, fe_collection, q_collection, domain_flag, case_flag);
    L_infty_error = compute_Linfty_error(dof_handler, solution, fe_collection, domain_flag, case_flag);
    H1_error = compute_H1_error(dof_handler, solution, fe_collection, q_collection, domain_flag, case_flag);
  }
  else if (case_flag == TORUS_CASE)
  {
    interface_error_qw = compute_interface_error_qw(dof_handler, solution, quads_dat_filename, weights_dat_filename);
  }
  else if (case_flag == PEANUT_CASE)
  {
    L_infty_error = compute_Linfty_error(dof_handler, solution, fe_collection, domain_flag, case_flag);
    std::cout << "  L_infty_error " << L_infty_error << std::endl;
    interface_error_parametric = compute_interface_error_peanut(dof_handler, solution);
    std::cout << "  surface error " << interface_error_parametric << std::endl;
  }
  else
    assert(0 && "Other case not implemented!");

  std::string vtk_filename = "../data/vtk/case_" + Utilities::int_to_string(case_flag, 1) +
                             "/narrow_band_" + Utilities::int_to_string(domain_flag, 1) +
                             "_refinement_level_" + Utilities::int_to_string(refinement_level, 1) +
                             "_" + Utilities::int_to_string(refinement_increment, 1) +
                             "_map_choice_" + Utilities::int_to_string(map_choice, 1) +
                             "_pore_" + Utilities::int_to_string(pore_number, 1) +
                             "_laplace_" + Utilities::int_to_string(constraint_type, 1) + ".vtk";
  output_vtk(vtk_filename);
}


template <int dim>
void NonlinearProblem<dim>::run()
{

  std::cout <<  std::endl <<  std::endl << "############################################################" << std::endl;

  std::cout << "  Start to set up system" << std::endl;
  setup_system();
  std::cout << "  End of set up system" << std::endl;

  std::cout << "  Start to cache interface" << std::endl;
  cache_interface();
  std::cout << "  Number of lagrangian points: "
            << cache_boundary_values.size()
            << std::endl;

  std::cout << "  Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl;

  std::cout << "  Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

  make_constraints();
  if (domain_flag == NARROW_BAND && constraint_type == POISSON_CONSTRAINT)
  {
    std::cout << "  Start to assemble system for poisson" << std::endl;
    assemble_system_poisson();
    std::cout << "  End of assemble system" << std::endl;
  }
  else
  {
    std::cout << "  Start to assemble system for projection" << std::endl;
    assemble_system_projection();
    std::cout << "  End of assemble system" << std::endl;
  }
  std::cout << "  Start to solve..." << std::endl;
  solve_picard();
  std::cout << "  End of solve" << std::endl;

  unsigned int picard_step = 0;
  output_cycle_vtk(picard_step);
  if (case_flag == PEANUT_CASE)
    write_narrow_band_solutions(dof_handler, solution, fe_collection, case_flag, refinement_level + refinement_increment, picard_step);

  constraint_type = TRIVIAL_CONSTRAINT;
  make_constraints();

  double res = 1e3;
  while (res > 1e-8 && picard_step < 1000)
  {
    std::cout << std::endl << "  Picard step " << picard_step << std::endl;

    // std::cout << "  Start to assemble system" << std::endl;
    assemble_system_picard();
    // std::cout << "  End of assemble system" << std::endl;

    // std::cout << "  Start to solve..." << std::endl;
    solve_picard();
    // std::cout << "  End of solve" << std::endl;

    Vector<double>  delta_solution = solution;
    delta_solution.sadd(-1, old_solution);
    res = delta_solution.l2_norm();
    std::cout << "  Delta phi norm: " << res  << std::endl;

    old_solution = solution;
    picard_step++;

    // output_cycle_vtk(picard_step);
    output_cycle_vtk(1);

    if (case_flag == PEANUT_CASE)
    {
      res = 1e3; // reset res=1e3 because we always want 1000 steps in the PEANUT_CASE
      write_narrow_band_solutions(dof_handler, solution, fe_collection, case_flag, refinement_level + refinement_increment, picard_step);
    }

    // if (picard_step % 10 == 0)
    // {

    //   L_infty_error = compute_Linfty_error(dof_handler, solution, fe_collection, domain_flag, case_flag);
    //   std::cout << "  L_infty_error " << L_infty_error << std::endl;

    //   interface_error_parametric = compute_interface_error_peanut(dof_handler, solution);
    //   // interface_error_parametric = compute_interface_error_pore(dof_handler, solution, c1, c2);
    //   std::cout << "  surface error " << interface_error_parametric << std::endl;

    // }

    // double L2_error = compute_l2_error(dof_handler, solution, fe_collection, q_collection, domain_flag);
    // std::cout << "  L2 error is " << L2_error << std::endl;
  }

  // double interface_error = compute_interface_error_pore(dof_handler, solution, c1, c2);
  // std::cout << "  interface error is " << interface_error << std::endl;
  output_binary();
}


#endif
