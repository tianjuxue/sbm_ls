#ifndef PROBLEM
#define PROBLEM

template <int dim>
class Problem
{
public:
  Problem();
  ~Problem();
  void run();
private:
  void setup_system();
  void assemble_system();
  void solve();
  void output_results(unsigned int cycle);
  Triangulation<dim>   triangulation;

  hp::DoFHandler<dim>     dof_handler;
  hp::FECollection<dim>   fe_collection;
  hp::QCollection<dim>    q_collection;
  hp::QCollection<dim-1>  q_collection_face;


  AffineConstraints<double> constraints;
  SparsityPattern           sparsity_pattern;
  SparseMatrix<double>      system_matrix;
  Vector<double>            solution;
  Vector<double>            system_rhs;

  LevelSet<dim> ls;
  Geometry<dim> geo;

  double h;
  double alpha;

};



template <int dim> 
Problem<dim>::Problem()
  :
  dof_handler(triangulation), 
  geo(ls),
  alpha(100)
{

  q_collection.push_back(QGauss<dim>(2));
  q_collection.push_back(QGauss<dim>(2));

  q_collection_face.push_back(QGauss<dim-1>(2));
  q_collection_face.push_back(QGauss<dim-1>(2));

  fe_collection.push_back(FE_Nothing<dim>());
  fe_collection.push_back(FE_Q<dim>(1));

}


template <int dim> 
Problem<dim>::~Problem()
{
  dof_handler.clear();
}


template <int dim>
void Problem<dim>::setup_system ()
{

  for (typename hp::DoFHandler<dim>::cell_iterator cell = dof_handler.begin_active();
       cell != dof_handler.end(); ++cell)
  {
    if (geo.is_surrogate(cell))
    {
      cell->set_active_fe_index(1);
    }
    else 
    {
      cell->set_active_fe_index(0);
    }
  }


  dof_handler.distribute_dofs (fe_collection);

  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());


  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler,
                                           constraints);
  constraints.close();


  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
}







template <int dim>
void Problem<dim>::assemble_system ()
{
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

  for (; cell!=endc; ++cell)
  {
    if (cell->active_fe_index() == 1)
    {

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


      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
      {
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<dofs_per_cell; ++j)
          {
            local_matrix(i,j) += (fe_values.shape_grad(i,q_index) *
                                  fe_values.shape_grad(j,q_index) *
                                  fe_values.JxW(q_index));
          }
          local_rhs(i) += (fe_values.shape_value(i,q_index) *
                          1.0 *
                          fe_values.JxW(q_index));
        }
      }


      for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no)
      {

        bool is_on_boundary = cell->neighbor(face_no)->active_fe_index() == 0;

        if (is_on_boundary)
        {
          // typename hp:DoFHandler<dim>::face_iterator face = cell->face(face_no);
          fe_values_face_hp.reinit (cell, face_no);
          const FEFaceValues<dim> &fe_values_face = fe_values_face_hp.get_present_fe_values();
          const unsigned int &n_face_q_points    = fe_values_face.n_quadrature_points;


          std::vector<Tensor<1, dim> > d(n_face_q_points);
          get_d(d, fe_values_face.get_quadrature_points(), n_face_q_points);

          for (unsigned int q=0; q<n_face_q_points; ++q)
          {

            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              for (unsigned int j=0; j<dofs_per_cell; ++j)
              {

                local_matrix(i, j) -= (fe_values_face.shape_value(i, q)+
                                       fe_values_face.shape_grad(i, q)*d[q])*
                                       fe_values_face.shape_grad(j, q)*
                                       fe_values_face.normal_vector(q)*
                                       fe_values_face.JxW(q);

                local_matrix(i, j) -= (fe_values_face.shape_value(j, q)+
                                       fe_values_face.shape_grad(j, q)*d[q])*
                                       fe_values_face.shape_grad(i, q)*
                                       fe_values_face.normal_vector(q)*
                                       fe_values_face.JxW(q);

                local_matrix(i, j) += fe_values_face.shape_grad(i, q)*d[q]*
                                      fe_values_face.shape_grad(j, q)*fe_values_face.normal_vector(q)*
                                      fe_values_face.JxW(q);

                local_matrix(i, j) += alpha/h*
                                      (fe_values_face.shape_value(j, q)+
                                      fe_values_face.shape_grad(j, q)*d[q])*
                                      (fe_values_face.shape_value(i, q)+
                                      fe_values_face.shape_grad(i, q)*d[q])*
                                      fe_values_face.JxW(q);
 
              }

              local_rhs(i) -= fe_values_face.shape_grad(i, q)*
                              fe_values_face.normal_vector(q)*
                              1.0*fe_values_face.JxW(q);

              local_rhs(i) += alpha/h*
                              (fe_values_face.shape_value(i, q)+
                               fe_values_face.shape_grad(i, q)*d[q])*   
                               1.0*fe_values_face.JxW(q);                         

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
void Problem<dim>::solve ()
{


  // std::ostringstream rhs_vector_filename;
  // rhs_vector_filename << "rhs_vector.out";
  // std::ofstream rhs_vector_output (rhs_vector_filename.str().c_str());
  // system_rhs.print(rhs_vector_output, 6, false, false);

  // std::ostringstream matrix_filename;
  // matrix_filename << "matrix.out";
  // std::ofstream matrix_output(matrix_filename.str().c_str());
  // system_matrix.print_formatted(matrix_output, 4, false);


  std::cout << "Start to solve " << std::endl;

  SparseDirectUMFPACK  A_direct;
  A_direct.initialize(system_matrix);
  A_direct.vmult (solution, system_rhs);

  constraints.distribute(solution);

  std::cout << "End of solve " << std::endl;

}


template <int dim>
void Problem<dim>::output_results (unsigned int cycle)
{
  std::vector<std::string> solution_names;
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation;

  solution_names.push_back("u");
  data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
  
  DataOut<dim, hp::DoFHandler<dim>> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, solution_names,
                            DataOut<dim, hp::DoFHandler<dim>>::type_dof_data,
                            data_component_interpretation);
  data_out.build_patches ();

  std::string filename = "solution-" + Utilities::int_to_string (cycle, 5) + ".vtk";
  std::ofstream output(filename.c_str());
  data_out.write_vtk(output);
}



template <int dim>
void Problem<dim>::run ()
{

  unsigned int cycle = 0; 
  std::cout << "Cycle " << cycle << ':' << std::endl;
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(5);
 
  h = GridTools::minimal_cell_diameter(triangulation);

  std::cout << "   Number of active cells:       "
            << triangulation.n_active_cells()
            << std::endl;
  setup_system();
  std::cout << "   Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;
  assemble_system();
  solve();
  output_results(cycle);

}


#endif