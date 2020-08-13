#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "problem.h"

using namespace dealii;


template <int dim>
class TimeEvolution
{
public:
  TimeEvolution();
  ~TimeEvolution();
  void run();
private:

  Triangulation<dim>  triangulation;
  hp::DoFHandler<dim> dof_handler_all;
  hp::FECollection<dim> fe_collection;
  Vector<double> solution_all;
  AdvectionVelocity<dim> velocity;
};


template <int dim>
TimeEvolution<dim>::TimeEvolution()
  :
  dof_handler_all(triangulation)
{
  fe_collection.push_back(FE_Q<dim>(1));
}


template <int dim>
TimeEvolution<dim>::~TimeEvolution()
{
  dof_handler_all.clear();
}


template <int dim>
void TimeEvolution<dim>::run()
{
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(6);

  int time_steps = 1;

  dof_handler_all.distribute_dofs(fe_collection);
  solution_all.reinit(dof_handler_all.n_dofs());

  initialize_distance_field(dof_handler_all, solution_all);

  for (int i = 0; i < time_steps; ++i)
  {
    bool in_out_flag = true;
    NonlinearProblem<2> nonlinear_problem_in(triangulation, dof_handler_all, solution_all, velocity, in_out_flag);
    nonlinear_problem_in.run();

    in_out_flag = false;
    NonlinearProblem<2> nonlinear_problem_out(triangulation, dof_handler_all, solution_all, velocity, in_out_flag);
    nonlinear_problem_out.run();

    std::cout << "  Number of degrees of freedom (in): "
              << nonlinear_problem_in.dof_handler.n_dofs()
              << std::endl;
    std::cout << "  Number of degrees of freedom (out): "
              << nonlinear_problem_out.dof_handler.n_dofs()
              << std::endl;
    std::cout << "  Number of degrees of freedom (total): "
              << dof_handler_all.n_dofs()
              << std::endl;

    union_distance_fields(dof_handler_all, solution_all,
                          nonlinear_problem_in.dof_handler, nonlinear_problem_in.solution,
                          nonlinear_problem_out.dof_handler, nonlinear_problem_out.solution);
  }


}



#endif