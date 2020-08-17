#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "problem.h"

using namespace dealii;


template <int dim>
class TimeEvolution
{
public:
  void run();
private:
  Triangulation<dim>  triangulation;
  AdvectionVelocity<dim> velocity;
};


template <int dim>
void TimeEvolution<dim>::run()
{
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(6);

  int time_steps = 30;

  NonlinearProblem<2> nonlinear_problem(triangulation, velocity);

  for (int i = 1; i < time_steps; ++i)
  {
    nonlinear_problem.cycle_no = i;
    std::cout << std::endl << "  cycle: " << i << std::endl;
    if (i == 1)
      nonlinear_problem.run(true);
    else
      nonlinear_problem.run(false);
    nonlinear_problem.output_results(i);
  }

}



#endif