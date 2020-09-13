#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "problem.h"
#include "poisson_problem.h"

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
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(6);

  int total_time_steps = 1000;
  double dt = 2*1e-3;

  NonlinearProblem<2> problem(triangulation, velocity, dt);
  // PoissonProblem<2> problem(triangulation, velocity);

  for (int i = 1; i < total_time_steps; ++i)
  {
    problem.cycle_no = i;
    std::cout << std::endl << "  cycle: " << i << std::endl;
    if (i == 1)
      problem.run_picard(true);
    else
      problem.run_picard(false);
    problem.output_results(i);
  }

}


#endif