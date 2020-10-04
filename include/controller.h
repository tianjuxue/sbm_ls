#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "problem.h"
#include "poisson_problem.h"

using namespace dealii;


void test_speed()
{
  std::vector<double> v(100000, 0);
  std::cout << "v has size " << v.size() << std::endl;
  for (int j = 0; j < 10000; ++j)
  {
    std::cout << "j = " << j << std::endl;
    for (int i = 0; i < v.size(); ++i)
    {
      v[i]++;
    }
  }
}

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

  // test_speed();
  // exit(0);


  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(6);

  int total_time_steps = 1000;
  // vortex
  // double dt = 10*1e-3;

  // moving square
  double dt = 2 * 1e-3;

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