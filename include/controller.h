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

  GridGenerator::hyper_cube(triangulation, -2, 2);
  triangulation.refine_global(5);

  int total_time_steps = 1000;
  // vortex
  // double T = 1;
  // double dt = T / total_time_steps;
  // double dt = 10 * 1e-3;

  // moving square
  double dt = 2 * 1e-3;

  NonlinearProblem<2> problem(triangulation, velocity, dt);
  // PoissonProblem<2> problem(triangulation, velocity);

  for (int i = 0; i < total_time_steps; ++i)
  {
    problem.cycle_no = i;
    std::cout << std::endl << "  cycle: " << i << std::endl;
    problem.run_picard();
  }

}


#endif