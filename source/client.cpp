/*
//
//
//                       _oo0oo_
//                      o8888888o
//                      88" . "88      |-------------------------------------|
//                      (| -_- |)  --> | You shall have no bug in this code. |
//                      0\  =  /0      |-------------------------------------|
//                    ___/`---'\___
//                  .' \\|     |// '.
//                 / \\|||  :  |||// \
//                / _||||| -:- |||||- \
//               |   | \\\  -  /// |   |
//               | \_|  ''\---/''  |_/ |
//               \  .-\__  '-'  ___/-. /
//             ___'. .'  /--.--\  `. .'___
//          ."" '<  `.___\_<|>_/___.' >' "".
//         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
//         \  \ `_.   \_ __\ /__ _/   .-` /  /
//     =====`-.____`.___ \_____/___.-`___.-'=====
//                       `=---='
//
//
//     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//              Praying for no bug... lol
//     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/


#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

#include <deal.II/base/tensor_function.h>
#include <deal.II/base/geometric_utilities.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition_block.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_enriched.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/q_collection.h>

#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <iomanip>

#include "common_definitions.h"
#include "general_utils.h"
#include "error_estimate.h"
#include "problem.h"


int main ()
{
  bool debug_mode = false;
  bool run_mode = false;
  if (debug_mode)
  {
    if (run_mode)
    {
      NonlinearProblem<2> problem(PORE_CASE, NARROW_BAND, 6, MAP_NEWTON, 4);;
      problem.run();
    }
    else
    {
      NonlinearProblem<2> problem;
      problem.error_analysis();
    }
  }
  else
  {
    if (run_mode)
    {
      int total_pore_number = 9;
      int coarse_refinement_level = 8;
      int fine_refinement_level = 8;
      for (int i = 0; i < total_pore_number; ++i)
      {
        for (int j = coarse_refinement_level; j < fine_refinement_level + 1; ++j)
        {
          NonlinearProblem<2> problem_newton(PORE_CASE, NARROW_BAND, j, MAP_NEWTON, i);
          problem_newton.run();
          NonlinearProblem<2> problem_binary_search(PORE_CASE, NARROW_BAND, j, MAP_BINARY_SEARCH, i);
          problem_binary_search.run();
        }
      }
    }
    else
    {
      int total_pore_number = 9;
      int coarse_refinement_level = 5;
      int fine_refinement_level = 8;
      for (int i = 0; i < total_pore_number; ++i)
      {
        std::vector<double> error_list_newton;
        std::vector<double> error_list_bs;

        std::string newton_dat_filename = "../data/dat/pore_" + Utilities::int_to_string(i, 1) + "_newton.dat";
        std::string bs_dat_filename = "../data/dat/pore_" + Utilities::int_to_string(i, 1) + "_bs.dat";

        std::ofstream newton_dat_file;
        std::ofstream bs_dat_file;
        newton_dat_file.open(newton_dat_filename);
        bs_dat_file.open(bs_dat_filename);

        for (int j = coarse_refinement_level; j < fine_refinement_level + 1; ++j)
        {
          NonlinearProblem<2> problem_newton(PORE_CASE, NARROW_BAND, j, MAP_NEWTON, i);
          problem_newton.error_analysis();

          NonlinearProblem<2> problem_binary_search(PORE_CASE, NARROW_BAND, j, MAP_BINARY_SEARCH, i);
          problem_binary_search.error_analysis();

          newton_dat_file << problem_newton.h << " " << problem_newton.L2_error << " "
                          << problem_newton.SD_error << " " << problem_newton.interface_error << std::endl;
          bs_dat_file << problem_binary_search.h <<  " " << problem_binary_search.L2_error << " "
                      << problem_binary_search.SD_error << " " << problem_binary_search.interface_error << std::endl;
        }

        // std::cout << std::endl << "--------------------------------------------------------" << std::endl;
        // for (int j = 0; j < fine_refinement_level - coarse_refinement_level; ++j)
        // {
        //   std::cout << "pore number " << i << " newton ratio  is "
        //             << error_list_newton[j] / error_list_newton[j + 1] << std::endl;
        // }
        // for (int j = 0; j < fine_refinement_level - coarse_refinement_level + 1; ++j)
        // {
        //   std::cout << "pore number " << i << " newton error " <<  error_list_newton[j] << std::endl;
        // }
        // std::cout << std::endl;

        // for (int j = 0; j < fine_refinement_level - coarse_refinement_level; ++j)
        // {
        //   std::cout << "pore number " << i << " bs ratio  is "
        //             << error_list_bs[j] / error_list_bs[j + 1] << std::endl;
        // }
        // for (int j = 0; j < fine_refinement_level - coarse_refinement_level + 1; ++j)
        // {
        //   std::cout << "pore number " << i << " bs error " << error_list_bs[j] << std::endl;
        // }
        // std::cout << "--------------------------------------------------------" << std::endl;

        newton_dat_file.close();
        bs_dat_file.close();

      }
    }
  }
  return 0;
}