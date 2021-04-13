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
#include <deal.II/base/mpi.h>

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
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_precondition.h>

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
      unsigned int refinement_level = 5;
      unsigned int refinement_increment = 3;
      unsigned int band_width = 4;
      // NonlinearProblem<2> problem(PEANUT_CASE, GLOBAL, refinement_level, refinement_increment, band_width,
      //                             MAP_NEWTON, 0);
      NonlinearProblem<2> problem(PEANUT_CASE, NARROW_BAND, refinement_level, refinement_increment, band_width,
                                  MAP_NEWTON, 0, POISSON_CONSTRAINT);
      // NonlinearProblem<2> problem(PORE_CASE, NARROW_BAND, refinement_level, refinement_increment,
      //                             band_width, MAP_NEWTON, 4, POISSON_CONSTRAINT);
      // NonlinearProblem<2> problem(PORE_CASE, GLOBAL, refinement_level, refinement_increment, band_width,
      //                             MAP_NEWTON, 4);
      // NonlinearProblem<3> problem(TORUS_CASE, NARROW_BAND, refinement_level, refinement_increment, band_width, MAP_NEWTON);
      problem.run();
    }
    else
    {
      unsigned int refinement_level = 5;
      unsigned int refinement_increment = 1;
      unsigned int band_width = 4;

      // narrow_band_1_refinement_level_8_0_map_choice_1_pore_1_laplace_0
      // NonlinearProblem<2> problem(PORE_CASE, GLOBAL, refinement_level, refinement_increment, band_width,
      //                             MAP_NEWTON, 4);
      // NonlinearProblem<2> problem(PORE_CASE, NARROW_BAND, refinement_level, refinement_increment,
      //                             band_width, MAP_NEWTON, 4, POISSON_CONSTRAINT);
      NonlinearProblem<2> problem(PEANUT_CASE, NARROW_BAND, refinement_level, refinement_increment, band_width,
                                  MAP_NEWTON, 0, POISSON_CONSTRAINT);
      problem.error_analysis();
    }
  }
  else
  {
    if (run_mode)
    {
      unsigned int total_pore_number;
      unsigned int refinement_level;
      unsigned int refinement_increment;
      unsigned int band_width;
      for (int case_flag = 2; case_flag < 4 ; case_flag++)
      {
        if (case_flag == PORE_CASE)
        {
          total_pore_number = 9;
          refinement_level = 5;
          refinement_increment = 4;
          band_width = 4;
          for (unsigned int i = 0; i < total_pore_number; ++i)
          {
            if (i == 0 || i == 4)
            {
              for (unsigned int j = 0; j < refinement_increment; ++j)
              {
                NonlinearProblem<2> problem_global_trivial(case_flag, GLOBAL, refinement_level, j, 0, MAP_NEWTON, i);
                problem_global_trivial.run();
                NonlinearProblem<2> problem_narrow_trivial(case_flag, NARROW_BAND, refinement_level, j, band_width, MAP_NEWTON, i);
                problem_narrow_trivial.run();
                NonlinearProblem<2> problem_narrow_laplace(case_flag, NARROW_BAND, refinement_level, j, band_width, MAP_NEWTON, i, POISSON_CONSTRAINT);
                problem_narrow_laplace.run();
              }
            }
          }
        }
        else if (case_flag == PEANUT_CASE)
        {
          // NOT IMPLEMENTED WARNING!
          // We have used two different initial level set functions: the "quadratic" and the "sin" multipliers.
          // To switch between these two, we have to do it manually! To be changed...
          refinement_level = 5;
          refinement_increment = 4;
          band_width = 4;
          for (unsigned int j = 0; j < refinement_increment; ++j)
          {
            NonlinearProblem<2> problem(PEANUT_CASE, NARROW_BAND, refinement_level, j, band_width, MAP_NEWTON, 0, POISSON_CONSTRAINT);
            problem.run();
          }
        }
        else if (case_flag == SPHERE_CASE || case_flag == TORUS_CASE)
        {
          refinement_level = 5;
          refinement_increment = 3;
          band_width = 1;
          for (unsigned int j = 0; j < refinement_increment; ++j)
          {
            NonlinearProblem<3> problem_newton(case_flag, NARROW_BAND, refinement_level, j, band_width, MAP_NEWTON);
            problem_newton.run();
          }
        }
      }
    }
    else
    {
      unsigned int total_pore_number;
      unsigned int refinement_level;
      unsigned int refinement_increment;
      unsigned int band_width;
      for (int case_flag = 2; case_flag < 4 ; case_flag++)
      {
        if (case_flag == PORE_CASE)
        {
          total_pore_number = 9;
          refinement_level = 5;
          refinement_increment = 4;
          band_width = 4;
          for (unsigned int i = 0; i < total_pore_number; ++i)
          {
            if (i == 0 || i == 4)
            {
              std::string global_trivial_dat_filename = "../data/dat/convergence_second_submission/case_" +  Utilities::int_to_string(case_flag, 1) +
                  "/pore_" + Utilities::int_to_string(i, 1) + "_global_trivial.dat";

              std::string narrow_trivial_dat_filename = "../data/dat/convergence_second_submission/case_" + Utilities::int_to_string(case_flag, 1) +
                  "/pore_" + Utilities::int_to_string(i, 1) + "_narrow_trivial.dat";

              std::string narrow_laplace_dat_filename = "../data/dat/convergence_second_submission/case_" + Utilities::int_to_string(case_flag, 1) +
                  "/pore_" + Utilities::int_to_string(i, 1) + "_narrow_laplace.dat";

              std::ofstream global_trivial_dat_file;
              std::ofstream narrow_trivial_dat_file;
              std::ofstream narrow_laplace_dat_file;
              global_trivial_dat_file.open(global_trivial_dat_filename);
              narrow_trivial_dat_file.open(narrow_trivial_dat_filename);
              narrow_laplace_dat_file.open(narrow_laplace_dat_filename);

              if (global_trivial_dat_file.is_open() && narrow_trivial_dat_file.is_open() && narrow_laplace_dat_file.is_open())
              {
                for (unsigned int j = 0; j < refinement_increment; ++j)
                {
                  NonlinearProblem<2> problem_global_trivial(case_flag, GLOBAL, refinement_level, j, 0, MAP_NEWTON, i);
                  problem_global_trivial.error_analysis();
                  NonlinearProblem<2> problem_narrow_trivial(case_flag, NARROW_BAND, refinement_level, j, band_width, MAP_NEWTON, i);
                  problem_narrow_trivial.error_analysis();
                  NonlinearProblem<2> problem_narrow_laplace(case_flag, NARROW_BAND, refinement_level, j, band_width, MAP_NEWTON, i, POISSON_CONSTRAINT);
                  problem_narrow_laplace.error_analysis();

                  global_trivial_dat_file << problem_global_trivial.h << " " << problem_global_trivial.L2_error << " "
                                          << problem_global_trivial.H1_error << " " << problem_global_trivial.L_infty_error << " "
                                          << problem_global_trivial.SD_error << " " << problem_global_trivial.interface_error_parametric << " "
                                          << problem_global_trivial.volume_error
                                          << std::endl;

                  narrow_trivial_dat_file << problem_narrow_trivial.h << " " << problem_narrow_trivial.L2_error << " "
                                          << problem_narrow_trivial.H1_error << " " << problem_narrow_trivial.L_infty_error << " "
                                          << problem_narrow_trivial.SD_error << " " << problem_narrow_trivial.interface_error_parametric << " "
                                          << problem_narrow_trivial.volume_error
                                          << std::endl;

                  narrow_laplace_dat_file << problem_narrow_laplace.h << " " << problem_narrow_laplace.L2_error << " "
                                          << problem_narrow_laplace.H1_error << " " << problem_narrow_laplace.L_infty_error << " "
                                          << problem_narrow_laplace.SD_error << " " << problem_narrow_laplace.interface_error_parametric << " "
                                          << problem_narrow_laplace.volume_error
                                          << std::endl;
                }
                global_trivial_dat_file.close();
                narrow_trivial_dat_file.close();
                narrow_laplace_dat_file.close();
              }
              else
              {
                assert(0 && "File not open!");
              }
            }
          }
        }
        else if (case_flag == SPHERE_CASE || case_flag == TORUS_CASE)
        {
          refinement_level = 5;
          refinement_increment = 3;
          band_width = 1;

          std::string newton_dat_filename = "../data/dat/convergence_second_submission/case_" +  Utilities::int_to_string(case_flag, 1) +
                                            "/narrow_trivial.dat";
          std::ofstream newton_dat_file;
          newton_dat_file.open(newton_dat_filename);
          if (newton_dat_file.is_open())
          {
            for (unsigned int j = 0; j < refinement_increment; ++j)
            {
              NonlinearProblem<3> problem_newton(case_flag, NARROW_BAND, refinement_level, j, band_width, MAP_NEWTON);
              problem_newton.error_analysis();
              newton_dat_file << problem_newton.h << " " << problem_newton.L2_error << " "
                              << problem_newton.H1_error << " " << problem_newton.L_infty_error << " "
                              << problem_newton.SD_error << " " << problem_newton.interface_error_qw << " "
                              << problem_newton.volume_error << " "
                              << std::endl;
            }
            newton_dat_file.close();
          }
          else
          {
            assert(0 && "File not open!");
          }
        }
      }
    }
  }
  return 0;
}