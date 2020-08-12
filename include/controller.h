#ifndef CONTROLLER_H
#define CONTROLLER_H

using namespace dealii;


template <int dim>
bool is_inside(hp::DoFHandler<dim> &dof_handler_all, Vector<double> &solution_all_old, Point<dim> &point)
{
    return VectorTools::point_value(dof_handler_all, solution_all_old, point) > 0 ? true : false;
}


template <int dim>
void union_fields(hp::DoFHandler<dim> &dof_handler_all, Vector<double> &solution_all, Vector<double> &old_solution_all,
                            hp::DoFHandler<dim> &dof_handler_in, Vector<double> &solution_in,
                            hp::DoFHandler<dim> &dof_handler_out, Vector<double> &solution_out)
{
  hp::MappingCollection<dim> mapping_collection;
  mapping_collection.push_back(MappingQ1<dim>());
  std::vector<Point<dim>> support_points(dof_handler_all.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping_collection, dof_handler_all, support_points);

  for (int i = 0; i < dof_handler_all.n_dofs(); i++)
  {
      if (is_inside(dof_handler_all, old_solution_all, support_points[i]))
        solution_all(i) = VectorTools::point_value(dof_handler_in, solution_in, support_points[i]);
      else
        solution_all(i) = VectorTools::point_value(dof_handler_out, solution_out, support_points[i]);
   
  }
}

void main()
{
  Triangulation<dim>   triangulation;
  hp::DoFHandler<dim>  dof_handler;

  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(4);
  
}

#endif