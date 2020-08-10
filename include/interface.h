#ifndef INTERFACE
#define INTERFACE

using namespace dealii;


template<int dim>
class LevelSet
{
public:

  LevelSet();
  ~LevelSet();
  double level_set(const dealii::Point<dim> &p) const;

};



template <int dim>
LevelSet<dim>::LevelSet()
{
}

template <int dim>
LevelSet<dim>::~LevelSet()
{
}

// Definition of the level set function and its gradient
template <int dim>
double LevelSet<dim>::level_set (const Point<dim> &p) const
{
  return p.norm() - 0.5;
}



template <int dim>
class Geometry
{
private:

  std::vector<Point<dim> > intersection_points;

public:

  LevelSet<dim>  &ls;

  Geometry(LevelSet<dim>  &ls_)
    : 
    ls(ls_)
  {}


  bool is_surrogate(typename hp::DoFHandler<dim>::cell_iterator &cell);

};



template <int dim>
bool Geometry<dim>::is_surrogate (typename hp::DoFHandler<dim>::cell_iterator &cell) 
{

  double round_off_threshold = -1e-10;

  if (ls.level_set(cell->vertex(0)) < round_off_threshold &&
      ls.level_set(cell->vertex(1)) < round_off_threshold &&
      ls.level_set(cell->vertex(2)) < round_off_threshold &&
      ls.level_set(cell->vertex(3)) < round_off_threshold)
  {
    return true;
  }
  else
  {
    return false;
  }

}




#endif
