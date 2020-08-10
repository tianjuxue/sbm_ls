#ifndef GENERAL_UTILS_H
#define GENERAL_UTILS_H

using namespace dealii;


template <int dim>
void get_d(std::vector<Tensor<1, dim>> &d, const std::vector<Point<dim>> &points, int length)
{
  double radius = 0.5;
  for (int i = 0; i < length; ++i)
  {
    d[i] = points[i]/points[i].norm()*radius - points[i];
  }
}

// template<int dim>
void vec2num_values(std::vector<Vector<double> > &vec, 
                    std::vector<double >  &num, int length)
{

  int dim = 2;
  int temp = 2*dim;
  for (int i = 0; i < length; ++i)
  {
    num[i] = vec[i][temp];  
  }
}

// From vector<Tensor<1, dim>>, extract Tensor<2, dim>
template <int dim>
void vec2num_grads(std::vector< std::vector< Tensor<1, dim> >> &vec,
                   std::vector<Tensor<1, dim>> & ten, int length)
{
  int temp = 2*dim;
  for (int i = 0; i < length; ++i)
  {
    for (int j = 0; j < dim; ++j)
    {
       ten[i][j] = vec[i][temp][j];   
    }
  }
}

#endif