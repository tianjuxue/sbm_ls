#ifndef COMMON_DEFINITIONS_H
#define COMMON_DEFINITIONS_H

enum field
{
  PORE_CASE, TORUS_CASE, FEM_CASE, IMAGE_CASE
};

enum band
{
  NARROW_BAND, GLOBAL
};

enum map
{
  MAP_NEWTON, MAP_BINARY_SEARCH
};

enum domain
{
  FLAG_IN, FLAG_OUT, FLAG_IN_BAND, FLAG_OUT_BAND
};

enum solver
{
  DISTANCE_SOLVER, POISSON_BAND_SOLVER, DISTANCE_BAND_SOLVER
};

unsigned int CIRCLE_PORE = 4; // Circular pore has analytical solutions, so we specially highlight it


#endif
