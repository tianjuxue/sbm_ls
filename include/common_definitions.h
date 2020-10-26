#ifndef COMMON_DEFINITIONS_H
#define COMMON_DEFINITIONS_H

enum field
{
  PORE_CASE, STAR_CASE, SPHERE_CASE, TORUS_CASE, FEM_CASE, IMAGE_CASE
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
  FLAG_IN, FLAG_OUT, FLAG_IN_BAND, FLAG_OUT_BAND, FLAG_MID_BAND
};

enum constraint
{
  TRIVIAL_CONSTRAINT, POISSON_CONSTRAINT
};

int CIRCLE_PORE = 4; // Circular pore has analytical solutions, so we specially highlight it
double DOMAIN_SIZE = 2;
double RADIUS = 1.;

#endif
