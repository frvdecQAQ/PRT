// Associated_Legendre_polynomials
#pragma once

#include "largefourierseries.hpp"
#include "shorder.hpp"

static const int maxdeg = 10*n;

typedef LargeFourierSeries<maxdeg> FS;

extern double fac(int n);
extern double Comb(float n, int k);
extern FS P_cosx(int l, int m);

typedef LargeFourierSeries<maxdeg> FS;