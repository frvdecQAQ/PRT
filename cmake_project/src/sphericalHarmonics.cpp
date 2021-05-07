#include "sphericalHarmonics.h"

double *Kvalue_data;
int *minus_one_power;
double *double_factorial;
const double sqrt_2 = sqrt(2.0f);

double Kvalue(int l, int m)
{
    if (m == 0)
    {
        return sqrt((2 * l + 1) / (4 * M_PI));
    }
    double up = (2 * l + 1) * factorial(l - abs(m));
    double down = (4 * M_PI) * factorial(l + abs(m));
    return sqrt(up / down);
}

double evaluateLegendre(double x, int l, int m)
{
    double result = 0.0;
    if (l == m)
    {
        result = minus_one_power[m] * double_factorial[m] * pow((1 - x * x), m / 2.0);
    }
    else if (l == m + 1)
    {
        result = x * (2 * m + 1) * evaluateLegendre(x, m, m);
    }
    else
    {
        result = (x * (2 * l - 1) * evaluateLegendre(x, l - 1, m) - (l + m - 1) * evaluateLegendre(x, l - 2, m)) /
            (l - m);
    }
    return result;
}

namespace SphericalH
{
    void prepare(int band)
    {
        Kvalue_data = new double[band*band];
        for(int l = 0; l < band; ++l)
        {
            for(int m = -l; m <= l; ++m)
            {
                int index = l*(l+1)+m;
                Kvalue_data[index] = Kvalue(l, m);
            }
        }
        minus_one_power = new int[band];
        for(int i = 0; i < band; ++i)minus_one_power[i] = ((i&1)?-1:1);
        double_factorial = new double[band];
        for(int i = 0; i < band; ++i)double_factorial[i] = doubleFactorial(2*i-1);
    }

    double SHvalue(double theta, double phi, int l, int m)
    {
        double result = 0.0;
        if (m == 0)
        {
            result = Kvalue(l, 0) * evaluateLegendre(cos(theta), l, 0);
        }
        else if (m > 0)
        {
            result = sqrt(2.0f) * Kvalue(l, m) * cos(m * phi) * evaluateLegendre(cos(theta), l, m);
        }
        else
        {
            result = sqrt(2.0f) * Kvalue(l, m) * sin(-m * phi) * evaluateLegendre(cos(theta), l, -m);
        }
        if (fabs(result) <= M_ZERO)
            result = 0.0;
        if (__isnan(result))
        {
            std::cout << "SPHERICAL HARMONIC NAN" << std::endl;
            std::cout << "theta: " << theta << " " << "phi: " << phi << std::endl;
        }
        return result;
    }

    void SHvalueALL(int band, double theta, double phi, float* coef) 
    {
        double cos_theta = cos(theta);
        double pow_base = 1-cos_theta*cos_theta;
        coef[0] = Kvalue_data[0];
        if(band <= 1)return;
        //calculate Legendre
        int index;
        for(int l = 1; l < band; ++l)
        {
            index = l*(l+1);
            for(int m = 0; m < l-1; ++m)
            {
                coef[index] = (cos_theta*((l<<1)-1)*coef[index-(l<<1)]-(l+m-1)*coef[index-(l<<2)+2])/(l-m);
                ++index;
            }
            coef[index] = cos_theta * (((l-1)<<1) + 1) * coef[index-(l<<1)];
            ++index;
            coef[index] = minus_one_power[l] * double_factorial[l] * pow(pow_base, l / 2.0);
        }
        //calcuate SH base
        index = 0;
        for(int l = 1; l < band; ++l)
        {
            for(int m = -l; m < 0; ++m)
            {
                ++index;
                coef[index] = sqrt_2*Kvalue_data[index]*sin(-m*phi)*coef[index-(m<<1)];
            }
            ++index;
            coef[index] = Kvalue_data[index] * coef[index];
            for(int m = 1; m <= l; ++m)
            {
                ++index;
                coef[index] = sqrt_2*Kvalue_data[index]*cos(m*phi)*coef[index];
            }
        }
    }
}
