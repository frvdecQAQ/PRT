#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <random>
#include "sampler.h"
#include "sphericalHarmonics.h"

Sampler::Sampler() {}

Sampler::Sampler(unsigned nn, bool random_offset)
{
    static std::default_random_engine e;
    static std::uniform_real_distribution<float> u(0, 1);

    for (unsigned i = 0; i < nn; i++)
    {
        for (unsigned j = 0; j < nn; j++)
        {
            glm::vec2 spherical;
            glm::vec3 cartesian;
            float x, y;
            if (random_offset) {
                x = (float)(i + u(e)) / float(nn);
                y = (float)(j + u(e)) / float(nn);
            }
            else {
                x = float(i) / float(nn);
                y = float(j) / float(nn);
            }

            // theta
            spherical[0] = acos(1 - 2*x);
            // phi
            spherical[1] = 2.0f * M_PI * y;
            // x
            cartesian.x = sin(spherical[0]) * cos(spherical[1]);
            // y
            cartesian.y = sin(spherical[0]) * sin(spherical[1]);
            // z
            cartesian.z = cos(spherical[0]);

            _samples.emplace_back(Sample(cartesian, spherical));
        }
    }
}

void Sampler::computeSH(int band)
{
    std::cout << band << std::endl;
    int band2 = band * band;
    size_t size = _samples.size();

    for (int i = 0; i < size; i++)
    {
        _samples[i]._SHvalue = new float[band2];
        std::cout << i << std::endl;
        SphericalH::SHvalueALL(band, _samples[i]._sphericalCoord[0], _samples[i]._sphericalCoord[1],
                            _samples[i]._SHvalue);
    }
}
