#ifndef SPHERICALHARMONICS_H_
#define SPHERICALHARMONICS_H_

#define _USE_MATH_DEFINES
#include <math.h>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "utils.h"

namespace SphericalH
{
    // Value for Spherical Harmonic.
    double SHvalue(double theta, double phi, int l, int m);
    void SHvalueALL(int band, double theta, double phi, float* coef);
    void prepare(int band);
    /*void static testVisMap(int band, int n, const float* coef, const std::string store_path) {
        int band2 = band * band;
        cv::Mat gray(n, n, CV_32FC1);
        float* sh_value = new float[band2];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                float x = (float)(i) / n;
                float y = (float)(j) / n;
                float theta = acos(1 - 2 * x);
                float phi = 2.0f * M_PI * y;
                for (int k = 0; k < band2; ++k)sh_value[k] = 0;
                SHvalueALL(band, theta, phi, sh_value);
                float pixel_value = 0;
                for (int k = 0; k < band2; ++k)pixel_value += coef[k] * sh_value[k];
                gray.at<float>(i, j) = pixel_value * 255.0;
                //std::cout << "theta = " << theta << ' ' << "phi = " << phi << ' ' << "pixel_value = " << pixel_value << std::endl;
            }//std::cout << std::endl;
        }
        delete[] sh_value;
        cv::imwrite(store_path, gray);
    }*/
};

#endif
