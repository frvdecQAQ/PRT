#ifndef BRDF_H_
#define BRDF_H_

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>

#define SAMPLE_NUMBER 64

// #define SHOW_BRDF

enum BRDF_TYPE
{
    BRDF_PHONG,
    BRDF_WARD_ISOTROPIC,
    BRDF_WARD_ANISOTROPIC
};

class BRDF
{
public:
    BRDF()
    {
        sampleNumber = SAMPLE_NUMBER;
    }

    void init(int band, BRDF_TYPE type);

    int band()
    {
        return _band;
    }

    void write2Diskbin(std::string filename);
    void readFDiskbin(std::string filename);

    int sampleNumber;
    std::vector<std::vector<float>>brdf_coef;

private:
    int _band;
};

#endif
