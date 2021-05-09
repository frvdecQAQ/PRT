#ifndef GENERALOBJECT_H_
#define GENERALOBJECT_H_

#include <vector>
#include <Eigen/Dense>
#include <glm/glm.hpp>
#include "bvhTree.h"
#include "object.h"
#include "sampler.h"
#include "shorder.hpp"

class GeneralObject : public Object
{
public:
    GeneralObject():Object()
    {
        _difforGeneral = true;
        brdf_sampler = Sampler(brdf_sample_num, false);
    }

    void project2SH(int mode, int band, int sampleNumber, int bounce, 
        std::vector<Object*>obj_list, int scene_obj_id) override;
    void write2Diskbin(std::string filename) override;
    void readFDiskbin(std::string filename) override;

    const float Kd = 0.5f;
    const float Ks = 5.0f;
    const float s = 0.08f;
    const int brdf_type = 0;
    const static int brdf_sample_num = 128;
    Sampler brdf_sampler;
    float brdf_lookup_table[brdf_sample_num][brdf_sample_num][n*n];
    std::vector<std::vector<float>> _TransferFunc;
    float *sh_base = nullptr;

private:
    void glossyUnshadow(int size, int band2, Sampler* sampler, TransferType type, 
    std::vector<Object*>obj_list, int scene_obj_id, BVHTree* Inbvht = nullptr);
    void glossyShadow(int size, int band2, Sampler* sampler, TransferType type, 
    std::vector<Object*>obj_list, int scene_obj_id, BVHTree* Inbvht = nullptr);
};

#endif
