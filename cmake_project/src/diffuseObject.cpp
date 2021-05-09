#define _USE_MATH_DEFINES
#include <cmath>
#include <fstream>
#include <istream>
#include <iostream>
#include "diffuseObject.h"

void DiffuseObject::write2Diskbin(std::string filename)
{
    std::ofstream out;
    out.open(filename.c_str(), std::ofstream::binary);
    int size = _vertices.size() / 3;
    int band2 = _band * _band;

    out.write((char *)&size, sizeof(int));
    out.write((char *)&_band, sizeof(int));
    out.write((char*)&_sample_size, sizeof(int));
    out.write((char*)&sphereNumber, sizeof(int));
    out.write((char*)&shadowSampleNumber, sizeof(int));
    out.write((char*)&rStep, sizeof(float));

    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < band2; ++j)
        {
            out.write((char *)&_TransferFunc[i][j].x, sizeof(float));
            out.write((char *)&_TransferFunc[i][j].y, sizeof(float));
            out.write((char *)&_TransferFunc[i][j].z, sizeof(float));
        }
    }

    int tmp = sqrt(shadowSampleNumber);
    size = sphereNumber * point_sample._samples.size();
    for (int i = 0; i < size; ++i) 
    {
        for (int j = 0; j < band2; ++j) 
        {
            out.write((char*)&_ShadowField[i][j], sizeof(float));
        }
    }
    size = point_sample._samples.size();
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < 2; ++j) {
            out.write((char*)&point_sample._samples[i]._sphericalCoord[j], sizeof(float));
        }
    }
    out.close();

    std::cout << "Diffuse object generated." << std::endl;
}

void DiffuseObject::readFDiskbin(std::string filename)
{
   _DTransferFunc.clear();
   _ShadowField.clear();
   std::string dataFile = filename;
   std::ifstream in(dataFile, std::ifstream::binary);
   // Assure data file eist in order to avoid memory leak.
   assert(in);
   unsigned int size, band2;

   in.read((char *)&size, sizeof(unsigned int));
   in.read((char *)&_band, sizeof(int));
   in.read((char*)&_sample_size, sizeof(int));
   in.read((char*)&sphereNumber, sizeof(int));
   in.read((char*)&shadowSampleNumber, sizeof(int));
   in.read((char*)&rStep, sizeof(float));

   std::cout << "Diffuse object: " << filename << std::endl;
   std::cout << "band = " << _band << std::endl;

   band2 = _band * _band;

   std::vector<glm::vec3> empty(band2, glm::vec3(0.0f));

   for (size_t i = 0; i < size; ++i)
   {
       _DTransferFunc.push_back(empty);

       for (size_t j = 0; j < band2; ++j)
       {
           in.read((char *)&_DTransferFunc[i][j].x, sizeof(float));
           in.read((char *)&_DTransferFunc[i][j].y, sizeof(float));
           in.read((char *)&_DTransferFunc[i][j].z, sizeof(float));
       }
   }

   size = sphereNumber * shadowSampleNumber;
   std::vector<float> tmp(band2, 0.0f);
   for (int i = 0; i < size; ++i)
   {
       _ShadowField.push_back(tmp);
       for (int j = 0; j < band2; ++j)
       {
           in.read((char*)&_ShadowField[i][j], sizeof(float));
       }
   }
   size = shadowSampleNumber;
   glm::vec2 spherical;
   glm::vec3 cartesian;
   point_sample._samples.clear();
   for (int i = 0; i < size; ++i) {
       in.read((char*)&spherical[0], sizeof(float));
       in.read((char*)&spherical[1], sizeof(float));
       cartesian.x = sin(spherical[0]) * cos(spherical[1]);
       cartesian.y = sin(spherical[0]) * sin(spherical[1]);
       cartesian.z = cos(spherical[0]);
       point_sample._samples.emplace_back(Sample(cartesian, spherical));
   }
   std::cout << "size = " << point_sample._samples.size() << std::endl;
   in.close();
}

float DiffuseObject::testCoef(float* coef, float theta, float phi) {
    int band = _band;
    int band2 = band * band;

    float* base = new float[band2];
    for (int l = 0; l < band; ++l) {
        for (int m = -l; m <= l; ++m) {
            int index = l * (l + 1) + m;
            base[index] = (float)SphericalH::SHvalue(theta, phi, l, m);
        }
    }
    float result = 0;
    for (int i = 0; i < band2; ++i) {
        result += coef[i] * base[i];
        //std::cout << "coef = " << coef[i] << ' ' << "base =" << base[i] << std::endl;
    }
    delete[] base;
    return result;
}

void DiffuseObject::testMap(float* coef, const std::string& path) {
    const int tmp_n = 128;
    cv::Mat gray(tmp_n, tmp_n, CV_32FC1);
    for (int i = 0; i < tmp_n; ++i) {
        for (int j = 0; j < tmp_n; ++j) {
            float x = float(i) / float(tmp_n);
            float y = float(j) / float(tmp_n);
            float theta = std::acos(1 - 2 * x);
            float phi = y * 2 * M_PI;
            float now_value = testCoef(coef, theta, phi);
            gray.at<float>(i, j) = now_value * 255.0;
        }
    }
    cv::imwrite(path, gray);
}

void DiffuseObject::transform(const glm::mat4& m, shRotate& sh_rotate) {
    int band2 = _band * _band;
    float* coef_in = new float[band2];
    float* coef_out = new float[band2];
    int sz = _vertices.size() / 3;
    float nx, ny, nz;
    nx = _cx;
    ny = _cy;
    nz = _cz;
    _cx = m[0][0] * nx + m[0][1] * ny + m[0][2] * nz + m[0][3];
    _cy = m[1][0] * nx + m[1][1] * ny + m[1][2] * nz + m[1][3];
    _cz = m[2][0] * nx + m[2][1] * ny + m[2][2] * nz + m[2][3];
    for (int i = 0; i < sz; ++i) {
        int cur_index = i * 3;
        nx = _vertices[cur_index];
        ny = _vertices[cur_index + 1];
        nz = _vertices[cur_index + 2];
        _vertices[cur_index] = m[0][0] * nx + m[0][1] * ny + m[0][2] * nz + m[0][3];
        _vertices[cur_index + 1] = m[1][0] * nx + m[1][1] * ny + m[1][2] * nz + m[1][3];
        _vertices[cur_index + 2] = m[2][0] * nx + m[2][1] * ny + m[2][2] * nz + m[2][3];
        for (int j = 0; j < band2; ++j)coef_in[j] = _DTransferFunc[i][j].r;
        sh_rotate.rotate(coef_in, coef_out, m, _band);
        for (int j = 0; j < band2; ++j) {
            _DTransferFunc[i][j].r = coef_out[j];
            coef_in[j] = _DTransferFunc[i][j].g;
        }
        sh_rotate.rotate(coef_in, coef_out, m, _band);
        for (int j = 0; j < band2; ++j) {
            _DTransferFunc[i][j].g = coef_out[j];
            coef_in[j] = _DTransferFunc[i][j].b;
        }
        sh_rotate.rotate(coef_in, coef_out, m, _band);
        for (int j = 0; j < band2; ++j)_DTransferFunc[i][j].b = coef_out[j];
    }
    int point_sz = point_sample._samples.size();
    for (int i = 0; i < sphereNumber*point_sz; ++i) {
        for (int j = 0; j < band2; ++j)coef_in[j] = _ShadowField[i][j];
        sh_rotate.rotate(coef_in, coef_out, m, _band);
        for (int j = 0; j < band2; ++j)_ShadowField[i][j] = coef_out[j];
    }

    glm::mat4 tmp_mat;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            tmp_mat[i][j] = 0;
            for (int k = 0; k < 4; ++k) {
                tmp_mat[i][j] += m[i][k] * rotate_mat[k][j];
            }
        }
    }
    for (int i = 0; i < 4; ++i)for (int j = 0; j < 4; ++j)rotate_mat[i][j] = tmp_mat[i][j];
    for (int i = 0; i < 3; ++i)for (int j = 0; j < 3; ++j)rotate_mat_inv[i][j] = rotate_mat[j][i];
    for (int i = 0; i < 3; ++i) {
        rotate_mat_inv[i][3] = 0;
        for (int j = 0; j < 3; ++j) {
            rotate_mat_inv[i][3] -= rotate_mat_inv[i][j] * rotate_mat[j][3];
        }
    }
    /*std::cout << "rotate" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << rotate_mat[i][j] << ' ';
        }std::cout << std::endl;
    }
    std::cout << "inv" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << rotate_mat_inv[i][j] << ' ';
        }std::cout << std::endl;
    }
    std::cout << "result" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            tmp_mat[i][j] = 0;
            for (int k = 0; k < 4; ++k)tmp_mat[i][j] += rotate_mat[i][k] * rotate_mat_inv[k][j];
            std::cout << tmp_mat[i][j] << ' ';
        }std::cout << std::endl;
    }*/

    delete[] coef_in;
    delete[] coef_out;
}

void DiffuseObject::diffuseUnshadow(int size, int band2, Sampler* sampler, TransferType type, BVHTree* Inbvht)
{
    bool shadow = false;
    if (type != T_UNSHADOW)
    {
        shadow = true;
    }

    std::vector<glm::vec3> empty(band2, glm::vec3(0.0f));
    _TransferFunc.resize(size, empty);

    // Build BVH.
    BVHTree bvht;
    if (shadow)
    {
        if (type == T_SHADOW)
            bvht.build(*this);
        else
            bvht = *Inbvht;
    }

    // Sample.
    const int sampleNumber = sampler->_samples.size();
    
    if (shadow) 
    {
        point_sample = Sampler((int)sqrt(shadowSampleNumber), false);

        std::vector<float> shadow_field_tmp(band2, 0.0f);
        int shadow_point_sz = point_sample._samples.size();
        _ShadowField.resize(sphereNumber * shadow_point_sz, shadow_field_tmp);

        Sampler shadow_sample(64);
        shadow_sample.computeSH(_band);
        int shadow_sample_sz = shadow_sample._samples.size();
        float weight = 4.0f * M_PI / shadow_sample_sz;

#pragma omp parallel for
        for (int t = 0; t < sphereNumber; ++t) 
        {
            float now_r = (rStart + t * rStep) * _r;
            for (int j = 0; j < shadow_point_sz; ++j) 
            {
                Sample& stemp = point_sample._samples[j];
                glm::vec3 now_pos = glm::vec3(_cx, _cy, _cz) + now_r*stemp._cartesCoord;
                now_pos.y += 1e-6;
                int i = t * shadowSampleNumber + j;

                if (i % 1000 == 0)std::cout << i << std::endl;
                for (int k = 0; k < shadow_sample_sz; ++k) 
                {
                    Ray testRay(now_pos, shadow_sample._samples[k]._cartesCoord);
                    bool visibility = !bvht.intersect(testRay);
                    if (visibility) 
                    {
                        Sample& tmp_sample = shadow_sample._samples[k];
                        for (int b = 0; b < band2; ++b) 
                        {
                            _ShadowField[i][b] += tmp_sample._SHvalue[b];
                        }
                    }
                }
                for (int b = 0; b < band2; ++b) {
                    _ShadowField[i][b] *= weight;
                }
            }
        }
    }
    std::cout << "shadow field done" << std::endl;

#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        int index = 3 * i;
        glm::vec3 normal(_normals[index + 0], _normals[index + 1], _normals[index + 2]);
        int sample_sz = sampler->_samples.size();
        for (int j = 0; j < sample_sz; j++)
        {
            Sample stemp = sampler->_samples[j];
            float H = std::max(glm::dot(glm::normalize(normal), glm::normalize(stemp._cartesCoord)), 0.0f);
            bool visibility;

            if (shadow)
            {
                Ray testRay(glm::vec3(_vertices[index + 0], _vertices[index + 1], _vertices[index + 2]),
                            stemp._cartesCoord);
                visibility = !bvht.intersect(testRay, false);
            }
            else
            {
                visibility = true;
            }
            H = 1.0f;
            if (!visibility)
            {
                H = 0.0f;
            }
            //Projection.
            for (int k = 0; k < band2; k++)
            {
                float SHvalue = stemp._SHvalue[k];

                _TransferFunc[i][k] += _albedo * SHvalue * H;
            }
        }
    }
    // Normalization.
    float weight = 4.0f * M_PI / sampler->_samples.size();
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < band2; ++j)
        {
            _TransferFunc[i][j] *= weight;
        }
    }
    if (type == T_UNSHADOW)
        std::cout << "Unshadowed transfer vector generated." << std::endl;
}

void DiffuseObject::diffuseShadow(int size, int band2, Sampler* sampler, TransferType type, BVHTree* Inbvht)
{
    std::cout << "shadow" << std::endl;
    diffuseUnshadow(size, band2, sampler, type, Inbvht);
    if (type == T_SHADOW)
        std::cout << "Shadowed transfer vector generated." << std::endl;
    //system("pause");
}

void DiffuseObject::diffuseInterreflect(int size, int band2, Sampler* sampler, TransferType type, int bounce)
{
    BVHTree bvht;
    bvht.build(*this);

    diffuseShadow(size, band2, sampler, type, &bvht);

    const int sampleNumber = sampler->_samples.size();

    auto interReflect = new std::vector<std::vector<glm::vec3>>[bounce + 1];

    interReflect[0] = _TransferFunc;
    std::vector<glm::vec3> empty(band2, glm::vec3(0.0f));

    float weight = 4.0f * M_PI / sampleNumber;

    for (int k = 0; k < bounce; k++)
    {
        std::vector<std::vector<glm::vec3>> zeroVector(size, empty);
        interReflect[k + 1].resize(size);

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            int offset = 3 * i;
            glm::vec3 normal = glm::vec3(_normals[offset + 0], _normals[offset + 1], _normals[offset + 2]);

            for (int j = 0; j < sampleNumber; j++)
            {
                Sample stemp = sampler->_samples[j];
                Ray rtemp(glm::vec3(_vertices[offset + 0], _vertices[offset + 1], _vertices[offset + 2]),
                          stemp._cartesCoord);

                bool visibility = !bvht.intersect(rtemp);
                if (visibility)
                    continue;
                // The direction which is invisible is where the indirect radiance comes from.
                float H = std::max(glm::dot(rtemp._dir, normal), 0.0f);

                int triIndex = 3 * rtemp._index;
                int voffset[3];
                glm::vec3 p[3];
                std::vector<glm::vec3>* SHTrans[3];
                for (int m = 0; m < 3; m++)
                {
                    voffset[m] = _indices[triIndex + m];
                    SHTrans[m] = &interReflect[k][voffset[m]];
                    voffset[m] *= 3;
                    p[m] = glm::vec3(_vertices[voffset[m] + 0], _vertices[voffset[m] + 1], _vertices[voffset[m] + 2]);
                }
                glm::vec3 pc = rtemp._o + (float)rtemp._t * rtemp._dir;

                float u, v, w;
                // Barycentric coordinates for interpolation.
                barycentric(pc, p, u, v, w);

                std::vector<glm::vec3> SHtemp;
                SHtemp.resize(band2);

                for (int m = 0; m < band2; m++)
                {
                    SHtemp[m] = u * SHTrans[0]->at(m) + v * SHTrans[1]->at(m) + w * SHTrans[2]->at(m);
                    zeroVector[i][m] += H * _albedo * SHtemp[m];
                }
            }
        }

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            interReflect[k + 1][i].resize(band2);
            for (int j = 0; j < band2; ++j)
            {
                zeroVector[i][j] *= weight;
                interReflect[k + 1][i][j] = interReflect[k][i][j] + zeroVector[i][j];
            }
        }
    }
    _TransferFunc = interReflect[bounce];
    delete[] interReflect;
    std::cout << "Interreflected transfer vector generated." << std::endl;
}

void DiffuseObject::project2SH(int mode, int band, int sampleNumber, int bounce)
{
    _band = band;

    int size = _vertices.size() / 3;
    int band2 = band * band;

    _sample_size = sampleNumber;
    Sampler stemp((unsigned)sqrt(sampleNumber));
    stemp.computeSH(band);

    if (mode == 1)
    {
        std::cout << "Transfer Type: unshadowed" << std::endl;
        diffuseUnshadow(size, band2, &stemp, T_UNSHADOW);
    }
    else if (mode == 2)
    {
        std::cout << "Transfer Type: shadowed" << std::endl;
        diffuseShadow(size, band2, &stemp, T_SHADOW);
    }
    else if (mode == 3)
    {
        std::cout << "Transfer Type: interreflect" << std::endl;
        diffuseInterreflect(size, band2, &stemp, T_INTERREFLECT, bounce);
    }
}
