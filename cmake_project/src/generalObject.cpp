#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <fstream>
#include "generalObject.h"

void GeneralObject::write2Diskbin(std::string filename)
{
    std::ofstream out(filename, std::ofstream::binary);
    int size = _vertices.size() / 3;
    int band2 = _band * _band;

    out.write((char *)&size, sizeof(int));
    out.write((char *)&_band, sizeof(int));

    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < band2; ++k) {
            out.write((char*)&_TransferFunc[i][k], sizeof(float));
        }
    }

    for(int i = 0; i < brdf_sample_num; ++i)
    {
        for(int j = 0; j < brdf_sample_num; ++j)
        {
            for(int k = 0; k < band2; ++k)out.write((char*)&brdf_lookup_table[i][j][k], sizeof(float));
        }
    }

    for(int i = 0; i < brdf_sample_num; ++i)
    {
        for(int j = 0; j < brdf_sample_num; ++j)
        {
            Sample &tmp = brdf_sampler._samples[i*brdf_sample_num+j];
            out.write((char*)&tmp._sphericalCoord[0], sizeof(float));
            out.write((char*)&tmp._sphericalCoord[1], sizeof(float));
            out.write((char*)&tmp._cartesCoord[0], sizeof(float));
            out.write((char*)&tmp._cartesCoord[1], sizeof(float));
            out.write((char*)&tmp._cartesCoord[2], sizeof(float));
        }
    }

    for(int i = 0; i < size*3*band2; ++i){
        out.write((char*)&light_coef[i], sizeof(float));
    }


    out.close();
    std::cout << "Glossy object generated." << std::endl;
}

void GeneralObject::readFDiskbin(std::string filename)
{
    //std::string transf[3] = {"DU.dat", "DS.dat", "DI.dat"};

    //for (int i = 0; i < 3; i++)_TransferMatrix[i].clear();

    std::ifstream in(filename, std::ifstream::binary);
    assert(in);
    
    int size, band2;
    
    in.read((char *)&size, sizeof(int));
    in.read((char *)&_band, sizeof(int));
    
    band2 = _band * _band;
    
    std::vector<float> empty(band2, 0.0f);
    _TransferFunc.resize(size, empty);

    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < band2; ++k) {
            in.read((char*)&_TransferFunc[i][k], sizeof(float));
        }
    }

    for(int i = 0; i < brdf_sample_num; ++i)
    {
        for(int j = 0; j < brdf_sample_num; ++j)
        {
            for(int k = 0; k < band2; ++k)in.read((char*)&brdf_lookup_table[i][j][k], sizeof(float));
        }
    }

    for(int i = 0; i < brdf_sample_num; ++i)
    {
        for(int j = 0; j < brdf_sample_num; ++j)
        {
            glm::vec3 _cartesCoord;
            glm::vec2 _sphericalCoord;
            in.read((char*)&_sphericalCoord[0], sizeof(float));
            in.read((char*)&_sphericalCoord[1], sizeof(float));
            in.read((char*)&_cartesCoord[0], sizeof(float));
            in.read((char*)&_cartesCoord[1], sizeof(float));
            in.read((char*)&_cartesCoord[2], sizeof(float));
            brdf_sampler._samples[i*brdf_sample_num+j] = Sample(_cartesCoord, _sphericalCoord);
        }
    }

    for(int i = 0; i < size*3*band2; ++i){
        in.read((char*)&light_coef[i], sizeof(float));
    }
    
    in.close();
    //computeTBN();
};

/*void GeneralObject::computeTBN()
{
    int vertexNumber = _vertices.size() / 3;
    int faceNumber = _indices.size() / 3;

    std::vector<glm::vec3> tan1;
    std::vector<glm::vec3> tan2;

    tan1.resize(vertexNumber, glm::vec3(0.0f));
    tan2.resize(vertexNumber, glm::vec3(0.0f));

    for (int i = 0; i < faceNumber; i++)
    {
        int offset = 3 * i;
        int vindex[3];
        int tindex[3];
        glm::vec3 p[3];
        glm::vec2 w[3];

        for (int j = 0; j < 3; j++)
        {
            vindex[j] = 3 * _indices[offset + j];
            tindex[j] = 2 * _indices[offset + j];
            p[j] = glm::vec3(_vertices[vindex[j]], _vertices[vindex[j] + 1], _vertices[vindex[j] + 2]);
            //w[j] = glm::vec2(_texcoords[tindex[j]], _texcoords[tindex[j] + 1]);

            tindex[j] /= 2;
        }

        float x1 = p[1].x - p[0].x;
        float x2 = p[2].x - p[0].x;
        float y1 = p[1].y - p[0].y;
        float y2 = p[2].y - p[0].y;
        float z1 = p[1].z - p[0].z;
        float z2 = p[2].z - p[0].z;

        float s1 = w[1].x - w[0].x;
        float s2 = w[2].x - w[0].x;
        float t1 = w[1].y - w[0].y;
        float t2 = w[2].y - w[0].y;

        if (fabs(s1 * t2 - s2 * t1) <= M_ZERO)
        {
            continue;
        }
        float r = 1.0f / (s1 * t2 - s2 * t1);

        glm::vec3 tan1Temp((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r);
        glm::vec3 tan2Temp((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r);

        if (_isnan(tan1Temp.x) || _isnan(tan1Temp.y) || _isnan(tan1Temp.z) || _isnan(tan2Temp.x) || _isnan(tan2Temp.y)
            || _isnan(tan2Temp.z))
        {
            system("pause");
            continue;
        }

        tan1[tindex[0]] += tan1Temp;
        tan1[tindex[1]] += tan1Temp;
        tan1[tindex[2]] += tan1Temp;

        tan2[tindex[0]] += tan2Temp;
        tan2[tindex[1]] += tan2Temp;
        tan2[tindex[2]] += tan2Temp;
    }
    _tangent.resize(vertexNumber, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    for (int i = 0; i < vertexNumber; i++)
    {
        int offset = 3 * i;
        glm::vec3 t = tan1[i];
        glm::vec3 n = glm::vec3(_normals[offset + 0], _normals[offset + 1], _normals[offset + 2]);

        glm::vec3 result = t - n * glm::dot(n, t);
        if (fabs(result.x * result.x + result.y * result.y + result.z * result.z) <= M_ZERO)
        {
            _tangent[i] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            //	std::cout << "tangent zero" << std::endl;
            //	system("pause");
        }
        else
        {
            _tangent[i] = glm::vec4(glm::normalize(result), 1.0f);
        }

        if (glm::dot(glm::cross(n, t), tan2[i]) < 0.0f)
        {
            _tangent[i].w = -1.0f;
        }
    }
}*/

void GeneralObject::glossyUnshadow(int size, int band2, class Sampler* sampler, TransferType type, 
std::vector<Object*>obj_list, int scene_obj_id, BVHTree* Inbvht)
{
    bool shadow = false;
    if (type != T_UNSHADOW)
    {
        shadow = true;
    }
    bool visibility;

    /*Eigen::MatrixXf empty(band2, band2);
    empty.setZero();
    _TransferMatrix[0].resize(size, empty);*/

    std::vector<float> empty(band2, 0.0f);
    _TransferFunc.resize(size, empty);

    // Build BVH.
    /*BVHTree bvht;
    if (shadow)
    {
        if (type == T_SHADOW)
        {
            bvht.build(*this);
        }
        else
        {
            bvht = *Inbvht;
        }
    }*/

    int obj_sz = (int)(obj_list.size());
    BVHTree bvht[obj_sz];
    for(int i = 0; i < obj_sz; ++i)bvht[i].build(*obj_list[i]);

    // Sample.
    const int sampleNumber = sampler->_samples.size();
    const float weight = 4.0f * M_PI / sampleNumber;
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        std::cout << "glossy : " << i << '/' << size << std::endl;
        int index = 3 * i;
        glm::vec3 normal = glm::vec3(_normals[index + 0], _normals[index + 1], _normals[index + 2]);
        normal = glm::normalize(normal);
        //std::cout << normal[0] << ' ' << normal[1] << ' ' << normal[2] << std::endl;
        glm::vec3 u;
        u = glm::cross(normal, glm::vec3(0.0f, 1.0f, 0.0f));
        if (glm::dot(u, u) < 1e-3f) {
            u = glm::cross(normal, glm::vec3(1.0f, 0.0f, 0.0f));
        }
        u = glm::normalize(u);
        glm::vec3 v = glm::cross(normal, u);
        //std::cout << u[0] << ' ' << u[1] << ' ' << u[2] << std::endl;
        //std::cout << v[0] << ' ' << v[1] << ' ' << v[2] << std::endl;
        for (int j = 0; j < sampleNumber; ++j) {
            Sample& stemp = sampler->_samples[j];
            /*glm::vec3 testDir =
                stemp._cartesCoord[0] * u +
                stemp._cartesCoord[1] * v +
                stemp._cartesCoord[2] * normal;*/
            //float G = std::max(glm::dot(glm::normalize(normal), glm::normalize(stemp._cartesCoord)), 0.0f);
            float G = 1.0f;
            if (shadow) {
                Ray testRay(glm::vec3(_vertices[index + 0], _vertices[index + 1], _vertices[index + 2]),
                    stemp._cartesCoord);
                bool visibility = true;
                for(int k = 0; k < obj_list.size(); ++k){
                    if(scene_obj_id != k)visibility &= (!bvht[k].intersect(testRay, false));
                    else visibility &= (!bvht[k].intersect(testRay, true));
                }
                if (!visibility) {
                    G = 0.0f;
                }
            }
            for (int k = 0; k < band2; k++){
                float SHvalue = stemp._SHvalue[k];
                _TransferFunc[i][k] += G * SHvalue;
            }
        }
        for (int k = 0; k < band2; ++k) {
            _TransferFunc[i][k] *= weight;
        }
    }

    int band = (int)(sqrt(band2));
    brdf_sampler.computeSH(band);
    const float brdf_weight = 4.0f*M_PI/(brdf_sample_num*brdf_sample_num);

    glm::vec3 n(0.0f, 1.0f, 0.0f);
    glm::vec3 v(1.0f, 0.0f, 0.0f);
    glm::vec3 u(0.0f, 0.0f, 1.0f);

    for(int i = 0; i < brdf_sample_num; ++i)
    {
        for(int j = 0; j < brdf_sample_num; ++j)
        {
            for(int k = 0; k < band2; ++k)brdf_lookup_table[i][j][k] = 0;
            Sample &vsp = brdf_sampler._samples[i*brdf_sample_num+j];
            if(brdf_type == 0)
            {
                for(int k = 0; k < brdf_sample_num*brdf_sample_num; ++k)
                {
                    Sample &lsp = brdf_sampler._samples[k];
                    float brdf;
                    //if(vsp._sphericalCoord[0] >= M_PI/2.0f || lsp._sphericalCoord[0] >= M_PI/2.0f)brdf = 0.0f;
                    //else
                    {
                        glm::vec3 reflect = 2*glm::dot(n, lsp._cartesCoord)*n-lsp._cartesCoord;
                        float specular = std::max(glm::dot(glm::normalize(reflect), glm::normalize(vsp._cartesCoord)),
                                        0.0f);
                        brdf = Kd+Ks*powf(specular, s);
                    }
                    for(int l = 0; l < band2; ++l)
                    {
                        brdf_lookup_table[i][j][l] += lsp._SHvalue[l]*brdf*std::max(0.0f, lsp._cartesCoord.z);
                    }
                }
            }
            else
            {
                for(int k = 0; k < brdf_sample_num*brdf_sample_num; ++k)
                {
                    Sample &lsp = brdf_sampler._samples[k];
                    glm::vec3 h = glm::normalize(vsp._cartesCoord + lsp._cartesCoord);
                    float delta = acos(glm::dot(h, n));

                    float brdf;
                    if(vsp._sphericalCoord[0] >= M_PI/2.0f || lsp._sphericalCoord[0] >= M_PI/2.0f)brdf = 0.0f;
                    else
                    {
                        float factor1 = 1.0f/sqrt(cos(lsp._cartesCoord[0])*cos(vsp._cartesCoord[0]));
                        float factor2 = exp(-pow(tan(delta), 2)/pow(s, 2))/(4.0f*M_PI*pow(s, 2));
                        brdf = Kd + Ks*factor1*factor2;
                    }
                    for(int l = 0; l < band2; ++l)
                    {
                        brdf_lookup_table[i][j][l] += lsp._SHvalue[l]*brdf*std::max(0.0f, lsp._cartesCoord.z);
                    }
                }
            }
            for(int k = 0; k < band2; ++k)
            {
                brdf_lookup_table[i][j][k] = brdf_lookup_table[i][j][k]*brdf_weight;
            }
        }
    }

     for(int i = 0; i < size*3*band2; ++i)light_coef[i] = 0;

#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        if(i % 1000 == 0)std::cout << i << "/" << size << std::endl;
        int index = 3 * i;
        int sample_sz = sampler->_samples.size();
        glm::vec3 normal = glm::vec3(_normals[index + 0], _normals[index + 1], _normals[index + 2]);
        normal = glm::normalize(normal);
        glm::vec3 u;
        u = glm::cross(normal, glm::vec3(0.0f, 1.0f, 0.0f));
        if (glm::dot(u, u) < 1e-3f) {
            u = glm::cross(normal, glm::vec3(1.0f, 0.0f, 0.0f));
        }
        u = glm::normalize(u);
        glm::vec3 v = glm::cross(normal, u);
        for (int j = 0; j < sample_sz; j++)
        {
            Sample stemp = sampler->_samples[j];
            /*glm::vec3 testDir =
                stemp._cartesCoord[0] * u +
                stemp._cartesCoord[1] * v +
                stemp._cartesCoord[2] * normal;*/
            float H = 0.0f;
            Ray testRay(glm::vec3(_vertices[index + 0], _vertices[index + 1], _vertices[index + 2]),
                            stemp._cartesCoord);
            bool visibility = false;
            for(int k = 0; k < 2; ++k){
                visibility |= rayTriangle(testRay, light_triangle[k], false);
            }
            if(visibility)
            {
                H = 1.0f;
            }
            //Projection.
            for (int k = 0; k < band2; k++)
            {
                float SHvalue = stemp._SHvalue[k];

                light_coef[i*3*band2+k] += SHvalue * H;
                light_coef[i*3*band2+band2+k] += SHvalue * H;
                light_coef[i*3*band2+2*band2+k] += SHvalue * H;
            }
        }
    }
#pragma omp parallel for
    for(int i = 0; i < size*3*band2; ++i){
        light_coef[i] *= weight;
    }
}

void GeneralObject::glossyShadow(int size, int band2, Sampler* sampler, TransferType type, 
std::vector<Object*>obj_list, int scene_obj_id, BVHTree* Inbvht)
{
    glossyUnshadow(size, band2, sampler, type, obj_list, scene_obj_id, Inbvht);
    if (type == T_SHADOW)
    {
        std::cout << "Shadowed transfer matrix generated." << std::endl;
    }
}

/*void GeneralObject::glossyInterReflect(int size, int band2, Sampler* sampler, TransferType type, int bounce)
{
    BVHTree bvht;
    bvht.build(*this);

    glossyShadow(size, band2, sampler, type, &bvht);

    const int sampleNumber = sampler->_samples.size();

    auto interReflect = new std::vector<Eigen::MatrixXf>[bounce + 1];

    interReflect[0] = _TransferMatrix[0];
    Eigen::MatrixXf empty(band2, band2);
    empty.setZero();

    float weight = 4.0f * M_PI / sampleNumber;

    for (int k = 0; k < bounce; k++)
    {
        std::vector<Eigen::MatrixXf> zeroVector(size, empty);
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
                {
                    continue;
                }
                // The direction which is invisible is where the indirect radiance comes from.
                float G = std::max(glm::dot(rtemp._dir, normal), 0.0f);

                int triIndex = 3 * rtemp._index;
                int voffset[3];
                glm::vec3 p[3];
                Eigen::MatrixXf SHTrans[3];
                for (int m = 0; m < 3; m++)
                {
                    voffset[m] = _indices[triIndex + m];
                    SHTrans[m] = interReflect[k][voffset[m]];
                    voffset[m] *= 3;
                    p[m] = glm::vec3(_vertices[voffset[m] + 0], _vertices[voffset[m] + 1],
                                     _vertices[voffset[m] + 2]);
                }
                glm::vec3 pc = rtemp._o + (float)rtemp._t * rtemp._dir;

                float u, v, w;
                // Barycentric coordinates for interpolation.
                barycentric(pc, p, u, v, w);

                for (int li = 0; li < _band; li++)
                {
                    for (int mi = -li; mi <= li; mi++)
                    {
                        for (int lj = 0; lj < _band; lj++)
                        {
                            for (int mj = -lj; mj <= lj; ++mj)
                            {
                                int iindex = li * (li + 1) + mi;
                                int jindex = lj * (lj + 1) + mj;

                                float SHtemp = u * SHTrans[0](iindex, jindex) + v * SHTrans[1](iindex, jindex) + w *
                                    SHTrans[2](iindex, jindex);
                                zeroVector[i](iindex, jindex) += SHtemp * G;
                            }
                        }
                    }
                }
            }
        }

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            interReflect[k + 1][i].resize(band2, band2);
            // Normalization and propogation.
            for (int li = 0; li < _band; li++)
            {
                for (int mi = -li; mi <= li; mi++)
                {
                    for (int lj = 0; lj < _band; lj++)
                    {
                        for (int mj = -lj; mj <= lj; mj++)
                        {
                            int iindex = li * (li + 1) + mi;
                            int jindex = lj * (lj + 1) + mj;

                            zeroVector[i](iindex, jindex) *= weight;
                            interReflect[k + 1][i](iindex, jindex) = interReflect[k][i](iindex, jindex) + zeroVector[i](
                                iindex, jindex);
                        }
                    }
                }
            }
        }
    }
    _TransferMatrix[0] = interReflect[bounce];
    delete[] interReflect;
    std::cout << "Interreflected transfer matrix generated." << std::endl;
}*/

void GeneralObject::project2SH(int mode, int band, int sampleNumber, int bounce,
    std::vector<Object*>obj_list, int scene_obj_id)
{
    _difforGeneral = true;
    _band = band;

    int size = _vertices.size() / 3;
    int band2 = _band * _band;

    Sampler stemp((unsigned)sqrt(sampleNumber));
    stemp.computeSH(band);

    if (mode == 1)
    {
        std::cout << "Transfer Type: unshadowed" << std::endl;
        glossyUnshadow(size, band2, &stemp, T_UNSHADOW, obj_list, scene_obj_id);
    }
    else if (mode == 2)
    {
        std::cout << "Transfer Type: shadowed" << std::endl;
        glossyShadow(size, band2, &stemp, T_SHADOW, obj_list, scene_obj_id);
    }
    else if (mode == 3)
    {
        std::cout << "Transfer Type: interreflect" << std::endl;
        //glossyInterReflect(size, band2, &stemp, T_INTERREFLECT, bounce);
    }
}
