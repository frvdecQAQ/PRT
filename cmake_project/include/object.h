#ifndef OBJECT_H_
#define OBJECT_H_

#include <vector>
#include <string>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include "utils.h"
#include "sampler.h"
#include "shorder.hpp"

enum TransferType
{
    T_UNSHADOW,
    T_SHADOW,
    T_INTERREFLECT
};

struct Intersection
{
    bool _intersect;
    Ray _ray;

    Intersection()
    {
        _intersect = false;
    }

    Intersection(bool in, Ray rayin)
        : _intersect(in), _ray(rayin)
    {
    }
};

class Object
{

public:
    Object():
        _theta(0.0f),
        _rx(0.0f),
        _ry(0.0f),
        _rz(1.0f),
        _difforGeneral(false)
    {
        light_triangle[0]._v0 = glm::vec3(1, 1.5, 0);
        light_triangle[0]._v1 = glm::vec3(-1, 1.5, 0);
        light_triangle[0]._v2 = glm::vec3(0, 1.5, 1);

        light_triangle[1]._v0 = glm::vec3(1, 1.5, 0);
        light_triangle[1]._v1 = glm::vec3(-1, 1.5, 0);
        light_triangle[1]._v2 = glm::vec3(0, 1.5, -1);
    }

    ~Object();

    void init(std::string path, glm::vec3 albedo, glm::vec3 scale, bool texture = true);
    void queryOOF(glm::vec3 p, float* coef, bool debug = false);

    // Project to SH function.
    virtual void project2SH(int mode, int band, int sampleNumber, int bounce, 
        std::vector<Object*>obj_list){}
    // IO functions.
    virtual void write2Diskbin(std::string filename){}
    virtual void readFDiskbin(std::string filename){}
    bool intersectTest(Ray& ray, int facenumber);
    void normVertices(glm::vec3 scale);
    unsigned index_from_str(const std::string& str);

    void setRotate(float theta, float x, float y, float z)
    {
        _theta = theta;
        _rx = x;
        _ry = y;
        _rz = z;
    }

    int band() { return _band; }

protected:
    float _vmaxX, _vmaxY, _vmaxZ;
    float _vminX, _vminY, _vminZ;

    glm::vec3 _albedo;
    int _band;
    int _sample_size;

    const int sphereNumber = 32;
    const int shadowSampleNumber = 48 * 48;
    const float rStep = 0.2f;
    const float rStart = 0.2f;

    bool _difforGeneral; //false means diffuse

public:
    std::vector<float> _vertices;
    std::vector<float> _normals;
    std::vector<glm::mat4> normal_space_rotate_matrix;
    std::vector<float> _texcoords;
    std::vector<GLuint> _indices;
    std::string _modelname;

    // Model rotation.
    float _theta;
    float _rx, _ry, _rz;

    glm::mat4 rotate_mat;
    glm::mat4 rotate_mat_inv;

    float _cx, _cy, _cz;
    float _r;

    float init_x, init_y, init_z;

    std::vector<std::vector<float>> _ShadowField;
    Sampler point_sample;

    float* light_coef = nullptr;
    Triangle light_triangle[2];
    
};

#endif
