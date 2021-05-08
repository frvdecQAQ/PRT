#ifndef RENDERER_H_
#define RENDERER_H_

#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <algorithm>

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>

#include "scene.h"
#include "hdrTextureCube.h"
#include "lighting.h"
#include "diffuseObject.h"
#include "generalObject.h"

#include "shproduct.h"
#include "fastmul.hpp"
#include "shorder.hpp"
#include "fftprecomputed.hpp"

#include "shRotateMatrix.h"
#include "test.h"

// Camera intrinsic parameters definetion.
#define ZOOM 45.0f
#define NEAR_PLANE 0.1f
#define FAR_PLANE 10000.0f

struct MeshVertex
{
    float x;
    float y;
    float z;
    float r;
    float g;
    float b;
};

class Renderer
{
public:
    Renderer() = default;
    virtual ~Renderer();

    void Init(int lightNumber);
    void Render(bool render_again = false);

    FourierSeries<n> sh2fs(SH<n> sh) {
        FourierSeries<n> fs;
#include "sh2fs.cpp"
        return fs;
    }

    SH<n> fs2sh(FourierSeries<2 * n - 1> fs) {
        SH<n> sh;
#include "fs2sh.cpp"
//#include <profileapi.h>
//#include <WTypesbase.h>s
        return sh;
    }

    void Setup(Scene* scene, Lighting* light);

    void SetupColorBuffer(int type, glm::vec3 viewDir, bool diffuse = true);
    void loadTriple(int _band);

private:
    //DiffuseObject* _diffObject;
    //GeneralObject* _genObject;

    Scene* _scene;

    Lighting* _lighting;
    std::vector<float> _colorBuffer;
    std::vector<MeshVertex> _meshBuffer;

    HDRTextureCube* hdrTextures;
    glm::mat4 projection;

    GLuint _VAO;
    GLuint _VBO;

    int band;
    std::vector<int>dst;
    std::vector<std::pair<int, int>>src;
    std::vector<double>coef;

    int triple_product_num;
    float *triple_product_data_a = nullptr;
    float *triple_product_data_b = nullptr;
    float *triple_product_data_c = nullptr;
    float *in_data_a = nullptr;
    float *in_data_b = nullptr;
    float *out_data_c = nullptr;
    cufftComplex *pool0_gpu = nullptr;
    cufftComplex *pool1_gpu = nullptr;
    cufftComplex *pool2_gpu = nullptr;
    float *sh_base_pool = nullptr;

    void objDraw();
    //void setupDiffuseBuffer(int type);
    void setupBuffer(int type, glm::vec3 viewDir);
    void tripleProduct(glm::vec3* result, float* second, int band2);
    void tripleProduct(float* result, float* second, int band2);
    float testCoef(float* coef, float theta, float phi);
    void testMap(float* coef, const std::string& path);
};

#endif
