#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "renderer.h"
#include "UI.h"
#include "resource_manager.h"
#include "sphericalHarmonics.h"
#include "brdf.h"


extern bool drawCubemap;
extern bool simpleLight;

extern std::string lightings[];
extern int lightingIndex;
extern int objectIndex;
extern int bandIndex;
extern int BRDFIndex;

// Window.
extern int WIDTH;
extern int HEIGHT;

// Camera.
extern float camera_dis;
extern glm::vec3 camera_pos;
extern glm::vec3 last_camera_pos;
extern glm::vec3 camera_dir;
extern glm::vec3 camera_up;

// Rotation.
extern int g_AutoRotate;
extern int g_RotateTime;
extern glm::fquat last_Rotation;
extern glm::mat4 rotateMatrix;

// Mesh information.
int vertices;
int faces;

Renderer::~Renderer()
{
    if(in_data_a != nullptr)delete[] in_data_a;
    if(in_data_b != nullptr)delete[] in_data_b;
    if(out_data_c != nullptr)delete[] out_data_c;
    if(sh_base_pool != nullptr)delete[] sh_base_pool;
    delete[]hdrTextures;
}

//load sparse_n file(traditional method)
void Renderer::loadTriple(int _band) {
    initGamma();

    band = _band;
    std::string sparse_file_path = "./processedData/triple/sparse" + std::to_string(band);
    FILE* sparse_file = fopen(sparse_file_path.c_str(), "r");
    std::cout << sparse_file_path << std::endl;
    char ch;
    int number_cnt = 0;
    int tmp_int = 0;
    double tmp_double, tmp_back;
    std::pair<int, int> tmp_pair;
    bool tmp_flag = false;
    while (true) {
        ch = fgetc(sparse_file);
        if (ch == EOF)break;
        if (ch == ',') {
            if (number_cnt == 0)dst.push_back(tmp_int);
            else if (number_cnt == 1)tmp_pair.first = tmp_int;
            else if (number_cnt == 2) {
                tmp_pair.second = tmp_int;
                src.push_back(tmp_pair);
            }
            else {
                if (tmp_flag)tmp_double = -tmp_double;
                coef.push_back(tmp_double);
            }
            number_cnt++;
            if (number_cnt == 4)number_cnt = 0;
            if (number_cnt < 3)tmp_int = 0;
            else {
                tmp_double = 0;
                tmp_back = 0;
                tmp_flag = false;
            }
        }
        if (ch != '-' && ch != '.' && (ch < '0' || ch > '9'))continue;
        if (ch == '.')tmp_back = 0.1;
        else if (ch == '-')tmp_flag = true;
        else if (number_cnt < 3)tmp_int = tmp_int * 10 + ch - '0';
        else {
            if (tmp_back != 0) {
                tmp_double += (ch - '0') * tmp_back;
                tmp_back *= 0.1;
            }
            else tmp_double = tmp_double * 10 + ch - '0';
        }
    }
    int sz = (int)(dst.size());
    fclose(sparse_file);
}

void Renderer::Init(const int lightNumber)
{
    // Initialize cubemap.
    hdrTextures = new HDRTextureCube[lightNumber];
    for (int i = 0; i < lightNumber; i++)
    {
        hdrTextures[i].Init("lightings/cross/" + lightings[i] + "_cross" + ".hdr");
    }

    // Initialize projection matrix.
    projection = glm::perspective(ZOOM, (float)WIDTH / (float)HEIGHT, NEAR_PLANE, FAR_PLANE);
}

void Renderer::Setup(Scene* scene, Lighting* light)
{
    _scene = scene;
    _lighting = light;
    triple_product_num = 0;
    for(int i = 0; i < scene->obj_num; ++i)
    {
        if(scene->type_list[i] == 1)
            triple_product_num += scene->obj_list[i]->_vertices.size();
    }
    if(triple_product_num%blocksize != 0)
    {
        triple_product_num = (triple_product_num/blocksize+1)*blocksize;
    }
    cudaMalloc(&triple_product_data_a, triple_product_num*n*n*sizeof(float));
    cudaMalloc(&triple_product_data_b, triple_product_num*n*n*sizeof(float));
    cudaMalloc(&triple_product_data_c, triple_product_num*n*n*sizeof(float));
    in_data_a = new float[triple_product_num*n*n];
    in_data_b = new float[triple_product_num*n*n];
    out_data_c = new float[triple_product_num*n*n];
    memset(in_data_a, 0, sizeof(in_data_a));
    memset(in_data_b, 0, sizeof(in_data_b));
    cudaMalloc((void**)&pool0_gpu, sizeof(cufftComplex)*N*N*triple_product_num);
	cudaMalloc((void**)&pool1_gpu, sizeof(cufftComplex)*N*N*triple_product_num);
	cudaMalloc((void**)&pool2_gpu, sizeof(cufftComplex)*N*N*triple_product_num);

    int band = _scene->_band;
    int band2 = band*band;
    shRotate sh_rotate(band);
    float *coef_in = new float[band2];
    float *coef_out = new float[band2];
    
    //prepare gpu data
    int in_data_a_index = 0;
    int in_data_b_index = 0;
    int max_vertex_num = 0;
    for(int obj_id = 0; obj_id < scene->obj_num; ++obj_id)
    {
        if(_scene->type_list[obj_id] == 0)continue;
        int vertex_number = _scene->obj_list[obj_id]->_vertices.size() / 3;
        max_vertex_num = std::max(vertex_number, max_vertex_num);
        GeneralObject* obj_now = dynamic_cast<GeneralObject*>(_scene->obj_list[obj_id]);

        for(int i = 0; i < vertex_number; ++i)
        {
            int offset = 3*i;

            for(int j = 0; j < 3; ++j){
                for(int k = 0; k < band2; ++k)coef_in[k] = _lighting->_Vcoeffs[j](k);
                sh_rotate.rotate(coef_in, coef_out, obj_now->normal_space_rotate_matrix[i], band);
                for(int k = 0; k < band2; ++k)in_data_a[in_data_a_index++] = coef_out[k];
            }

            for(int j = 0; j < 3; ++j){
                for(int k = 0; k < band2; ++k)in_data_b[in_data_b_index++] = obj_now->_TransferFunc[i][k];
            }
        }
    }
    //if(max_vertex_num != 0)sh_base_pool = new float[max_vertex_num*band2];

    delete[] coef_in;
    delete[] coef_out;
}

void Renderer::SetupColorBuffer(int type, glm::vec3 viewDir, bool diffuse)
{
    setupBuffer(type, viewDir);
}

void Renderer::tripleProduct(glm::vec3* result, float* second, int band2) {
    std::vector<glm::vec3>first;
    first.clear();
    for (int i = 0; i < band2; ++i) {
        first.push_back(result[i]);
        result[i] = glm::vec3(0, 0, 0);
    }
    int sz = dst.size();
    for (int i = 0; i < sz; ++i) {
        double tmp = second[src[i].second] * coef[i];

        result[dst[i]].r += first[src[i].first].r * tmp;
        result[dst[i]].g += first[src[i].first].g * tmp;
        result[dst[i]].b += first[src[i].first].b * tmp;
    }
    /*SH<n> sh1, sh2, sh3, sh4, sh5, sh6, sh7;
    for (int l = 0; l < band; ++l) {
        for (int m = -l; m <= l; ++m) {
            int index = l * (l + 1) + m;
            sh1.at(l, m) = result[index].r;
            sh2.at(l, m) = result[index].g;
            sh3.at(l, m) = result[index].b;
            sh4.at(l, m) = second[index];
        }
    }
    sh5 = fs2sh(fastmul(sh2fs(sh1), sh2fs(sh4)));
    sh6 = fs2sh(fastmul(sh2fs(sh2), sh2fs(sh4)));
    sh7 = fs2sh(fastmul(sh2fs(sh3), sh2fs(sh4)));
    for (int l = 0; l < n; ++l) {
        for (int m = -l; m <= l; ++m) {
            int index = l * (l + 1) + m;
            result[index] = glm::vec3(sh5.at(l, m), sh6.at(l, m), sh7.at(l, m));
        }
    }*/
}

void Renderer::tripleProduct(float* result, float* second, int band2) {
    std::vector<float>first;
    first.clear();
    for (int i = 0; i < band2; ++i) {
        first.push_back(result[i]);
        result[i] = 0.0f;
    }
    int sz = dst.size();
    for (int i = 0; i < sz; ++i) {
        result[dst[i]] += first[src[i].first] * second[src[i].second] * coef[i];
    }
    /*SH<n> sh1, sh2, sh3;
    for (int l = 0; l < n; ++l) {
        for (int m = -l; m <= l; ++m) {
            int index = l * (l + 1) + m;
            sh1.at(l, m) = result[index];
            sh2.at(l, m) = second[index];
        }
    }
    sh3 = fs2sh(fastmul(sh2fs(sh1), sh2fs(sh2)));
    for (int l = 0; l < n; ++l) {
        for (int m = -l; m <= l; ++m) {
            int index = l * (l + 1) + m;
            result[index] = sh3.at(l, m);
        }
    }*/
}

float Renderer::testCoef(float* coef, float theta, float phi) {
    int band = _scene->_band;
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

void Renderer::testMap(float* coef, const std::string& path) {
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

void Renderer::setupBuffer(int type, glm::vec3 viewDir)
{
    int band = _scene->_band;
    int band2 = band * band;
    int sz = (int)_scene->obj_list.size();
    _meshBuffer.clear();
    //_meshBuffer.resize(_scene->indices_num*3);
    viewDir = glm::normalize(viewDir);

    //shRotate sh_rotate(band);
    //float *coef_in = new float[band2];
    //float *coef_out = new float[band2];
    
    /*for(int obj_id = 0; obj_id < sz; ++obj_id)
    {
        if(_scene->type_list[obj_id] == 0)continue;
        int vertex_number = _scene->obj_list[obj_id]->_vertices.size() / 3;

        for(int i = 0; i < vertex_number; ++i)
        {
            int offset = 3*i;
            GeneralObject* obj_now = dynamic_cast<GeneralObject*>(_scene->obj_list[obj_id]);
            glm::vec3 normal(obj_now->_normals[offset], obj_now->_normals[offset + 1],
                obj_now->_normals[offset + 2]);

            normal = glm::normalize(normal);
            glm::vec3 u;
            u = glm::cross(normal, glm::vec3(0.0f, 1.0f, 0.0f));
            if (glm::dot(u, u) < 1e-3f) {
                u = glm::cross(normal, glm::vec3(1.0f, 0.0f, 0.0f));
            }
            u = glm::normalize(u);
            glm::vec3 v = glm::cross(normal, u);

            glm::mat4 rotateMatrix;
            rotateMatrix[3][3] = 1;
            rotateMatrix[0][0] = u[0];
            rotateMatrix[0][1] = u[1];
            rotateMatrix[0][2] = u[2];
            rotateMatrix[1][0] = v[0];
            rotateMatrix[1][1] = v[1];
            rotateMatrix[1][2] = v[2];
            rotateMatrix[2][0] = normal[0];
            rotateMatrix[2][1] = normal[1];
            rotateMatrix[2][2] = normal[2];

            for(int j = 0; j < 3; ++j){
                for(int k = 0; k < band2; ++k)coef_in[k] = _lighting->_Vcoeffs[j](k);
                sh_rotate.rotate(coef_in, coef_out, rotateMatrix, band);
                for(int k = 0; k < band2; ++k)in_data_a[in_data_a_index++] = coef_out[k];
            }

            for(int j = 0; j < 3; ++j){
                for(int k = 0; k < band2; ++k)in_data_b[in_data_b_index++] = obj_now->_TransferFunc[i][k];
            }
        }
    }*/
    double start_time = glfwGetTime();
    cudaMemcpy(triple_product_data_a, in_data_a, triple_product_num*n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(triple_product_data_b, in_data_b, triple_product_num*n*n*sizeof(float), cudaMemcpyHostToDevice);
    //traditional_method_gpu(triple_product_data_a, triple_product_data_b, triple_product_data_c, triple_product_num);
    shprod_many(triple_product_data_a, triple_product_data_b, triple_product_data_c, triple_product_num,
        pool0_gpu, pool1_gpu, pool2_gpu);
    cudaMemcpy(out_data_c, triple_product_data_c, triple_product_num*n*n*sizeof(float), cudaMemcpyDeviceToHost);
    double end_time = glfwGetTime();
    std::cout << "triple_product_time = " << end_time-start_time << std::endl;
    start_time = end_time;
    //check code
    /*for(int i = 0; i < band2; ++i){
        coef_in[i] = in_data_a[i];
        coef_out[i] = in_data_b[i];
    }

    tripleProduct(coef_out, coef_in, band2);
    for(int i = 0; i < band2; ++i){
        std::cout << coef_out[i] << ' ';
    }std::cout << std::endl;
    std::cout << "check code" << std::endl;
    for(int i = 0; i < band2; ++i){
        std::cout << out_data_c[i] << ' ';
    }std::cout << std::endl;*/

    int base_index_obj = 0;
    //int base_index_face = 0;
    for (int obj_id = 0; obj_id < sz; ++obj_id) 
    {
        int vertex_number = _scene->obj_list[obj_id]->_vertices.size() / 3;
        _colorBuffer.clear();
        _colorBuffer.resize(vertex_number * 3);

        if(_scene->type_list[obj_id] == 0)
        {
            DiffuseObject* obj_now = dynamic_cast<DiffuseObject*>(_scene->obj_list[obj_id]);
#pragma omp parallel for
            for(int i = 0; i < vertex_number; ++i)
            {
                int offset = 3*i;
                float cr = 0.0f, cg = 0.0f, cb = 0.0f;

                for (int j = 0; j < band2; j++)
                {
                   cr += _lighting->_Vcoeffs[0](j)*obj_now->_DTransferFunc[i][j].r;
                   cg += _lighting->_Vcoeffs[1](j)*obj_now->_DTransferFunc[i][j].g;
                   cb += _lighting->_Vcoeffs[2](j)*obj_now->_DTransferFunc[i][j].b;
                }

                cr *= _lighting->hdrEffect().r;
                cg *= _lighting->hdrEffect().g;
                cb *= _lighting->hdrEffect().b;

                _colorBuffer[offset] = cr;
                _colorBuffer[offset + 1] = cg;
                _colorBuffer[offset + 2] = cb;
            }
        }
        else
        {
            GeneralObject* obj_now = dynamic_cast<GeneralObject*>(_scene->obj_list[obj_id]);
            float* coef_out = new float[band2];
//#pragma omp parallel for
            for(int i = 0; i < vertex_number; ++i){
                int offset = 3*i;
                float cr = 0.0f, cg = 0.0f, cb = 0.0f;
                
                int base_index_r = base_index_obj+3*band2*i;
                int base_index_g = base_index_r+band2;
                int base_index_b = base_index_g+band2;
                for (int j = 0; j < band2; ++j) 
                {
                    cr += in_data_a[base_index_r+j]*in_data_b[base_index_r+j];
                    cg += in_data_a[base_index_g+j]*in_data_b[base_index_g+j];
                    cb += in_data_a[base_index_b+j]*in_data_b[base_index_b+j];
                }

                cr = std::max(cr, 0.0f);
                cg = std::max(cg, 0.0f);
                cb = std::max(cb, 0.0f);

                cr *= obj_now->Kd;
                cg *= obj_now->Kd;
                cb *= obj_now->Kd;

                glm::vec3 view_dir_local;
                glm::mat4 &rotate_matrix = obj_now->normal_space_rotate_matrix[i];
                for (int j = 0; j < 3; ++j) 
                {
                    view_dir_local[j] = 0;
                    for (int k = 0; k < 3; ++k) view_dir_local[j] += rotate_matrix[j][k] * viewDir[k];
                }

                view_dir_local = glm::normalize(view_dir_local);

                float theta = std::acos(view_dir_local[2]);
                float phi = std::atan2(view_dir_local[1], view_dir_local[0]);
                if (phi < 0)phi += 2 * M_PI;

                //int sh_base_index = band2*i;
                SphericalH::SHvalueALL(band, theta, phi, coef_out);

                float tmp_r, tmp_g, tmp_b;
                tmp_r = tmp_g = tmp_b = 0.0f;
                for (int l = 0; l < band; ++l) 
                {
                    float tmp = l * l;
                    tmp /= (obj_now->s * 2.0f);
                    float rho = exp(-tmp);
                    for (int m = -l; m <= l; ++m) 
                    {
                        int index = l * (l + 1) + m;
                        float &sh_base = coef_out[index];
                        tmp_r += rho * out_data_c[base_index_r+index] * sh_base;
                        tmp_g += rho * out_data_c[base_index_g+index] * sh_base;
                        tmp_b += rho * out_data_c[base_index_b+index] * sh_base;
                    }
                }
                cr += std::max(tmp_r, 0.0f) * obj_now->Ks;
                cg += std::max(tmp_g, 0.0f) * obj_now->Ks;
                cb += std::max(tmp_b, 0.0f) * obj_now->Ks;

                _colorBuffer[offset] = cr;
                _colorBuffer[offset + 1] = cg;
                _colorBuffer[offset + 2] = cb;
            }
            base_index_obj += (vertex_number*3*band2);
            delete[] coef_out;
        }
        // Generate mesh buffer.
        Object *obj_now = _scene->obj_list[obj_id];
        int facenumber = obj_now->_indices.size() / 3;

//#pragma omp parallel for
        for (int i = 0; i < facenumber; i++)
        {
            int offset = 3 * i;
            int index[3] = {
                obj_now->_indices[offset + 0],
                obj_now->_indices[offset + 1],
                obj_now->_indices[offset + 2],
            };

            for (int j = 0; j < 3; j++)
            {
                int Vindex = 3 * index[j];
                MeshVertex vertex = {
                    obj_now->_vertices[Vindex + 0],
                    obj_now->_vertices[Vindex + 1],
                    obj_now->_vertices[Vindex + 2],
                    _colorBuffer[Vindex + 0],
                    _colorBuffer[Vindex + 1],
                    _colorBuffer[Vindex + 2]
                };
                //_meshBuffer[base_index_face+3*i+j] = vertex;
                _meshBuffer.push_back(vertex);
            }
        }
        //base_index_face += (facenumber*3);
    }

    end_time = glfwGetTime();
    std::cout << "shading_time = " << end_time-start_time << std::endl;

    // Set the objects we need in the rendering process (namely, VAO, VBO and EBO).
    if (!_VAO)
    {
        glGenVertexArrays(1, &_VAO);
    }
    if (!_VBO)
    {
        glGenBuffers(1, &_VBO);
    }
    glBindVertexArray(_VAO);
         
    glBindBuffer(GL_ARRAY_BUFFER, _VBO);
    glBufferData(GL_ARRAY_BUFFER, _meshBuffer.size() * sizeof(MeshVertex), &(_meshBuffer[0]), GL_STATIC_DRAW);

    // Position attribute.
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (GLvoid*)0);
    glEnableVertexAttribArray(0);
    // Color attribute.
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (GLvoid*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Unbind.
    glBindVertexArray(0);
}
void Renderer::objDraw()
{
    glBindVertexArray(_VAO);
    vertices = _scene->vertice_num;
    faces = _scene->indices_num;
    glDrawArrays(GL_TRIANGLES, 0, _meshBuffer.size());

    // Unbind.
    glBindVertexArray(0);
}

void Renderer::Render(bool render_again)
{
    // Render objects.
    glm::mat4 view = glm::lookAt(camera_dis * camera_pos, camera_dir, camera_up);
    glm::mat4 model = glm::mat4(1.0f);
    Shader shader = ResourceManager::GetShader("prt");
    shader.Use();
    shader.SetMatrix4("model", model);
    shader.SetMatrix4("view", view);
    shader.SetMatrix4("projection", projection);

    if (render_again)setupBuffer(0, camera_dis*camera_pos);

    objDraw();

    //std::cout << "Render done" << std::endl;

    if (drawCubemap)
    {
        // Render cubemap.
        shader = ResourceManager::GetShader("cubemap");
        shader.Use();
        // Remove translation from the view matrix.
        view = glm::mat4(glm::mat3(view));
        shader.SetMatrix4("view", view);
        shader.SetMatrix4("projection", projection);
        hdrTextures[lightingIndex].Draw();
    }
}
