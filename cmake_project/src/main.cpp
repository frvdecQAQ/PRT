#include <ctime>
#include <string>
#include <iostream>
#include <fstream>
#include <io.h>
#include <cmath>

// GLEW
// #define GLEW_STATIC
#include <GL/glew.h>
// GLFW
#include <GLFW/glfw3.h>
// GLM
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
// AntTweakBar
#include <AntTweakBar.h>

#include "UI.h"
#include "brdf.h"
#include "scene.h"
#include "lighting.h"
#include "renderer.h"
#include "diffuseObject.h"
#include "generalObject.h"
#include "resource_manager.h"
#include "fftprecomputed.hpp"
#include "shRotateMatrix.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
//#define FULL_SCREEN

// Window size.
int WIDTH, HEIGHT;
GLuint width = 1280, height = 768;

// Keyboard.
bool keys[1024];

// Mouse.
GLfloat lastX = WIDTH / 2.0f, lastY = HEIGHT / 2.0f;
GLfloat yaw = -90.f, pitch = 0.0f;
bool firstMouse = true;

const int LightNumber = 5;
int lightingIndex;

std::string objects[] = {"buddha", "maxplanck"};
std::string gobjects[] = {"buddha", "maxplanck"};
std::string lightings[] = {"galileo", "grace", "rnl", "stpeters", "uffizi"};
std::string bands[] = {"linear", "quadratic", "cubic", "quartic"};
BRDF_TYPE BRDFs[] = {BRDF_PHONG, BRDF_WARD_ISOTROPIC, BRDF_WARD_ANISOTROPIC};

glm::vec3 albedo(0.12f, 0.12f, 0.12f);

Scene scene;
DiffuseObject diffObject;
GeneralObject genObject;
Lighting now_light;
BRDF brdf;
Renderer renderer;

int batchsize = 4096;
int sampleNumber = 128 * 128;
int band = 5;
int sphereNumber = 32;
int shadowSampleNumber = 48 * 48;

// Cubemap & Simple Light.
bool drawCubemap = false;
bool simpleLight = true;
bool lastSimple = true;
bool renderbar = true;
// Camera.

float camera_dis = 2.9f;
float fov = 45.0f;
glm::vec3 camera_pos(-0.326521, 0.319368, -0.889599);
glm::vec3 last_camera_pos(0.0f, 0.0f, 1.0f);
glm::vec3 camera_dir(0.0f, 0.0f, 0.0f);
glm::vec3 camera_up(0.0f, 1.0f, 0.0f);
glm::vec3 camera_front(0.0f, 0.0f, -1.0f);

//Simple lighting.
glm::vec3 light_dir(0.0f, 0.0f, 1.0f);
glm::vec3 last_light_dir(0.0f, 0.0f, 1.0f);

// Rotation.
glm::fquat g_Rotation(0.0f, 0.0f, 0.0f, 1.0f);
glm::fquat last_Rotation(0.0f, 0.0f, 0.0f, 1.0f);
glm::fquat g_RotateStart(0.0f, 0.0f, 0.0f, 1.0f);
glm::mat4 rotateMatrix;
int g_AutoRotate = 0;
int g_RotateTime = 0;

// FPS.
double currTime;
double lastTime;
int frames = 0;
int fps;

// GLFW
GLFWwindow* window;

/*
 *  Function prototypes.
 */
// Callback functions.
int key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
int button_calback(GLFWwindow* window, int button, int action, int mods);
int mouse_callback(GLFWwindow* window, double xpos, double ypos);
int scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
// Data processing function.
void dataProcessing(int argc, char** argv);
// Initialization functions.
void dataLoading(int argc, char** argv);
void shaderLoading();
// Miscellaneous.
void calculateFPS();

void saveImage(char* filepath, GLFWwindow* w) {
	int width, height;
	glfwGetFramebufferSize(w, &width, &height);
	GLsizei nrChannels = 3;
	GLsizei stride = nrChannels * width;
	stride += (stride % 4) ? (4 - stride % 4) : 0;
	GLsizei bufferSize = stride * height;
	std::vector<char> buffer(bufferSize);
	glPixelStorei(GL_PACK_ALIGNMENT, 4);
	glReadBuffer(GL_FRONT);
	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer.data());
	stbi_flip_vertically_on_write(true);
	stbi_write_png(filepath, width, height, nrChannels, buffer.data(), stride);
    FILE* pos_file = fopen("pos.txt", "w");
    fprintf(pos_file, "%lf %lf %lf\n", camera_pos[0], camera_pos[1], camera_pos[2]);
    fprintf(pos_file, "%lf\n", camera_dis);
    fclose(pos_file);
}


int main(int argc, char** argv){
    // ./PRT.exe -s/-l -d/-g [band] [sample number] [sphereNumber] [shadowSampleNumber]
    renderer.loadTriple(n);
    initGamma();
    SphericalH::prepare(n);
    dataProcessing(argc, argv);

    std::cout << "check point 0" << std::endl;
    // Init GLFW.
    glfwInit();
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_AUTO_ICONIFY, GL_FALSE);
#endif
     glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
     glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
     glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
     glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
     glfwWindowHint(GLFW_SAMPLES, 4);

#ifdef FULL_SCREEN
     // Create a "Windowed full screen" window in the primary monitor.
     GLFWmonitor* monitor = glfwGetPrimaryMonitor();
     const GLFWvidmode* mode = glfwGetVideoMode(monitor);
     glfwWindowHint(GLFW_RED_BITS, mode->redBits);
     glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
     glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
     glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
     window = glfwCreateWindow(mode->width, mode->height, "prt-SH", monitor, nullptr);
#else
     window = glfwCreateWindow(width, height, "prt-SH", nullptr, nullptr);
#endif
    glfwMakeContextCurrent(window);

    // Note: The required callback functions are set with AntTweakBar UI.

    // Set this to true so GLEW knows to use a modern approach to retrieving function pointers and extensions.
    glewExperimental = GL_TRUE;
    // Initialize GLEW to setup the OpenGL Function pointers.
    glewInit();

    // Define the viewport dimensions.
    glfwGetFramebufferSize(window, &WIDTH, &HEIGHT);
    glViewport(0, 0, WIDTH, HEIGHT);

    width = WIDTH;
    height = HEIGHT;

    std::cout << "GLFW version                : " << glfwGetVersionString() << std::endl;
    std::cout << "GL_VERSION                  : " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GL_VENDOR                   : " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "GL_RENDERER                 : " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "GL_SHADING_LANGUAGE_VERSION : " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    std::cout << "WINDOW WIDTH                : " << WIDTH << std::endl;
    std::cout << "WINDOW HEIGHT               : " << HEIGHT << std::endl;

    // Setup some OpenGL options.
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glDepthFunc(GL_LESS);
    // Anti-aliasing.
    glEnable(GL_MULTISAMPLE);
    glfwSwapInterval(0);

    // Do some initialization (including loading data, shaders, models, etc.)
    std::cout << "load start" << std::endl;
    dataLoading(argc, argv);
    std::cout << "load done" << std::endl;
    //scene.prepareData(30);
    //std::cout << "prepare done" << std::endl;
    //return 0;
    shaderLoading();
    renderer.Init(LightNumber);
    UIInit();
    lastTime = glfwGetTime();

    int translate_cnt = 0;
    bool translate_flag = true;

    //scene.change();

    // Loop.

    std::cout << "======================Render Start====================" << std::endl;
    int render_cnt = 0;
    double render_time = 0;
    while (!glfwWindowShouldClose(window))
    {
        std::cout << " ============================ " << std::endl;
        double start_time = glfwGetTime();
        // Calculate FPS.
        calculateFPS();

        // Check if any events have been activated (key pressed, mouse moved etc.) and call corresponding response functions.
        glfwPollEvents();

        // Clear the color buffer.
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // Render.
        //int f;
        //scanf("%d", &f);
        bool render_again = true;
        renderer.Render(render_again);

        //0.0106241
        //0.00500424

        // Render AntTweakBar UI.
        if(renderbar)
        TwDraw();

        // Swap the screen buffers.
        glfwSwapBuffers(window);
        double end_time = glfwGetTime();
        std::cout << "time = " << end_time - start_time << std::endl;
        render_cnt++;
        render_time += (end_time-start_time);
        std::cout << "avg = " << render_time/render_cnt << std::endl;


        /*GLubyte* pPixelData;
        const int picWidth = 1920;
        const int picHeight = 1080;
        pPixelData = (GLubyte*)malloc(picWidth*picHeight*4);

        if(pPixelData == 0)return 0;
        glReadBuffer(GL_FRONT);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        glReadPixels(0, 0, picWidth, picHeight, GL_RGBA, GL_UNSIGNED_BYTE, pPixelData);

        cv::Mat imgPlanes[3];
        cv::Mat img = cv::M

        for(int i = 0; i < picHeight; ++i){
            unsigned char* plane0Ptr = imgPlanes[0].ptr<unsigned char>(i);
            unsigned char* plane1Ptr = imgPlanes[1].ptr<unsigned char>(i);
            unsigned char* plane2Ptr = imgPlanes[2].ptr<unsigned char>(i);
            for(int j = 0; j < picWidth; ++j){
                int k = 4*(i*picWidtj);
                plane2Ptr[j] = pPixelData[k];
                plane1Ptr[j] = pPixelData[k+1];
                plane2Ptr[j] = pPixelData[k+2];
            }
        }

        cv::merge(imgPlanes, img);
        cv::flip(img, img, 0);
        cv::imwrite("./opengl_result.jpg", img);*/

    }
    // Terminate GLFW, clearing any resources allocated by GLFW.
    glfwTerminate();
    releaseGamma();
    return 0;
}

void calculateFPS()
{
    // Return current time in seconds.
    currTime = glfwGetTime();
    frames++;

    if ((currTime - lastTime) >= 1.0f)
    {
        fps = frames / (currTime - lastTime);
        frames = 0;
        lastTime = currTime;
    }
}

void dataLoading(int argc, char** argv)
{
    std::cout << "Loading data ................ " << std::endl;

    std::string ptype(argv[1]);
    assert(ptype == "-s" || ptype == "-l");

    std::string diffGeneal(argv[2]);
    assert(diffGeneal == "-d" || diffGeneal == "-g");

    glm::vec3 hdrEffect[] = {
       glm::vec3(1.2f, 1.2f, 1.2f),
       glm::vec3(2.2f, 2.2f, 2.2f),
       glm::vec3(1.2f, 1.2f, 1.2f),
       glm::vec3(1.8f, 1.8f, 1.8f),
       glm::vec3(1.8f, 1.8f, 1.8f)
    };
    glm::vec3 glossyEffect[] = {
        glm::vec3(1.2f, 1.2f, 1.2f),
        glm::vec3(2.2f, 2.2f, 2.2f),
        glm::vec3(1.2f, 1.2f, 1.2f),
        glm::vec3(1.8f, 1.8f, 1.8f),
        glm::vec3(1.8f, 1.8f, 1.8f)
    };

    std::string path(argv[4]);

    if (ptype == "-s")
    {
        std::string save_path = path + "/light_simple.dat";
        now_light.init(save_path, glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f));
    }
    else
    {
        drawCubemap = true;
        std::string read_path(argv[3]);
        size_t beginIndex = read_path.rfind('/');
        size_t endIndex = read_path.rfind('.');
        std::string lighting_effect = read_path.substr(beginIndex + 1, endIndex - beginIndex - 1);
        std::string save_path = path + "/light_" + lighting_effect + ".dat";
        for (int i = 0; i < LightNumber; ++i) {
            if (lightings[i]+"_probe" == lighting_effect) {
                lightingIndex = i;
                now_light.init(save_path, hdrEffect[i], glossyEffect[i]);
                break;
            }
        }
    }

    int tmp_cnt = 0;
    for (int i = 0; i < scene.obj_num; ++i) {
        std::string obj_name = path + "/" + scene.name_list[i];
        size_t beginIndex = obj_name.rfind('/');
        size_t endIndex = obj_name.rfind('.');
        std::string save_path = path + "/" + obj_name.substr(beginIndex + 1, endIndex - beginIndex - 1) + "U.dat";
        Object* tmp;
        if (scene.type_list[i] == 0)tmp = new DiffuseObject();
        else tmp = new GeneralObject();
        tmp->init(obj_name, albedo, scene.scale[tmp_cnt]);
        tmp_cnt++;
        tmp->readFDiskbin(save_path);
        scene.obj_list.push_back(tmp);
    }scene._band = band;
    scene.prepare();

    renderer.Setup(&scene, &now_light);
    renderer.SetupColorBuffer(0, camera_dis * camera_pos, false);

    std::cout << "Loading Done" << std::endl;
}

void shaderLoading()
{
    std::cout << "Loading shaders ............. ";
    ResourceManager::LoadShader(_SHADER_PREFIX_"/cubemap.vert", _SHADER_PREFIX_"/cubemap.frag", "", "cubemap");
    ResourceManager::LoadShader(_SHADER_PREFIX_"/prt.vert", _SHADER_PREFIX_"/prt.frag", "", "prt");
    std::cout << "Done" << std::endl;
}

bool check_light_process(std::string path, int band)
{
    std::ifstream in(path, std::ifstream::binary);
    if (!in)return false;
    int tmp_band;
    in.read((char*)&tmp_band, sizeof(int));
    in.close();
    if (tmp_band == band)return true;
    return false;
}

bool check_obj_process(std::string path, int type, int band, int sample, int sphere, int shadow) 
{
    std::ifstream in(path, std::ifstream::binary);
    if (!in)return false;
    int size, tmp_band, tmp_sample, tmp_sphere, tmp_shadow;
    if (type == 0) {
        in.read((char*)&size, sizeof(int));
        in.read((char*)&tmp_band, sizeof(int));
        in.read((char*)&tmp_sample, sizeof(int));
        in.read((char*)&tmp_sphere, sizeof(int));
        in.read((char*)&tmp_shadow, sizeof(int));
        in.close();
        if (tmp_band != band)return false;
        if (tmp_sample != sample)return false;
        if (tmp_sphere != sphere)return false;
        if (tmp_shadow != shadow)return false;
    }
    else {
        in.read((char*)&size, sizeof(int));
        in.read((char*)&tmp_band, sizeof(int));
        in.close();
        if (tmp_band != band)return false;
    }
    return true;
}

bool check_brdf_process(std::string path, int band) {
    std::ifstream in(path, std::ifstream::binary);
    if (!in)return false;
    int tmp_band;
    in.read((char*)&tmp_band, sizeof(int));
    in.close();
    if (tmp_band != band)return false;
    return true;
}

void dataProcessing(int argc, char** argv)
{
    // ./PRT.exe -s/-l -d/-g xxx.probe xxx.obj [band] [sample number] [sphereNumber] [shadowSampleNumber]

    std::string ptype(argv[1]);
    assert(ptype == "-s" || ptype == "-l");

    std::string diffGeneal(argv[2]);
    assert(diffGeneal == "-d" || diffGeneal == "-g");

    band = n;
    if (argc > 6)sampleNumber = atoi(argv[6]);
    if (argc > 7)sphereNumber = atoi(argv[7]);
    if (argc > 8)shadowSampleNumber = atoi(argv[8]);

    std::string path(argv[4]);
    scene.init(path);
    if (ptype == "-l") 
    {
        std::string read_path(argv[3]);
        size_t beginIndex = read_path.rfind('/');
        size_t endIndex = read_path.rfind('.');
        std::string save_path = path + "/light_" + read_path.substr(beginIndex + 1, endIndex - beginIndex - 1) + ".dat";
        std::cout << "light_save_path = " << save_path << std::endl;
        bool exist_flag = check_light_process(save_path, band);
        if (!exist_flag)
        {
            Lighting pattern(argv[3], PROBE, band);
            std::cout << "!!!" << std::endl;
            pattern.process(sampleNumber, true);
            std::cout << "!!!" << std::endl;
            pattern.write2Diskbin(save_path);
        }
    }
    else
    {
        std::string save_path = path + "/light_simple.dat";
        bool exist_flag = check_light_process(save_path, band);
        if (!exist_flag)
        {
            Lighting simplePattern("", PROBE, band);
            simplePattern.process(sampleNumber, false);
            simplePattern.write2Diskbin(save_path);
        }
    }

    std::cout << "!!!" << std::endl;

    std::vector<std::string>file_name;
    for (int i = 0; i < scene.obj_num; ++i) {
        file_name.push_back(path + "/" + scene.name_list[i]);
    }
 
    int transferType = 2;
    int tmp_cnt = 0;
    std::vector<Object*>obj_list;
    for (auto obj_path : file_name) {
        std::cout << obj_path << std::endl;
        size_t beginIndex = obj_path.rfind('/');
        size_t endIndex = obj_path.rfind('.');
        std::string save_path = path + '/' + obj_path.substr(beginIndex + 1, endIndex - beginIndex - 1) + "U.dat";
        bool exist_flag = check_obj_process(save_path, scene.type_list[tmp_cnt],
            band, sampleNumber, sphereNumber, shadowSampleNumber);
        if (!exist_flag)
        {
            Object* tmp;
            if (scene.type_list[tmp_cnt] == 1)tmp = new GeneralObject();
            else tmp = new DiffuseObject();
            tmp->init(obj_path.c_str(), albedo, scene.scale[tmp_cnt]);
            //tmp->project2SH(transferType, band, sampleNumber, 1);
            //tmp->write2Diskbin(save_path);
            obj_list.push_back(tmp);
        }
        tmp_cnt++;
    }
    for(int i = 0; i < obj_list.size(); ++i){
        std::string obj_path = file_name[i];
        size_t beginIndex = obj_path.rfind('/');
        size_t endIndex = obj_path.rfind('.');
        std::string save_path = path + '/' + obj_path.substr(beginIndex + 1, endIndex - beginIndex - 1) + "U.dat";
        obj_list[i]->project2SH(transferType, band, sampleNumber, 1, obj_list);
        obj_list[i]->write2Diskbin(save_path);
    }
    //fftprecomputed fs_pre;
    //fs_pre.init();
}

// Is called whenever a key is pressed/released via GLFW.
int key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);

    float cameraSpeed = 0.2;
    if (key == GLFW_KEY_W && action == GLFW_PRESS)
        camera_pos += cameraSpeed * camera_front;
    if (key == GLFW_KEY_S && action == GLFW_PRESS)
        camera_pos -= cameraSpeed * camera_front;
    if (key == GLFW_KEY_A && action == GLFW_PRESS)
        camera_pos -= cameraSpeed * glm::normalize(glm::cross((camera_dir - camera_dis * camera_pos), camera_up));
    if (key == GLFW_KEY_D && action == GLFW_PRESS)
        camera_pos += cameraSpeed * glm::normalize(glm::cross((camera_dir - camera_dis * camera_pos), camera_up));
    if (key == GLFW_KEY_K && action == GLFW_PRESS)
        saveImage("./result.png", window);
    if (key == GLFW_KEY_H && action == GLFW_PRESS)
        renderbar = !renderbar;
    
    if (key >= 0 && key < 1024)
    {
        if (action == GLFW_PRESS)
            keys[key] = true;
        else if (action == GLFW_RELEASE)
            keys[key] = false;
    }

    return TwEventKeyGLFW(key, action);
}

int mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    lastX = xpos;
    lastY = ypos;

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.02f; // change this value to your liking
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    // make sure that when pitch is out of bounds, screen doesn't get flipped
    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    camera_front = glm::normalize(front);


    return TwMouseMotion((int)xpos, (int)ypos);
}

// Is called whenever MOUSE_LEFT or MOUSE_RIGHT is pressed/released via GLFW.
int button_calback(GLFWwindow* window, int button, int action, int mods)
{
    return TwEventMouseButtonGLFW(button, action);
}


int scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    //  fov -= (float)yoffset;
    // if (fov < 1.0f)
    //     fov = 1.0f;
    // if (fov > 45.0f)
    //     fov = 45.0f;
    if (yoffset > 0 && camera_dis >= 1.0f)
    {
        camera_dis -= 0.2f;
    }
    else
    {
        camera_dis += 0.2f;
    }
    return TwEventMouseWheelGLFW(yoffset);
}