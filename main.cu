#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <GL/freeglut.h>
#include <helper_gl.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

// Utilities and timing functions
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <helper_functions.h> // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h> // helper functions for CUDA error check

#include <vector_types.h>

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD 0.30f
#define REFRESH_DELAY 10 // ms

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 512;
const unsigned int window_height = 512;

constexpr unsigned int PARTICLE_COUNT = 256;
constexpr unsigned int BLOCK_SIZE = std::min(1024u, PARTICLE_COUNT);
constexpr unsigned int BLOCK_COUNT = PARTICLE_COUNT / BLOCK_SIZE;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource* cuda_vbo_resource;
void* d_vbo_buffer = NULL;

float g_fAnim = 0.0;

StopWatchInterface* timer = NULL;

// Auto-Verification Code
int fpsCount = 0; // FPS count for averaging
int fpsLimit = 1; // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

#define MAX(a, b) ((a > b) ? a : b)

// declaration, forward
bool run_program(int argc, char** argv);
void cleanup();

// GL functionality
bool init_gl(int* argc, char** argv);
void create_vbo(GLuint* vbo, struct cudaGraphicsResource** vbo_res, unsigned int vbo_res_flags);
void delete_vbo(GLuint* vbo, struct cudaGraphicsResource* vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void timer_event(int value);

// Cuda functionality
void run_cuda(struct cudaGraphicsResource** vbo_resource);

const char* sSDKsample = "simpleGL (VBO)";

__global__ void simple_vbo_kernel(float4* pos, float time) {
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    pos[thread_id] = make_float4(-0.9f + thread_id * 0.005f, 0.9f, 1.0f, 1.0f);
}

void launch_kernel(float4* pos, unsigned int particle_count, float time) {
    simple_vbo_kernel<<<BLOCK_SIZE, BLOCK_COUNT>>>(pos, time);
}

int main(int argc, char** argv) {
    setenv("DISPLAY", ":0", 0);
    printf("%s starting...\n", sSDKsample);

    run_program(argc, argv);

    printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void compute_fps() {
    frameCount++;
    fpsCount++;

    if(fpsCount == fpsLimit) {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
    glutSetWindowTitle(fps);
}

bool init_gl(int* argc, char** argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cuda GL Interop (VBO)");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutTimerFunc(REFRESH_DELAY, timer_event, 0);

    // initialize necessary OpenGL extensions
    if(!isGLVersionSupported(2, 0)) {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    SDK_CHECK_ERROR_GL();

    return true;
}

bool run_program(int argc, char** argv) {
    // Create the CUTIL timer
    sdkCreateTimer(&timer);

    // use command-line specified CUDA device, otherwise use device with highest
    // Gflops/s
    int devID = findCudaDevice(argc, (const char**)argv);

    if(!init_gl(&argc, argv)) {
        return false;
    }

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutCloseFunc(cleanup);

    create_vbo(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
    run_cuda(&cuda_vbo_resource);
    glutMainLoop();

    return true;
}

void run_cuda(struct cudaGraphicsResource** vbo_resource) {
    // map OpenGL buffer object for writing from CUDA
    float4* dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, *vbo_resource));

    launch_kernel(dptr, PARTICLE_COUNT, g_fAnim);

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

#ifndef FOPEN
#define FOPEN(fHandle, filename, mode) (fHandle = fopen(filename, mode))
#endif

void sdkDumpBin2(void* data, unsigned int bytes, const char* filename) {
    printf("sdkDumpBin: <%s>\n", filename);
    FILE* fp;
    FOPEN(fp, filename, "wb");
    fwrite(data, bytes, 1, fp);
    fflush(fp);
    fclose(fp);
}

void create_vbo(GLuint* vbo, struct cudaGraphicsResource** vbo_res, unsigned int vbo_res_flags) {
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = PARTICLE_COUNT * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();
}

void delete_vbo(GLuint* vbo, struct cudaGraphicsResource* vbo_res) {

    // unregister this buffer object with CUDA
    checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

void display() {
    sdkStartTimer(&timer);

    // run CUDA kernel to generate vertex positions
    run_cuda(&cuda_vbo_resource);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_POINTS, 0, PARTICLE_COUNT);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();

    g_fAnim += 0.01f;

    sdkStopTimer(&timer);
    compute_fps();
}

void timer_event(int value) {
    if(glutGetWindow()) {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timer_event, 0);
    }
}

void cleanup() {
    sdkDeleteTimer(&timer);

    if(vbo) {
        delete_vbo(&vbo, cuda_vbo_resource);
    }
}

void keyboard(unsigned char key, int /*x*/, int /*y*/) {
    switch(key) {
    case(27):
        glutDestroyWindow(glutGetWindow());
        return;
    }
}
