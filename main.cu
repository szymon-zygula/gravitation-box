#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <memory>

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

#include "constants.cuh"
#include "particle_system_cpu.cuh"
#include "particle_system_gpu.cuh"

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD 0.30f
#define REFRESH_DELAY 10 // ms

// vbo variables
GLuint vbo;
struct cudaGraphicsResource* cuda_vbo_resource;
void* d_vbo_buffer = NULL;

float g_f_anim = 0.0;

StopWatchInterface* timer = NULL;

// Auto-Verification Code
int fps_count = 0; // FPS count for averaging
int fps_limit = 1; // FPS limit for sampling
int g_Index = 0;
float average_fps = 0.0f;
size_t frame_count = 0;
size_t g_total_errors = 0;
bool g_b_qa_eadback = false;

std::unique_ptr<ParticleSystem> particle_system;
std::unique_ptr<ParticleSystemCuda> cuda_particle_system;

// declaration, forward
bool run_program(int argc, char** argv);
void cleanup();

// GL functionality
bool init_gl(int* argc, char** argv);
void create_vbo(GLuint* vbo, struct cudaGraphicsResource** vbo_res, size_t vbo_res_flags);
void delete_vbo(GLuint* vbo, struct cudaGraphicsResource* vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void timer_event(int value);

void do_calculations(struct cudaGraphicsResource** vbo_resource);
void run_cuda(struct cudaGraphicsResource** vbo_resource);
void run_cpu();

const char* sSDKsample = "simpleGL (VBO)";

int main(int argc, char** argv) {
    setenv("DISPLAY", ":0", 0);
    printf("%s starting...\n", sSDKsample);

    run_program(argc, argv);

    printf("%s completed, returned %s\n", sSDKsample, (g_total_errors == 0) ? "OK" : "ERROR!");
    exit(g_total_errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void compute_fps() {
    frame_count++;
    fps_count++;

    if(fps_count == fps_limit) {
        average_fps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fps_count = 0;
        fps_limit = (int)std::max(average_fps, 1.0f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "Gravitation Box: %3.1f fps", average_fps);
    glutSetWindowTitle(fps);
}

bool init_gl(int* argc, char** argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
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
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    SDK_CHECK_ERROR_GL();

    return true;
}

bool run_program(int argc, char** argv) {
    // Create the CUTIL timer
    sdkCreateTimer(&timer);

    // Create data
    std::vector<Particle> particles;
    particles.reserve(PARTICLE_COUNT);
    for(size_t i = 0; i < PARTICLE_COUNT; ++i) {
        particles.push_back(Particle::create_random());
    }

    if(GPU_ACCELERATION) {
        cuda_particle_system = std::make_unique<ParticleSystemCuda>();
        ParticleSystemCuda::randomize_system(*cuda_particle_system);
    } else {
        particle_system = std::make_unique<ParticleSystem>(particles);
    }

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
    glutMainLoop();

    cuda_particle_system->destroy();

    return true;
}

void do_calculations(struct cudaGraphicsResource** vbo_resource) {
    if(GPU_ACCELERATION) {
        run_cuda(vbo_resource);
    } else {
        run_cpu();
    }
}

void run_cuda(struct cudaGraphicsResource** vbo_resource) {
    // map OpenGL buffer object for writing from CUDA
    float4* dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, *vbo_resource));

    progress_system_1<<<BLOCK_COUNT, BLOCK_SIZE>>>(*cuda_particle_system);
    cudaDeviceSynchronize();
    progress_system_2<<<BLOCK_COUNT, BLOCK_SIZE>>>(*cuda_particle_system, dptr);

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

void run_cpu() {
    particle_system->progress();

    float4 particle_positions[PARTICLE_COUNT];

    size_t i = 0;
    for(const Particle& particle : particle_system->get_particles().get()) {
        particle_positions[i] =
            make_float4(particle.position.x - 1.0f, particle.position.y - 1.0f, 1.0f, 1.0f);
        i += 1;
    }

    glBufferData(
        GL_ARRAY_BUFFER, PARTICLE_COUNT * sizeof(float4), particle_positions, GL_DYNAMIC_DRAW);
}

#ifndef FOPEN
#define FOPEN(fHandle, filename, mode) (fHandle = fopen(filename, mode))
#endif

void sdkDumpBin2(void* data, size_t bytes, const char* filename) {
    printf("sdkDumpBin: <%s>\n", filename);
    FILE* fp;
    FOPEN(fp, filename, "wb");
    fwrite(data, bytes, 1, fp);
    fflush(fp);
    fclose(fp);
}

void create_vbo(GLuint* vbo, struct cudaGraphicsResource** vbo_res, size_t vbo_res_flags) {
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    size_t size = PARTICLE_COUNT * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);

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
    do_calculations(&cuda_vbo_resource);

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

    g_f_anim += 0.01f;

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
