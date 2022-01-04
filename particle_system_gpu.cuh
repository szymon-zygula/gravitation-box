#ifndef PARTICLE_SYSTEM_GPU_CUH
#define PARTICLE_SYSTEM_GPU_CUH

#include <cmath>

#include <random>
#include <vector>

#include "constants.cuh"

struct ParticleSystemCuda {
    float* d_masses;
    float* d_positions_x;
    float* d_positions_y;
    float* d_velocities_x;
    float* d_velocities_y;
    float* d_temp_velocities_x;
    float* d_temp_velocities_y;
    float* d_forces_x;
    float* d_forces_y;
    float* d_derivative;
    float* d_state;

  public:
    static __host__ void randomize_system(ParticleSystemCuda& particle_system);
    __host__ void destroy();
    __device__ void progress_1(size_t idx);
    __device__ void progress_2(size_t idx);
    __device__ void progress_3(size_t idx);
    __device__ float get_position_x(size_t idx);
    __device__ float get_position_y(size_t idx);

  private:
    __device__ void clear_force(size_t idx);
    __device__ void compute_gravity(size_t idx);
    __device__ void compute_particle_collisions_1(size_t idx);
    __device__ void compute_particle_collisions_2(size_t idx);
    __device__ void compute_border_collisions(size_t idx);
    __device__ void compute_force_1(size_t idx);
    __device__ void compute_force_2(size_t idx);
    __device__ void compute_force_3(size_t idx);
    __device__ void calculate_derivative_1(size_t idx);
    __device__ void calculate_derivative_2(size_t idx);
    __device__ void calculate_derivative_3(size_t idx);
    __device__ void scale_derivative(size_t idx);
    __device__ void calculate_state(size_t idx);
    __device__ void add_derivative_to_state(size_t idx);
    __device__ void restore_state(size_t idx);
};

__global__ void progress_system_1(ParticleSystemCuda particle_system);
__global__ void progress_system_2(ParticleSystemCuda particle_system);
__global__ void progress_system_3(ParticleSystemCuda particle_system, float4* positions);

#endif
