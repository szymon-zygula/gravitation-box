#include "particle_system_gpu.cuh"

__host__ void ParticleSystemCuda::randomize_system(ParticleSystemCuda& particle_system) {
    size_t mem_size = sizeof(float) * PARTICLE_COUNT;
    cudaMalloc(&particle_system.d_masses, mem_size);
    cudaMalloc(&particle_system.d_positions_x, mem_size);
    cudaMalloc(&particle_system.d_positions_y, mem_size);
    cudaMalloc(&particle_system.d_velocities_x, mem_size);
    cudaMalloc(&particle_system.d_velocities_y, mem_size);
    cudaMalloc(&particle_system.d_temp_velocities_x, mem_size);
    cudaMalloc(&particle_system.d_temp_velocities_y, mem_size);
    cudaMalloc(&particle_system.d_forces_x, mem_size);
    cudaMalloc(&particle_system.d_forces_y, mem_size);
    cudaMalloc(&particle_system.d_derivative, mem_size * 4);
    cudaMalloc(&particle_system.d_state, mem_size * 4);

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    std::vector<float> h_masses(PARTICLE_COUNT);
    std::vector<float> h_positions_x(PARTICLE_COUNT);
    std::vector<float> h_positions_y(PARTICLE_COUNT);
    std::vector<float> h_velocities_x(PARTICLE_COUNT);
    std::vector<float> h_velocities_y(PARTICLE_COUNT);

    for(size_t i = 0; i < PARTICLE_COUNT; ++i) {
        h_masses[i] = distribution(generator);
        h_positions_x[i] = distribution(generator) * BOX_SIDE_LENGTH;
        h_positions_y[i] = distribution(generator) * BOX_SIDE_LENGTH;
        h_velocities_x[i] = distribution(generator) - 0.5f;
        h_velocities_y[i] = distribution(generator) - 0.5f;
    }

    cudaMemcpy(particle_system.d_masses, h_masses.data(), mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(
        particle_system.d_positions_x, h_positions_x.data(), mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(
        particle_system.d_positions_y, h_positions_y.data(), mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(
        particle_system.d_velocities_x, h_velocities_x.data(), mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(
        particle_system.d_velocities_y, h_positions_y.data(), mem_size, cudaMemcpyHostToDevice);
}

__device__ float ParticleSystemCuda::get_position_x(size_t idx) {
    return d_positions_x[idx];
}

__device__ float ParticleSystemCuda::get_position_y(size_t idx) {
    return d_positions_y[idx];
}

__device__ void ParticleSystemCuda::progress_1(size_t idx) {
    calculate_derivative_1(idx);
}

__device__ void ParticleSystemCuda::progress_2(size_t idx) {
    calculate_derivative_2(idx);
    scale_derivative(idx);
    calculate_state(idx);
    add_derivative_to_state(idx);
    restore_state(idx);
}

__device__ void ParticleSystemCuda::clear_force(size_t idx) {
    d_forces_x[idx] = 0.0f;
    d_forces_y[idx] = 0.0f;
}

__device__ void ParticleSystemCuda::compute_gravity(size_t idx) {
    d_forces_y[idx] += d_masses[idx] * GRAVITATIONAL_ACCELERATION;
}

__device__ void ParticleSystemCuda::compute_particle_collisions_1(size_t idx) {
    d_temp_velocities_x[idx] = d_velocities_x[idx];
    d_temp_velocities_y[idx] = d_velocities_y[idx];

    for(size_t i = 0; i < PARTICLE_COUNT; ++i) {
        if(i == idx) {
            continue;
        }

        float diff_x = d_positions_x[idx] - d_positions_x[i];
        float diff_y = d_positions_y[idx] - d_positions_y[i];
        float distance_squared = diff_x * diff_x + diff_y * diff_y;
        if(distance_squared <= 4 * PARTICLE_RADIUS * PARTICLE_RADIUS) {
            float m1 = d_masses[idx];
            float m2 = d_masses[i];
            float v1_x = d_velocities_x[idx];
            float v1_y = d_velocities_y[idx];
            float v2_x = d_velocities_x[i];
            float v2_y = d_velocities_y[i];

            d_temp_velocities_x[idx] =
                ((m1 - m2) / (m1 + m2)) * v1_x + (2.0f * m2 / (m1 + m2)) * v2_x;
            d_temp_velocities_y[idx] =
                ((m1 - m2) / (m1 + m2)) * v1_y + (2.0f * m2 / (m1 + m2)) * v2_y;

            break;
        }
    }
}

__device__ void ParticleSystemCuda::compute_particle_collisions_2(size_t idx) {
    d_velocities_x[idx] = d_temp_velocities_x[idx];
    d_velocities_y[idx] = d_temp_velocities_y[idx];
}

__device__ void ParticleSystemCuda::compute_border_collisions(size_t idx) {
    if(d_positions_x[idx] <= 0.0f) {
        d_velocities_x[idx] *= BOX_COLLISION_MULTIPLIER;
        d_positions_x[idx] *= -1.0f;
    } else if(d_positions_y[idx] <= 0.0f) {
        d_velocities_y[idx] *= BOX_COLLISION_MULTIPLIER;
        d_positions_y[idx] *= -1.0f;
    } else if(d_positions_x[idx] >= BOX_SIDE_LENGTH) {
        d_velocities_x[idx] *= BOX_COLLISION_MULTIPLIER;
        d_positions_x[idx] -= 2.0f * (d_positions_x[idx] - BOX_SIDE_LENGTH);
    } else if(d_positions_y[idx] >= BOX_SIDE_LENGTH) {
        d_velocities_y[idx] *= BOX_COLLISION_MULTIPLIER;
        d_positions_y[idx] -= 2.0f * (d_positions_y[idx] - BOX_SIDE_LENGTH);
    }
}

__device__ void ParticleSystemCuda::compute_force_1(size_t idx) {
    compute_gravity(idx);
    compute_border_collisions(idx);
    compute_particle_collisions_1(idx);
}

__device__ void ParticleSystemCuda::compute_force_2(size_t idx) {
    compute_particle_collisions_2(idx);
}

__device__ void ParticleSystemCuda::calculate_derivative_1(size_t idx) {
    clear_force(idx);
    compute_force_1(idx);
}

__device__ void ParticleSystemCuda::calculate_derivative_2(size_t idx) {
    compute_force_2(idx);

    size_t idx4 = 4 * idx;
    d_derivative[idx4 + 0] = d_velocities_x[idx];
    d_derivative[idx4 + 1] = d_velocities_y[idx];
    d_derivative[idx4 + 2] = d_forces_x[idx] / d_masses[idx];
    d_derivative[idx4 + 3] = d_forces_y[idx] / d_masses[idx];
}

__device__ void ParticleSystemCuda::scale_derivative(size_t idx) {
    size_t idx4 = 4 * idx;
    d_derivative[idx4 + 0] *= TIME_DELTA;
    d_derivative[idx4 + 1] *= TIME_DELTA;
    d_derivative[idx4 + 2] *= TIME_DELTA;
    d_derivative[idx4 + 3] *= TIME_DELTA;
}

__device__ void ParticleSystemCuda::calculate_state(size_t idx) {
    size_t idx4 = 4 * idx;
    d_state[idx4 + 0] = d_positions_x[idx];
    d_state[idx4 + 1] = d_positions_y[idx];
    d_state[idx4 + 2] = d_velocities_x[idx];
    d_state[idx4 + 3] = d_velocities_y[idx];
}

__device__ void ParticleSystemCuda::add_derivative_to_state(size_t idx) {
    size_t idx4 = 4 * idx;
    d_state[idx4 + 0] += d_derivative[idx4 + 0];
    d_state[idx4 + 1] += d_derivative[idx4 + 1];
    d_state[idx4 + 2] += d_derivative[idx4 + 2];
    d_state[idx4 + 3] += d_derivative[idx4 + 3];
}

__device__ void ParticleSystemCuda::restore_state(size_t idx) {
    size_t idx4 = 4 * idx;
    d_positions_x[idx] = d_state[idx4 + 0];
    d_positions_y[idx] = d_state[idx4 + 1];
    d_velocities_x[idx] = d_state[idx4 + 2];
    d_velocities_y[idx] = d_state[idx4 + 3];
}

__global__ void progress_system_1(ParticleSystemCuda particle_system) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= PARTICLE_COUNT) {
        return;
    }

    particle_system.progress_1(idx);
}

__global__ void progress_system_2(ParticleSystemCuda particle_system, float4* positions) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= PARTICLE_COUNT) {
        return;
    }

    particle_system.progress_2(idx);
    positions[idx] = make_float4(
        particle_system.get_position_x(idx) - 1.0f, particle_system.get_position_y(idx) - 1.0f,
        1.0f, 1.0f);
}

void ParticleSystemCuda::destroy() {
    cudaFree(d_masses);
    cudaFree(d_positions_x);
    cudaFree(d_positions_y);
    cudaFree(d_velocities_x);
    cudaFree(d_velocities_y);
    cudaFree(d_temp_velocities_x);
    cudaFree(d_temp_velocities_y);
    cudaFree(d_forces_x);
    cudaFree(d_forces_y);
    cudaFree(d_derivative);
    cudaFree(d_state);
}
