#include "particle_system.cuh"

Vector2::Vector2(float x, float y) : x(x), y(y) {
}

Vector2::Vector2() : x(0.0f), y(0.0f) {
}

float Vector2::norm() {
    return std::sqrt(x * x + y * y);
}

Vector2 operator*(const float a, const Vector2& u) {
    return Vector2(a * u.x, a * u.y);
}

Vector2 operator*(const Vector2& u, const float a) {
    return a * u;
}

Vector2 operator+(const Vector2& u, const Vector2& v) {
    return Vector2(u.x + v.x, u.y + v.y);
}

Vector2 operator-(const Vector2& u, const Vector2& v) {
    return Vector2(u.x - v.x, u.y - v.y);
}

Particle::Particle(float mass, Vector2 position, Vector2 velocity)
    : mass(mass), position(position), velocity(velocity), force(Vector2()) {
}

Particle Particle::create_random() {
    return Particle(
        distribution(generator),
        Vector2(distribution(generator), distribution(generator)) * ParticleSystem::BOX_SIDE_LENGTH,
        Vector2(distribution(generator) - 0.5f, distribution(generator) - 0.5f));
}

void Particle::clear_force() {
    force.x = 0;
    force.y = 0;
}

float Particle::distance(Particle& p1, Particle& p2) {
    return (p1.position - p2.position).norm();
}

float ParticleSystem::BOX_COLLISION_MULTIPLIER = -0.5f;
float ParticleSystem::BOX_SIDE_LENGTH = 2.0f;
std::default_random_engine Particle::generator;
std::uniform_real_distribution<float> Particle::distribution(0.0f, 1.0f);

void add_std_vectors(std::vector<float>& v, const std::vector<float>& u) {
    for(size_t i = 0; i < v.size() && i < u.size(); ++i) {
        v[i] += u[i];
    }
}

ParticleSystem::ParticleSystem(
    std::vector<Particle>& particles, float particle_radius, float gravitational_acceleration)
    : particle_radius(particle_radius), gravitational_acceleration(gravitational_acceleration) {
    this->particles = particles;
    state.reserve(particles.size());
    derivative.reserve(particles.size());
    clock = 0.0f;
    // TODO: Calculate delta
    time_delta = 0.01f;
}

void ParticleSystem::progress() {
    calculate_derivative();
    scale_derivative();
    calculate_state();
    add_std_vectors(state, derivative);
    restore_state();
    // TODO: Calculate delta
    clock += time_delta;
}

std::reference_wrapper<const std::vector<Particle>> ParticleSystem::get_particles() const {
    return std::cref(particles);
}

void ParticleSystem::clear_forces() {
    for(Particle& particle : particles) {
        particle.clear_force();
    }
}

void ParticleSystem::compute_gravity() {
    for(Particle& particle : particles) {
        particle.force.y += particle.mass * gravitational_acceleration;
    }
}

void ParticleSystem::compute_particle_collisions() {
    for(size_t i = 0; i < particles.size(); ++i) {
        for(size_t j = 0; j < i; ++j) {
            if(Particle::distance(particles[i], particles[j]) <= 2 * particle_radius) {
                float m1 = particles[i].mass;
                float m2 = particles[j].mass;
                Vector2 v1 = particles[i].velocity;
                Vector2 v2 = particles[j].velocity;

                particles[i].velocity = ((m1 - m2) / (m1 + m2)) * v1 + (2.0f * m2 / (m1 + m2)) * v2;
                particles[j].velocity = (2.0f * m1 / (m1 + m2)) * v1 + ((m2 - m1) / (m1 + m2)) * v2;
            }
        }
    }
}

void ParticleSystem::compute_border_collisions() {
    for(Particle& particle : particles) {
        if(particle.position.x <= 0.0f) {
            particle.velocity.x *= BOX_COLLISION_MULTIPLIER;
            particle.position.x *= -1.0f;
        } else if(particle.position.y <= 0.0f) {
            particle.velocity.y *= BOX_COLLISION_MULTIPLIER;
            particle.position.y *= -1.0f;
        } else if(particle.position.x >= BOX_SIDE_LENGTH) {
            particle.velocity.x *= BOX_COLLISION_MULTIPLIER;
            particle.position.x -= 2.0f * (particle.position.x - BOX_SIDE_LENGTH);
        } else if(particle.position.y >= BOX_SIDE_LENGTH) {
            particle.velocity.y *= BOX_COLLISION_MULTIPLIER;
            particle.position.y -= 2.0f * (particle.position.y - BOX_SIDE_LENGTH);
        }
    }
}

void ParticleSystem::compute_forces() {
    compute_gravity();
    compute_particle_collisions();
    compute_border_collisions();
}

void ParticleSystem::calculate_derivative() {
    clear_forces();
    compute_forces();
    derivative.clear();

    for(Particle& particle : particles) {
        derivative.push_back(particle.velocity.x);
        derivative.push_back(particle.velocity.y);
        derivative.push_back(particle.force.x / particle.mass);
        derivative.push_back(particle.force.y / particle.mass);
    }
}

void ParticleSystem::scale_derivative() {
    for(float& coord : derivative) {
        coord *= time_delta;
    }
}

void ParticleSystem::calculate_state() {
    state.clear();

    for(Particle& particle : particles) {
        state.push_back(particle.position.x);
        state.push_back(particle.position.y);
        state.push_back(particle.velocity.x);
        state.push_back(particle.velocity.y);
    }
}

void ParticleSystem::restore_state() {
    for(size_t i = 0; i < particles.size(); ++i) {
        particles[i].position.x = state[4 * i + 0];
        particles[i].position.y = state[4 * i + 1];
        particles[i].velocity.x = state[4 * i + 2];
        particles[i].velocity.y = state[4 * i + 3];
    }
}
