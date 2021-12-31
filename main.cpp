#include <cmath>
#include <random>

#include <iostream>
#include <vector>

constexpr float PARTICLE_RADIUS = 0.01f;
constexpr float GRAVITATIONAL_ACCELERATION = 9.81f;
constexpr float BOX_SIDE_LENGTH = 2.0f;
constexpr size_t PARTICLE_COUNT = 50;

struct Vector2 {
    float x;
    float y;

    Vector2(float x, float y) : x(x), y(y) {
    }

    Vector2() : x(0.0f), y(0.0f) {
    }

    float norm() {
        return std::sqrt(x * x + y * y);
    }
};

Vector2 operator*(const float a, const Vector2& u) {
    return Vector2(a * u.x, a * u.y);
}

Vector2 operator+(const Vector2& u, const Vector2& v) {
    return Vector2(u.x + v.x, u.y + v.y);
}

Vector2 operator-(const Vector2& u, const Vector2& v) {
    return Vector2(u.x - v.x, u.y - v.y);
}

struct Particle {
    static std::default_random_engine generator;
    static std::uniform_real_distribution<float> distribution;

    float mass;
    Vector2 position;
    Vector2 velocity;
    Vector2 force;

  public:
    Particle(float mass, Vector2 position, Vector2 velocity)
        : mass(mass), position(position), velocity(velocity), force(Vector2()) {
    }

    static Particle create_random() {
        return Particle(
            distribution(generator), Vector2(distribution(generator), distribution(generator)),
            Vector2(distribution(generator), distribution(generator)));
    }

    void clear_force() {
        force.x = 0;
        force.y = 0;
    }

    static float distance(Particle& p1, Particle& p2) {
        return (p1.position - p2.position).norm();
    }
};

std::default_random_engine Particle::generator;
std::uniform_real_distribution<float> Particle::distribution(0.0f, BOX_SIDE_LENGTH);

void add_std_vectors(std::vector<float>& v, const std::vector<float>& u) {
    for(size_t i = 0; i < v.size() && i < u.size(); ++i) {
        v[i] += u[i];
    }
}

class ParticleSystem {
    std::vector<Particle> particles;
    std::vector<float> state;
    std::vector<float> derivative;
    float clock;
    float time_delta;

  public:
    ParticleSystem(std::vector<Particle>& particles) {
        this->particles = particles;
        state.reserve(particles.size());
        derivative.reserve(particles.size());
        clock = 0.0f;
        time_delta = 0.01f;
    }

    void progress() {
        calculate_derivative();
        scale_derivative();
        calculate_state();
        add_std_vectors(state, derivative);
        restore_state();
        clock += time_delta;
    }

  private:
    void clear_forces() {
        for(Particle& particle : particles) {
            particle.clear_force();
        }
    }

    void compute_gravity() {
        for(Particle& particle : particles) {
            particle.force.y += particle.mass * GRAVITATIONAL_ACCELERATION;
        }
    }

    void compute_particle_collisions() {
        for(size_t i = 0; i < particles.size(); ++i) {
            for(size_t j = 0; j < i; ++j) {
                if(Particle::distance(particles[i], particles[j]) <= 2 * PARTICLE_RADIUS) {
                    float& m1 = particles[i].mass;
                    float& m2 = particles[j].mass;
                    Vector2& v1 = particles[i].velocity;
                    Vector2& v2 = particles[j].velocity;

                    particles[i].velocity = (m1 - m2) / (m1 + m2) * v1 + 2 * m2 / (m1 + m2) * v2;
                    particles[j].velocity = 2 * m1 / (m1 - m2) * v1 + (m2 - m1) / (m1 + m2) * v2;
                }
            }
        }
    }

    void compute_border_collisions() {
        for(Particle& particle : particles) {
            if(particle.position.x <= 0.0f) {
                particle.velocity.x *= -1.0f;
                particle.position.x *= -1.0f;
            } else if(particle.position.y <= 0.0f) {
                particle.velocity.y *= -1.0f;
                particle.position.y *= -1.0f;
            } else if(particle.position.x >= BOX_SIDE_LENGTH) {
                particle.velocity.x *= -1.0f;
                particle.position.x -= 2.0f * (particle.position.x - BOX_SIDE_LENGTH);
            } else if(particle.position.y >= BOX_SIDE_LENGTH) {
                particle.velocity.y *= -1.0f;
                particle.position.y -= 2.0f * (particle.position.y - BOX_SIDE_LENGTH);
            }
        }
    }

    void compute_forces() {
        compute_gravity();
        compute_particle_collisions();
        compute_border_collisions();
    }

    void calculate_derivative() {
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

    void scale_derivative() {
        for(float& coord : derivative) {
            coord *= time_delta;
        }
    }

    void calculate_state() {
        state.clear();

        for(Particle& particle : particles) {
            state.push_back(particle.position.x);
            state.push_back(particle.position.y);
            state.push_back(particle.velocity.x);
            state.push_back(particle.velocity.y);
        }
    }

    void restore_state() {
        for(size_t i = 0; i < particles.size(); ++i) {
            particles[i].position.x = state[4 * i + 0];
            particles[i].position.y = state[4 * i + 1];
            particles[i].velocity.x = state[4 * i + 2];
            particles[i].velocity.y = state[4 * i + 3];
        }
    }
};

int main() {
    std::vector<Particle> particles;
    particles.reserve(PARTICLE_COUNT);

    for(size_t i = 0; i < PARTICLE_COUNT; ++i) {
        particles.push_back(Particle::create_random());
    }

    ParticleSystem particle_system(particles);

    for(;;) {
        particle_system.progress();
    }
}
