#include <cmath>

#include <functional>
#include <random>
#include <vector>

struct Vector2 {
    float x;
    float y;

    Vector2(float x, float y);
    Vector2();

    float norm();
};

Vector2 operator*(const float a, const Vector2& u);
Vector2 operator*(const Vector2& u, const float a);
Vector2 operator+(const Vector2& u, const Vector2& v);
Vector2 operator-(const Vector2& u, const Vector2& v);

struct Particle {
    static std::default_random_engine generator;
    static std::uniform_real_distribution<float> distribution;

    float mass;
    Vector2 position;
    Vector2 velocity;
    Vector2 force;

  public:
    Particle(float mass, Vector2 position, Vector2 velocity);

    static Particle create_random();

    void clear_force();

    static float distance(Particle& p1, Particle& p2);
};

void add_std_vectors(std::vector<float>& v, const std::vector<float>& u);

class ParticleSystem {
  private:
    std::vector<Particle> particles;
    std::vector<float> state;
    std::vector<float> derivative;
    float clock;
    float particle_radius;
    float gravitational_acceleration;

  public:
    ParticleSystem(
        std::vector<Particle>& particles, float particle_radius, float gravitational_acceleration);

    void progress();
    std::reference_wrapper<const std::vector<Particle>> get_particles() const;

  private:
    void clear_forces();
    void compute_gravity();
    void compute_particle_collisions();
    void compute_border_collisions();
    void compute_forces();
    void calculate_derivative();
    void scale_derivative();
    void calculate_state();
    void restore_state();
};
