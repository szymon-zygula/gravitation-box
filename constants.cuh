#include <algorithm>

constexpr size_t WINDOW_WIDTH = 1024;
constexpr size_t WINDOW_HEIGHT = 1024;

constexpr float PARTICLE_RADIUS = 0.0025f;
constexpr float GRAVITATIONAL_ACCELERATION = -3.0f;
constexpr size_t PARTICLE_COUNT = 1250;

constexpr bool GPU_ACCELERATION = false;

constexpr size_t BLOCK_SIZE = std::min((size_t)1024, PARTICLE_COUNT);
constexpr size_t BLOCK_COUNT = PARTICLE_COUNT / BLOCK_SIZE;

constexpr float TIME_DELTA = 0.01f;
constexpr float BOX_COLLISION_MULTIPLIER = -0.5f;
constexpr float BOX_SIDE_LENGTH = 2.0f;
