#pragma once

#include "../utils.hpp"

// Forward declaration to avoid circular include
class DataMemory;

void write_image(
    uint64_t width, 
    uint64_t height, 
    std::vector<std::vector<uint32_t>> &pixels,
    std::string filename
);

void render_framebuffer(
    DataMemory &memory,
    uint64_t base_addr,
    uint64_t width,
    uint64_t height,
    const std::string &filename
);