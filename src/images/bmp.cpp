#include "bmp.hpp"
#include "../mem/mem_data.hpp"

void empty_file(std::string filename) {
    std::ofstream file(filename);
    file.close();
}

void write_header(std::ofstream &file, uint64_t image_width, uint64_t image_height) {
    // Bitmap Header
    std::string signature = "BM";
    uint32_t file_size = image_width * image_height + 0x35;
    uint32_t reserved = 0;
    uint32_t data_offset = 0x35;

    file.write(signature.c_str(), sizeof(char) * 2); // Type Signature
    file.write((char *) &file_size, sizeof(char) * 4); // File Size
    file.write((char *) &reserved, sizeof(char) * 4); // Reserved
    file.write((char *) &data_offset, sizeof(char) * 4); // DataOffset

    // Info Header
    uint32_t size = 40;
    uint32_t width = image_width;
    uint32_t height = image_height;
    uint16_t planes = 1;
    uint16_t bits_per_colour = 24;
    uint32_t compression = 0;
    uint32_t image_size = 0;
    uint32_t xppm = 0;
    uint32_t yppm = 0;
    uint32_t colours = 24 * 256;
    uint32_t important_colours = 0;

    file.write((char *) &size, sizeof(char) * 4); // InfoHeader Size
    file.write((char *) &width, sizeof(char) * 4); // Width
    file.write((char *) &height, sizeof(char) * 4); // Height
    file.write((char *) &planes, sizeof(char) * 2); // Planes
    file.write((char *) &bits_per_colour, sizeof(char) * 2); // 24 bit colour
    file.write((char *) &compression, sizeof(char) * 4); // Compression Off
    file.write((char *) &image_size, sizeof(char) * 4); // Image Size (compressed)
    file.write((char *) &xppm, sizeof(char) * 4); // X pixels per metre (idk why we need this)
    file.write((char *) &yppm, sizeof(char) * 4); // Y pixels per metre (oh turns out it's not for PC)
    file.write((char *) &colours, sizeof(char) * 4); // Colours used
    file.write((char *) &important_colours, sizeof(char) * 4); // Important colours
}

void write_image(
    uint64_t width, 
    uint64_t height, 
    std::vector<std::vector<uint32_t>> &pixels,
    std::string filename
) {
    empty_file(filename);
    std::ofstream file(filename, std::ios::out | std::ios::binary | std::ios::app);

    // Write Bitmap header
    write_header(file, width, height);

    // Write pixel data
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            uint32_t pixel = pixels[y][x];
            uint8_t r = static_cast<uint8_t>((pixel >> 16) & 0xFF);
            uint8_t g = static_cast<uint8_t>((pixel >> 8) & 0xFF);
            uint8_t b = static_cast<uint8_t>(pixel & 0xFF);
            file.write((char *) &b, sizeof(char));
            file.write((char *) &g, sizeof(char));
            file.write((char *) &r, sizeof(char));
        }
    }

    file.close();
}

void render_framebuffer(
    DataMemory &memory,
    uint64_t base_addr,
    uint64_t width,
    uint64_t height,
    const std::string &filename
) {
    // Read pixels from memory
    size_t pixel_count = width * height;
    std::vector<uint32_t> flat_pixels = memory.get_memory_region(base_addr, pixel_count);
    
    // Convert flat array to 2D array for write_image
    std::vector<std::vector<uint32_t>> pixels(height);
    for (uint64_t y = 0; y < height; y++) {
        pixels[y].resize(width);
        for (uint64_t x = 0; x < width; x++) {
            size_t idx = y * width + x;
            if (idx < flat_pixels.size()) {
                pixels[y][x] = flat_pixels[idx];
            } else {
                pixels[y][x] = 0;  // Black for missing pixels
            }
        }
    }
    
    // Write to BMP file
    write_image(width, height, pixels, filename);
}