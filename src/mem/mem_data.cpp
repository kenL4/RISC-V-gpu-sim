#include "mem_data.hpp"

// Helper functions for sign/zero extension (used by both DataMemory and CoalescingUnit)
int64_t sign_extend(uint64_t val, size_t bytes) {
  switch (bytes) {
  case 1:
    return int64_t(int8_t(val & 0xFF));
  case 2:
    return int64_t(int16_t(val & 0xFFFF));
  case 4:
    return int64_t(int32_t(val & 0xFFFFFFFF));
  case 8:
    return int64_t(val); // already 64-bit
  default:
    throw std::invalid_argument("Invalid load size");
  }
}

int64_t zero_extend(uint64_t val, size_t bytes) {
  switch (bytes) {
  case 1:
    return int64_t(val & 0xFF);  // Zero-extend byte to 64 bits
  case 2:
    return int64_t(val & 0xFFFF);  // Zero-extend halfword to 64 bits
  case 4:
    return int64_t(val & 0xFFFFFFFF);  // Zero-extend word to 64 bits
  case 8:
    return int64_t(val); // already 64-bit
  default:
    throw std::invalid_argument("Invalid load size");
  }
}

int64_t DataMemory::load(uint64_t addr, size_t bytes) {
  uint64_t raw = 0;
  for (int i = 0; i < bytes; i++) {
    if (memory.find(addr + i) == memory.end()) {
      continue;
    }

    raw += (uint64_t)memory[addr + i] << (8 * i);
  }

  int64_t res = sign_extend(raw, bytes);
  return res;
}

void DataMemory::store(uint64_t addr, size_t size, uint64_t val) {
  for (size_t i = 0; i < size; i++) {
    memory[addr + i] = (val >> (8 * i)) & 0xFF;
  }
}

std::vector<uint32_t> DataMemory::get_memory_region(uint64_t addr, size_t count) {
  std::vector<uint32_t> result;
  result.reserve(count);
  
  for (size_t i = 0; i < count; i++) {
    // Read 4 bytes (32-bit pixel) in little-endian format
    uint64_t pixel_addr = addr + (i * 4);
    uint32_t pixel = 0;
    
    for (int j = 0; j < 4; j++) {
      auto it = memory.find(pixel_addr + j);
      if (it != memory.end()) {
        pixel |= static_cast<uint32_t>(it->second) << (8 * j);
      }
    }
    
    result.push_back(pixel);
  }
  
  return result;
}