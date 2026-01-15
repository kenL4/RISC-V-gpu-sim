#pragma once

#include "utils.hpp"

class DataMemory {
public:
  int64_t load(uint64_t addr, size_t bytes);
  void store(uint64_t addr, size_t bytes, uint64_t val);
  
  std::vector<uint32_t> get_memory_region(uint64_t addr, size_t count);
  const std::map<uint64_t, uint8_t>& get_raw_memory() const { return memory; }

private:
  std::map<uint64_t, uint8_t> memory;
};