#pragma once

#include "config.hpp"
#include <cstdint>
#include <vector>

// Cache line structure
struct CacheLine {
  bool valid = false;
  bool dirty = false;
  uint64_t tag = 0;
  std::vector<uint8_t> data;

  CacheLine() : data(SIM_CACHE_LINE_SIZE, 0) {}
};

// Simple direct-mapped writeback data cache
// Sits between the coalescing unit and DRAM
class DataCache {
public:
  DataCache();

  // Check if address is in cache (hit) without loading
  bool probe(uint64_t addr);

  // Access the cache - returns true on hit, false on miss
  // On miss, the cache line is fetched from backing memory
  bool access(uint64_t addr, bool is_store);

  // Load a byte from cache (assumes line is present)
  uint8_t load_byte(uint64_t addr);

  // Store a byte to cache (assumes line is present)
  void store_byte(uint64_t addr, uint8_t val);

  // Get cache statistics
  uint64_t get_hits() { return hits; }
  uint64_t get_misses() { return misses; }
  double get_hit_rate();

  // Reset statistics
  void reset_stats();

  // Set backing memory pointer for miss handling
  void set_backing_memory(class DataMemory *mem) { backing_memory = mem; }

  // Flush cache (write all dirty lines back)
  void flush();

private:
  std::vector<CacheLine> lines;
  class DataMemory *backing_memory = nullptr;

  // Statistics
  uint64_t hits = 0;
  uint64_t misses = 0;

  // Helper functions
  uint64_t get_tag(uint64_t addr);
  size_t get_line_index(uint64_t addr);
  size_t get_line_offset(uint64_t addr);
  uint64_t get_line_base_addr(uint64_t tag, size_t index);

  // Fetch a cache line from backing memory
  void fetch_line(size_t index, uint64_t addr);

  // Write back a dirty cache line to backing memory
  void writeback_line(size_t index);
};
