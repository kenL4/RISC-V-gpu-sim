#include "data_cache.hpp"
#include "mem_data.hpp"

DataCache::DataCache() : lines(SIM_CACHE_NUM_LINES) {}

uint64_t DataCache::get_tag(uint64_t addr) {
  return addr >> (SIM_CACHE_LINE_SIZE_LOG + SIM_CACHE_NUM_LINES_LOG);
}

size_t DataCache::get_line_index(uint64_t addr) {
  return (addr >> SIM_CACHE_LINE_SIZE_LOG) & (SIM_CACHE_NUM_LINES - 1);
}

size_t DataCache::get_line_offset(uint64_t addr) {
  return addr & (SIM_CACHE_LINE_SIZE - 1);
}

uint64_t DataCache::get_line_base_addr(uint64_t tag, size_t index) {
  return (tag << (SIM_CACHE_LINE_SIZE_LOG + SIM_CACHE_NUM_LINES_LOG)) |
         (index << SIM_CACHE_LINE_SIZE_LOG);
}

bool DataCache::probe(uint64_t addr) {
  size_t index = get_line_index(addr);
  uint64_t tag = get_tag(addr);
  const CacheLine &line = lines[index];
  return line.valid && (line.tag == tag);
}

bool DataCache::access(uint64_t addr, bool is_store) {
  size_t index = get_line_index(addr);
  uint64_t tag = get_tag(addr);
  CacheLine &line = lines[index];

  // Check for hit
  if (line.valid && line.tag == tag) {
    hits++;
    if (is_store) {
      line.dirty = true;
    }
    return true;
  }

  // Miss - need to fetch the line
  misses++;

  // If current line is dirty, write it back first
  if (line.valid && line.dirty) {
    writeback_line(index);
  }

  // Fetch the new line
  fetch_line(index, addr);

  if (is_store) {
    line.dirty = true;
  }

  return false;
}

uint8_t DataCache::load_byte(uint64_t addr) {
  size_t index = get_line_index(addr);
  size_t offset = get_line_offset(addr);
  return lines[index].data[offset];
}

void DataCache::store_byte(uint64_t addr, uint8_t val) {
  size_t index = get_line_index(addr);
  size_t offset = get_line_offset(addr);
  lines[index].data[offset] = val;
  lines[index].dirty = true;
}

double DataCache::get_hit_rate() {
  uint64_t total = hits + misses;
  if (total == 0)
    return 0.0;
  return static_cast<double>(hits) / total;
}

void DataCache::reset_stats() {
  hits = 0;
  misses = 0;
}

void DataCache::fetch_line(size_t index, uint64_t addr) {
  if (!backing_memory)
    return;

  CacheLine &line = lines[index];
  uint64_t tag = get_tag(addr);
  uint64_t base_addr = get_line_base_addr(tag, index);

  // Repeated one byte loads for the entire cache line from backing memory
  for (size_t i = 0; i < SIM_CACHE_LINE_SIZE; i++) {
    int64_t val = backing_memory->load(base_addr + i, 1);
    line.data[i] = static_cast<uint8_t>(val & 0xFF);
  }

  line.valid = true;
  line.dirty = false;
  line.tag = tag;
}

void DataCache::writeback_line(size_t index) {
  if (!backing_memory)
    return;

  CacheLine &line = lines[index];
  if (!line.valid || !line.dirty)
    return;

  uint64_t base_addr = get_line_base_addr(line.tag, index);

  // Write the entire cache line back to backing memory
  for (size_t i = 0; i < SIM_CACHE_LINE_SIZE; i++) {
    backing_memory->store(base_addr + i, 1, line.data[i]);
  }

  line.dirty = false;
}

void DataCache::flush() {
  for (size_t i = 0; i < SIM_CACHE_NUM_LINES; i++) {
    if (lines[i].valid && lines[i].dirty) {
      writeback_line(i);
    }
  }
}
