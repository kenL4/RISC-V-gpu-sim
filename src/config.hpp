#pragma once

#include <cstddef>

// GPU Simulator Configuration - Copied from SIMTight defaults in Config.h
constexpr size_t DRAM_BEAT_BYTES = 64;
constexpr size_t DRAM_BEAT_LOG_BYTES = 6;
constexpr size_t SIM_CACHE_LINE_SIZE = DRAM_BEAT_BYTES;
constexpr size_t SIM_CACHE_LINE_SIZE_LOG = DRAM_BEAT_LOG_BYTES;
constexpr size_t SIM_CACHE_NUM_LINES = 512;
constexpr size_t SIM_CACHE_NUM_LINES_LOG = 9;
constexpr size_t SIM_DRAM_LATENCY = 30;
constexpr size_t SIM_CACHE_HIT_LATENCY = 2;

// For command line options that I pass
class Config {
public:
  static Config &instance() {
    static Config inst;
    return inst;
  }

  void setDebug(bool value) { debug = value; }
  bool isDebug() { return debug; }

  void setRegisterDump(bool value) { regDump = value; }
  bool isRegisterDump() { return regDump; }

  void setCPUDebug(bool value) { cpuDebug = value; }
  bool isCPUDebug() { return cpuDebug; }

private:
  bool debug = false;
  bool regDump = false;
  bool cpuDebug = false;
  Config() = default;
};