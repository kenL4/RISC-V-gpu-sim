#pragma once

#include <cstddef>

// GPU Simulator Configuration - Copied from SIMTight defaults in Config.h
constexpr size_t DRAM_BEAT_BYTES = 64;
constexpr size_t DRAM_BEAT_LOG_BYTES = 6;
constexpr size_t SIM_DRAM_LATENCY = 30;

// Functional unit latencies (matching SIMTight)
// Multiplier: 3 cycles (3-stage pipeline)
constexpr size_t SIM_MUL_LATENCY = 3;
// Divider: 32 cycles (sequential divider, default in SIMTight)
// Note: Full-throughput divider is configurable (12 cycles) but disabled by default
constexpr size_t SIM_DIV_LATENCY = 32;
constexpr size_t SIM_REM_LATENCY = 32;  // Same as division (uses same unit)

// Memory request queue capacity (matching SIMTight: makeSizedQueueCore 5 = 2^5 = 32)
// This is the INPUT queue capacity - the pipeline tracking (inflightCount) has separate capacity 4
constexpr size_t MEM_REQ_QUEUE_CAPACITY = 32;  // Matching SIMTight: memReqsQueue input queue capacity

constexpr size_t SIM_SHARED_SRAM_BASE = 0xBFFF0000;
constexpr size_t SIM_SIMT_STACK_BASE = 0xC0000000;
constexpr size_t SIM_REG_SPILL_SIZE = 0x00080000; // I don't actually do any spilling rn
constexpr size_t SIM_CPU_STACK_BASE = SIM_SHARED_SRAM_BASE - SIM_REG_SPILL_SIZE;
constexpr size_t SIM_CPU_INITIAL_SP = SIM_CPU_STACK_BASE - 8;

// GPU pipeline configuration
constexpr size_t NUM_LANES = 32;
constexpr size_t NUM_WARPS = 64;
constexpr size_t NUM_REGISTERS = 32;  // RISC-V has 32 general-purpose registers

// For command line options that I pass
class Config {
public:
  // C++17 inline static for thread-safe singleton
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

  void setStatsOnly(bool value) { statsOnly = value; }
  bool isStatsOnly() { return statsOnly; }

private:
  bool debug = false;
  bool regDump = false;
  bool cpuDebug = false;
  bool statsOnly = false;
  Config() = default;
};