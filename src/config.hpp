#pragma once

#include <cstddef>

// GPU Simulator Configuration - Copied/derived from SIMTight defaults in SIMTight Config.h
constexpr size_t DRAM_BEAT_BYTES = 64;
constexpr size_t DRAM_BEAT_LOG_BYTES = 6;
constexpr size_t SIM_DRAM_LATENCY = 30;

// TODO: Bring back MUL/DIV latency
constexpr size_t SIM_MUL_LATENCY = 3;
constexpr size_t SIM_DIV_LATENCY = 32;
constexpr size_t SIM_REM_LATENCY = 32;
constexpr size_t MEM_REQ_QUEUE_CAPACITY = 32;

constexpr size_t SIM_SHARED_SRAM_BASE = 0xBFFF0000;
constexpr size_t SIM_SIMT_STACK_BASE = 0xC0000000;
constexpr size_t SIM_REG_SPILL_SIZE = 0x00080000; // I don't actually do any spilling rn
constexpr size_t SIM_CPU_STACK_BASE = SIM_SHARED_SRAM_BASE - SIM_REG_SPILL_SIZE;
constexpr size_t SIM_CPU_INITIAL_SP = SIM_CPU_STACK_BASE - 8;

constexpr size_t SIMT_LOG_BYTES_PER_STACK = 19;
constexpr size_t SIMT_BYTES_PER_STACK = 1ULL << SIMT_LOG_BYTES_PER_STACK;
constexpr size_t SIMT_LOG_LANES = 5;
constexpr size_t SIMT_LOG_WARPS = 6;

// GPU pipeline configuration
constexpr size_t NUM_LANES = 32;
constexpr size_t NUM_WARPS = 64;
constexpr size_t NUM_REGISTERS = 32;  // RISC-V has 32 general-purpose registers

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

  void setStatsOnly(bool value) { statsOnly = value; }
  bool isStatsOnly() { return statsOnly; }

  void setQuick(bool value) {quick = value; }
  bool isQuick() { return quick; }

private:
  bool debug = false;
  bool regDump = false;
  bool cpuDebug = false;
  bool statsOnly = false;
  bool quick = false;
  Config() = default;
};