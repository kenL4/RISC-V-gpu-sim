#pragma once

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