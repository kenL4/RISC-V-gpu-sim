#pragma once

class Config {
public:
    static Config& instance() {
        static Config inst;
        return inst;
    }

    void setDebug(bool value) { debug = value; }
    bool isDebug() { return debug; }

    void setRegisterDump(bool value) { regDump = value; }
    bool isRegisterDump() { return regDump; }
private:
    bool debug = false;
    bool regDump = false;
    Config() = default;
};