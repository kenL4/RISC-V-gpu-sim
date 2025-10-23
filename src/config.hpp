#pragma once

class Config {
public:
    static Config& instance() {
        static Config inst;
        return inst;
    }

    void setDebug(bool value) { debug = value; }
    bool isDebug() { return debug; }
private:
    bool debug = false;
    Config() = default;
};