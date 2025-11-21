#include "utils.hpp"

void debug_log(std::string message) {
    if (!Config::instance().isDebug()) return;
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);

    std::cout << "[" << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "] "
              << message << std::endl;
}

void log(std::string name, std::string message) {
    if (!Config::instance().isDebug()) return;
    debug_log("[" + name + "] " + message);
}

void log_error(std::string name, std::string message) {
    debug_log("**ERROR** " "[" + name + "] " + message);
}

std::string operandToString(const MCOperand &Op) {
    if (Op.isReg()) {
        return "x" + std::to_string(Op.getReg() - llvm::RISCV::X0);
    } else if (Op.isImm()) {
        return std::to_string(Op.getImm());
    }
    // Should not have the expression case afaik
    return "<unknown>";
}