#pragma once

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <vector>
#include "llvm/MC/TargetRegistry.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCDisassembler/MCRelocationInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#define GET_REGINFO_ENUM
#include "gen/llvm_riscv_registers.h"

#include "config.hpp"
#include "parser.hpp"
#include "stats/stats.hpp"

/*
 * Prints a generic message with an associated timestamp
 */
void debug_log(std::string message);

/*
 * Prints a named message with an associated timestamp
 */
void log(std::string name, std::string message);

/*
 * Prints a named error message with an associated timestamp
 */
void log_error(std::string name, std::string message);

/*
 * Returns the string form of an LLVM operand
 */
std::string operandToString(const llvm::MCOperand &Op);