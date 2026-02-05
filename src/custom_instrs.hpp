#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <optional>

/*
 * One custom instruction: name, synthetic opcode, and one of:
 *   (a) byte_pattern: match instruction bytes by prefix (exact bytes at start), or
 *   (b) mask + value: match when (insn32 & mask) == value (insn = 4 bytes LE).
 * Handler type selects the execution handler (e.g. "noclpush", "noop").
 */
struct CustomInstrEntry {
  std::string name;
  unsigned int opcode;
  std::vector<uint8_t> byte_pattern;  // if non-empty, use prefix match
  uint32_t mask = 0;   // if non-zero, use (insn32 & mask) == value (overrides byte_pattern)
  uint32_t value = 0;
  std::string handler_type;
};

/*
 * Load custom instructions from a file.
 * Two formats (one per line, # = comment):
 *
 *   (1) Byte-prefix:  NAME OPCODE_HEX BYTE_PATTERN_HEX [HANDLER_TYPE]
 *   (2) Mask+value:   NAME OPCODE_HEX mask=MASK_HEX value=VALUE_HEX [HANDLER_TYPE]
 *
 * (1) Matches when the instruction bytes start with BYTE_PATTERN (hex, e.g. 0900).
 * (2) Matches when (insn32 & MASK) == VALUE; insn32 is the 4-byte instruction (little-endian).
 *     Use this for RISC-V-style encodings where only some fields matter, e.g.:
 *     NOCL_PUSH:  opcode=0x09, funct3=000  -> mask=0x707F value=0x0009
 *     NOCL_POP:   opcode=0x09, funct3=001  -> mask=0x707F value=0x1009
 *     CACHE_FLUSH: opcode=0x08, funct3=000 -> mask=0x707F value=0x0008
 *
 * HANDLER_TYPE defaults to "noop" if omitted.
 */
std::vector<CustomInstrEntry> load_custom_instrs(const std::string &path);

/*
 * Look up custom instruction by synthetic opcode.
 */
std::optional<std::string> custom_opcode_to_name(
    const std::vector<CustomInstrEntry> &entries, unsigned int opcode);

/*
 * Look up custom instruction by name to get handler type.
 */
std::optional<std::string> custom_name_to_handler_type(
    const std::vector<CustomInstrEntry> &entries, const std::string &name);

/*
 * Check if raw instruction bytes match any custom pattern; if so set out_opcode and return true.
 * Uses first match.
 */
bool match_custom_instruction(const std::vector<CustomInstrEntry> &entries,
                              const uint8_t *data, size_t size,
                              unsigned int *out_opcode);
