#include "custom_instrs.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace {

bool parse_hex_byte(const std::string &s, size_t i, uint8_t *out) {
  if (i + 1 >= s.size()) return false;
  auto hex = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  int hi = hex(s[i]), lo = hex(s[i + 1]);
  if (hi < 0 || lo < 0) return false;
  *out = static_cast<uint8_t>((hi << 4) | lo);
  return true;
}

std::vector<uint8_t> parse_byte_pattern(const std::string &hex_str) {
  std::vector<uint8_t> out;
  std::string s;
  for (char c : hex_str) {
    if (c == ' ' || c == '\t') continue;
    s += c;
  }
  for (size_t i = 0; i + 1 < s.size(); i += 2) {
    uint8_t b;
    if (!parse_hex_byte(s, i, &b)) break;
    out.push_back(b);
  }
  return out;
}

unsigned int parse_opcode(const std::string &s) {
  std::string h = s;
  if (h.size() >= 2 && (h.substr(0, 2) == "0x" || h.substr(0, 2) == "0X"))
    h = h.substr(2);
  unsigned int x = 0;
  for (char c : h) {
    int d = -1;
    if (c >= '0' && c <= '9') d = c - '0';
    else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
    else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
    if (d < 0) break;
    x = (x << 4) | static_cast<unsigned>(d);
  }
  return x;
}

uint32_t parse_u32_hex(const std::string &s) {
  std::string h = s;
  if (h.size() >= 2 && (h.substr(0, 2) == "0x" || h.substr(0, 2) == "0X"))
    h = h.substr(2);
  uint32_t x = 0;
  for (char c : h) {
    int d = -1;
    if (c >= '0' && c <= '9') d = c - '0';
    else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
    else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
    if (d < 0) break;
    x = (x << 4) | static_cast<uint32_t>(d);
  }
  return x;
}

// Parse "mask=0x707F" or "value=0x1009"; return true and set *out if successful
bool parse_mask_value_token(const std::string &token, const std::string &prefix,
                            uint32_t *out) {
  if (token.size() <= prefix.size() || token.substr(0, prefix.size()) != prefix)
    return false;
  *out = parse_u32_hex(token.substr(prefix.size()));
  return true;
}

std::string trim(const std::string &s) {
  auto start = s.find_first_not_of(" \t");
  if (start == std::string::npos) return "";
  auto end = s.find_last_not_of(" \t");
  return s.substr(start, end == std::string::npos ? end : end - start + 1);
}

}  // namespace

std::vector<CustomInstrEntry> load_custom_instrs(const std::string &path) {
  std::vector<CustomInstrEntry> entries;
  std::ifstream f(path);
  if (!f) return entries;

  std::string line;
  while (std::getline(f, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') continue;

    std::istringstream iss(line);
    std::string name, opcode_str, third, handler_type = "noop";
    if (!(iss >> name >> opcode_str >> third)) continue;

    CustomInstrEntry e;
    e.name = name;
    e.opcode = parse_opcode(opcode_str);
    e.handler_type = "noop";

    if (third.substr(0, 5) == "mask=") {
      if (!parse_mask_value_token(third, "mask=", &e.mask)) continue;
      std::string fourth;
      if (!(iss >> fourth) || fourth.substr(0, 6) != "value=") continue;
      if (!parse_mask_value_token(fourth, "value=", &e.value)) continue;
      iss >> e.handler_type;
    } else {
      e.byte_pattern = parse_byte_pattern(third);
      if (e.byte_pattern.empty()) continue;
      iss >> e.handler_type;
    }
    if (e.handler_type.empty()) e.handler_type = "noop";

    if (e.mask == 0 && e.byte_pattern.empty()) continue;

    entries.push_back(std::move(e));
  }
  return entries;
}

std::optional<std::string> custom_opcode_to_name(
    const std::vector<CustomInstrEntry> &entries, unsigned int opcode) {
  for (const auto &e : entries) {
    if (e.opcode == opcode) return e.name;
  }
  return std::nullopt;
}

std::optional<std::string> custom_name_to_handler_type(
    const std::vector<CustomInstrEntry> &entries, const std::string &name) {
  for (const auto &e : entries) {
    if (e.name == name) return e.handler_type;
  }
  return std::nullopt;
}

bool match_custom_instruction(const std::vector<CustomInstrEntry> &entries,
                              const uint8_t *data, size_t size,
                              unsigned int *out_opcode) {
  for (const auto &e : entries) {
    if (e.mask != 0) {
      if (size < 4) continue;
      uint32_t insn = static_cast<uint32_t>(data[0]) |
                      (static_cast<uint32_t>(data[1]) << 8) |
                      (static_cast<uint32_t>(data[2]) << 16) |
                      (static_cast<uint32_t>(data[3]) << 24);
      if ((insn & e.mask) == e.value) {
        *out_opcode = e.opcode;
        return true;
      }
    } else {
      if (e.byte_pattern.size() > size) continue;
      bool match = true;
      for (size_t i = 0; i < e.byte_pattern.size(); ++i) {
        if (data[i] != e.byte_pattern[i]) {
          match = false;
          break;
        }
      }
      if (match) {
        *out_opcode = e.opcode;
        return true;
      }
    }
  }
  return false;
}
