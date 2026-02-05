#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdint>

enum EventType {
    MEM_REQ_ISSUE,
    DRAM_REQ_ISSUE,
    INSTR_EXEC,
};

struct TraceEvent {
    uint64_t cycle;
    uint64_t pc;
    uint64_t warp_id;
    int lane_id;  // 0-31 for thread/lane, -1 for N/A (e.g. DRAM, or legacy 4-field)
    EventType event_type;

    // For mem_reqs
    std::vector<uint64_t> addrs;

    TraceEvent() : lane_id(-1) {}
};

class Tracer {
public:
    Tracer(std::string file_name) {
        file.open(file_name);
    }

    ~Tracer() {
        file.close();
    }

    void trace_event(TraceEvent event) {
        file << event.cycle << ","
             << "0x" << std::hex << std::setfill('0') << std::setw(8) << event.pc << std::dec << ","
             << event.warp_id << ","
             << event.lane_id << ","
             << static_cast<int>(event.event_type) << std::endl;
        if (event.event_type == MEM_REQ_ISSUE || event.event_type == DRAM_REQ_ISSUE) {
            for (uint64_t addr : event.addrs) {
                file << "0x" << std::hex << std::setfill('0') << std::setw(8) << addr << std::dec << ",";
            }
            file << std::endl;
        }
    }
private:
    std::ofstream file;
};