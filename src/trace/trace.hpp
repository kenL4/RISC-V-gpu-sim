#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

enum EventType {
    MEM_REQ_ISSUE,
    DRAM_REQ_ISSUE,
};

struct TraceEvent {
    uint64_t cycle;
    uint64_t pc;
    uint64_t warp_id;
    EventType event_type;

    // For mem_reqs
    std::vector<uint64_t> addrs;
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
             << event.pc << "," 
             << event.warp_id << ","
             << event.event_type << std::endl;
        if (event.event_type == MEM_REQ_ISSUE || event.event_type == DRAM_REQ_ISSUE) {
            for (uint64_t addr : event.addrs) {
                file << addr << ",";
            }
            file << std::endl;
        }
    }
private:
    std::ofstream file;
};