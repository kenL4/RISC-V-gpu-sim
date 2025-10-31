#include "cxxopts.hpp"
#include "utils.hpp"
#include "pipeline.hpp"
#include "pipeline_warp_scheduler.hpp"
#include "pipeline_ats.hpp"
#include "pipeline_instr_fetch.hpp"
#include "pipeline_op_fetch.hpp"
#include "pipeline_execute.hpp"
#include "pipeline_writeback.hpp"
#include "mem_instr.hpp"

#define NUM_LANES 32
#define NUM_WARPS 1

Pipeline* initialize_pipeline(InstructionMemory *im, RegisterFile *rf) {
    Pipeline *p = new Pipeline();

    // Construct stages
    p->add_stage<WarpScheduler>(NUM_LANES, NUM_WARPS, im->get_base_addr());
    p->add_stage<ActiveThreadSelection>();
    p->add_stage<InstructionFetch>(im);
    p->add_stage<OperandFetch>();
    p->add_stage<ExecuteSuspend>(rf, im->get_max_addr());
    p->add_stage<WritebackResume>(rf);

    std::shared_ptr<WarpScheduler> warp_scheduler_stage = std::dynamic_pointer_cast<WarpScheduler>(p->get_stage(0));
    std::shared_ptr<ExecuteSuspend> execute_stage = std::dynamic_pointer_cast<ExecuteSuspend>(p->get_stage(4));
    execute_stage->insert_warp = [ws = warp_scheduler_stage] (Warp *warp) {
        ws->insert_warp(warp);
    };

    // Initialize latches
    PipelineLatch *latches[6];
    for (int i = 0; i < 6; i++) {
        latches[i] = new PipelineLatch();
    }

    p->get_stage(0)->set_latches(latches[5], latches[0]);
    p->get_stage(1)->set_latches(latches[0], latches[1]);
    p->get_stage(2)->set_latches(latches[1], latches[2]);
    p->get_stage(3)->set_latches(latches[2], latches[3]);
    p->get_stage(4)->set_latches(latches[3], latches[4]);
    p->get_stage(5)->set_latches(latches[4], latches[5]);

    return p;
}

int main(int argc, char* argv[]) {
    cxxopts::Options options("RISCVGpuSim", "A software simulator for a RISC-V GPU");

    options.add_options()
        ("filename", "Input filename", cxxopts::value<std::string>())
        ("d,debug", "Turn on debugging logs")
        ("r,regdump", "Dump the register values after each writeback stage")
        ("h,help", "Show help");
    options.parse_positional({"filename"});
    options.positional_help("<Input File>");
    auto result = options.parse(argc, argv);

    if (result.count("help") || !result.count("filename")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    Config::instance().setDebug(result.count("debug") > 0);
    Config::instance().setRegisterDump(result.count("regdump") > 0);

    std::string filename = result["filename"].as<std::string>();
    
    debug_log("Loading ELF file...");
    parse_output out;
    parse_error parse_err = parse_binary(filename, &out);
    if (parse_err != PARSE_SUCCESS) {
        std::cout << "Failed to load/parse file: " << filename << std::endl;
        return 1;
    }
    debug_log("Successfully loaded ELF file!");

    InstructionMemory tcim(&out);
    debug_log("Instruction memory has base_addr " + std::to_string(tcim.get_base_addr()));

    size_t register_count = 32;
    RegisterFile rf(register_count, NUM_LANES);
    debug_log("Register file instantiated with " + std::to_string(register_count) + " registers");

    Pipeline *p = initialize_pipeline(&tcim, &rf);
    while (p->has_active_stages()) {
        p->execute();
    }
    delete p;

    // Make sure to clean up the capstone instructions
    if (out.count > 0) cs_free(out.insn, out.count);
    cs_close(&out.handle);
    return 0;
}