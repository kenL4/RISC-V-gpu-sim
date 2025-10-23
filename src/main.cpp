#include "cxxopts.hpp"
#include "utils.hpp"
#include "pipeline.cpp"
#include "pipeline_warp_scheduler.cpp"
#include "pipeline_ats.cpp"
#include "pipeline_instr_fetch.cpp"
#include "mem_instr.hpp"

Pipeline* initialize_pipeline(InstructionMemory *im) {
    Pipeline *p = new Pipeline();

    // Construct stages
    p->add_stage<WarpScheduler>(32, 2, im->get_base_addr());
    p->add_stage<ActiveThreadSelection>();
    p->add_stage<InstructionFetch>(im);
    p->add_stage<MockPipelineStage>("Operand Fetch");
    p->add_stage<MockPipelineStage>("Execute/Suspend");
    p->add_stage<MockPipelineStage>("Writeback/Resume");

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
        ("d, debug", "Turn on debugging logs");
        ("h,help", "Show help");
    options.parse_positional({"filename"});
    options.positional_help("<Input File>");
    auto result = options.parse(argc, argv);

    if (result.count("help") || !result.count("filename")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    Config::instance().setDebug(result.count("debug") > 0);

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

    Pipeline *p = initialize_pipeline(&tcim);
    while (p->has_active_stages()) {
        p->execute();
    }
    delete p;

    // Make sure to clean up the capstone instructions
    if (out.count > 0) cs_free(out.insn, out.count);
    cs_close(&out.handle);
    return 0;
}