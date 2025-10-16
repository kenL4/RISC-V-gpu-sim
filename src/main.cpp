#include "cxxopts.hpp"
#include <iostream>
#include <fstream>

int main(int argc, char* argv[]) {
    cxxopts::Options options("RISCVGpuSim", "A software simulator for a RISC-V GPU");

    options.add_options()
        ("filename", "Input filename", cxxopts::value<std::string>())
        ("h,help", "Show help");
    options.parse_positional({"filename"});
    options.positional_help("<Input File>");
    auto result = options.parse(argc, argv);

    if (result.count("help") || !result.count("filename")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    std::string filename = result["filename"].as<std::string>();
    std::cout << "Input file: " << filename << std::endl;
    // TODO: Actually read the file!
}