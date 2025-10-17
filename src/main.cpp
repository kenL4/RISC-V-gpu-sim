#include "cxxopts.hpp"
#include "utils.hpp"
#include "parser.hpp"

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
    std::cout << "Loading ELF file... " << std::endl;
    parse_error parse_error_code = parse_binary(filename);
    if (parse_error_code != PARSE_SUCCESS) {
        std::cout << "Failed to load/parse file: " << filename << std::endl;
        return 1;
    }
    std::cout << "Successfully loaded ELF file!" << std::endl;

    return 0;
}