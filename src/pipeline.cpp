#include "utils.hpp"

struct PipelineStage {
    std::string name;
    PipelineStage(std::string name): name(name) {};
    void execute() {
        std::cout << "Executing " << name << std::endl;
    };
};

struct Pipeline {
    void add_stage(PipelineStage&& stage) {
        stages.emplace_back(stage);
    }
    void execute() {
        for (auto stage : stages) {
            stage.execute();
        }
    }
private:
    std::vector<PipelineStage> stages;
};