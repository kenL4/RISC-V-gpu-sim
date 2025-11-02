#pragma once

#include "utils.hpp"
#include "pipeline.hpp"

/*
 * Represents the warp scheduler unit in the pipeline
 * It will use a barrel scheduler to fairly pick between the warps
 * that have no suspended threads
 */
class WarpScheduler : public PipelineStage {
public:
    WarpScheduler(int warp_size, int warp_count, uint64_t start_pc);
    void execute() override;
    bool is_active() override;
    void insert_warp(Warp *warp);
    ~WarpScheduler();
private:
    int warp_size;
    int warp_count;
    std::queue<Warp*> warp_queue;
    std::queue<Warp*> new_warp_queue;

    void flush_new_warps();
};