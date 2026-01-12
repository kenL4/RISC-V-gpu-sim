#pragma once

#include "utils.hpp"

/*
 * An individual warp. This maintains the per-warp state
 * Matching SIMTight's SIMTThreadState structure
 */
class Warp {
public:
  uint64_t warp_id;
  size_t size;
  std::vector<uint64_t> pc;
  std::vector<uint64_t> nesting_level;
  std::vector<bool> finished;
  std::vector<bool> retrying;  // Matching SIMTight's simtRetry: per-thread retry flag
  bool suspended;
  Warp(uint64_t warp_id, size_t size, uint64_t start_pc, bool is_cpu)
      : warp_id(warp_id), size(size), is_cpu(is_cpu), suspended(false),
        pc(size, start_pc), nesting_level(size, 0), finished(size, false),
        retrying(size, false) {
  };

  bool is_cpu;
  ~Warp() {};
};

/*
 * A latch between each pipeline stage that defines
 * the input/output interface between stages
 * TODO: Refactor the latch so that we
 * can vary it between stages (union/interface)
 */
class PipelineLatch {
public:
  bool updated;
  Warp *warp;
  std::vector<uint64_t> active_threads;
  llvm::MCInst inst;
};

/*
 * The PipelineStage interface defines the
 * required capabilities of the stages in the pipeline
 */
class PipelineStage {
public:
  virtual ~PipelineStage() = default;
  virtual void execute() {};
  virtual bool is_active() { return false; };
  virtual void set_latches(PipelineLatch *input, PipelineLatch *output) {
    input_latch = input;
    output_latch = output;
  };

  virtual void set_debug(bool enabled) { debug_enabled = enabled; }
  void log(std::string name, std::string message) {
    if (debug_enabled)
      ::log(name, message);
  }

protected:
  PipelineLatch *input_latch;
  PipelineLatch *output_latch;
  bool debug_enabled = true;
};

/*
 * The Pipeline class represents a series of computation stages
 * that each warp will pass through.
 */
class Pipeline {
public:
  /*
   * Insert a stage in our polymorphic stage container
   */
  template <typename T, typename... Args> void add_stage(Args... args) {
    stages.emplace_back(std::make_shared<T>(args...));
  }

  /*
   * Execute one cycle of the pipeline
   */
  void execute();

  /*
   * Returns true if any of the associated pipeline stages
   * are still active
   */
  bool has_active_stages();

  /*
   * Returns the pipeline stage with index "index"
   */
  std::shared_ptr<PipelineStage> get_stage(int index);

  /*
   * Set pipeline active state (matching SIMTight's pipelineActive)
   * Pipeline stays active from kernel launch until all warps terminate
   */
  void set_pipeline_active(bool active) { pipeline_active = active; }
  bool is_pipeline_active() const { return pipeline_active; }

  void set_debug(bool enabled) {
    for (auto stage : stages) {
      stage->set_debug(enabled);
    }
  }

private:
  std::vector<std::shared_ptr<PipelineStage>> stages;
  bool pipeline_active = false;  // Matching SIMTight's pipelineActive
};

/*
 * A dummy implementation of a pipeline stage
 */
class MockPipelineStage : public PipelineStage {
public:
  std::string name;
  MockPipelineStage(std::string name) : name(name) {};
  void execute() override;
  bool is_active() override;
  ~MockPipelineStage() {};
};