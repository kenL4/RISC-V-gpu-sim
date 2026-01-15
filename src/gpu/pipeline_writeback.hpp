#include "mem/mem_coalesce.hpp"
#include "pipeline.hpp"
#include "register_file.hpp"
#include "utils.hpp"

/*
 * The Writeback/Resume unit writes back the per-lane
 * results to the register file for each active thread.
 * It also handles the clearing of the suspension bit,
 * if there are no writes to be done.
 */
class WritebackResume : public PipelineStage {
public:
  WritebackResume(CoalescingUnit *cu, RegisterFile *rf, bool is_cpu_pipeline);
  void execute() override;
  bool is_active() override;
  
  std::function<void(Warp *warp)> insert_warp;

  ~WritebackResume() {};

private:
  CoalescingUnit *cu;
  RegisterFile *rf;
  bool is_cpu_pipeline;  // True if this is the CPU pipeline, false if GPU
};