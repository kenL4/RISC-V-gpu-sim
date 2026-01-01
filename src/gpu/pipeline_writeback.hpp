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
  WritebackResume(CoalescingUnit *cu, RegisterFile *rf);
  void execute() override;
  bool is_active() override;

  ~WritebackResume() {};

private:
  CoalescingUnit *cu;
  RegisterFile *rf;
};