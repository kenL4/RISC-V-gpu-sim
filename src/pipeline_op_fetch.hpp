#include "utils.hpp"
#include "pipeline.hpp"
#include "register_file.hpp"

/*
 * The Operand Fetch unit performs a lookup on the register file
 * for a given warp ID and the source register IDs
 * 
 * For now, I am deferring resolution of operands to
 * the execute stage so this is essentially a no-op.
 */
class OperandFetch: public PipelineStage {
public:
    OperandFetch();
    void execute() override;
    bool is_active() override;
    ~OperandFetch() {};
};