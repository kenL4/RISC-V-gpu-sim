#include "utils.hpp"
#include "pipeline.hpp"
#include "pipeline_warp_scheduler.hpp"
#include "register_file.hpp"
#include "mem_coalesce.hpp"

typedef struct execute_result {
    bool success;
    bool write_required;
} execute_result;

/*
 * The Execution Unit is the unit that handles the actual
 * computation and production of side-effects of instructions
 * in the pipeline.
 */
class ExecutionUnit {
public:
    ExecutionUnit(CoalescingUnit *cu, RegisterFile *rf);
    execute_result execute(Warp *warp, std::vector<size_t> active_threads, cs_insn *insn);
private:
    CoalescingUnit *cu;
    RegisterFile *rf;
    bool add(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool addi(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool neg(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool sub(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool and_(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool andi(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool not_(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool or_(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool ori(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool xor_(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool xori(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool sll(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool slli(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool srl(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool srli(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool sra(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool srai(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool li(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool lui(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool auipc(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool lw(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool lh(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool lhu(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool lb(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool lbu(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool la(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool sw(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool sh(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
    bool sb(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv);
};

/*
 * The Execute/Suspend unit executes the instruction and reinserts
 * the warp ID into the warp queue. It also performs the memory access
 * request.
 * 
 * We don't need to handle hazards as there will only be one
 * instruction per warp in the pipeline at any given time.
 */
class ExecuteSuspend: public PipelineStage {
public:
    std::function<void(Warp *warp)> insert_warp;
    ExecuteSuspend(CoalescingUnit *cu, RegisterFile *rf, uint64_t max_addr);
    void execute() override;
    bool is_active() override;
    ~ExecuteSuspend();
private:
    CoalescingUnit *cu;
    ExecutionUnit *eu;
    uint64_t max_addr;
};