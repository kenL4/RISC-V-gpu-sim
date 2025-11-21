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
    ExecutionUnit(CoalescingUnit *cu, RegisterFile *rf, LLVMDisassembler *disasm);
    execute_result execute(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst &inst);
private:
    CoalescingUnit *cu;
    RegisterFile *rf;
    LLVMDisassembler *disasm;
    bool add(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool addi(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool sub(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool and_(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool andi(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool or_(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool ori(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool xor_(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool xori(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool sll(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool slli(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool srl(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool srli(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool sra(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool srai(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool lui(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool auipc(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool lw(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool lh(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool lhu(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool lb(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool lbu(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool sw(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool sh(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool sb(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool jal(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool jalr(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool beq(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool bne(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool blt(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool bltu(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool bge(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool bgeu(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool ecall(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool ebreak(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
    bool csrrw(Warp *warp, std::vector<size_t> active_threads, MCInst *in);
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
    ExecuteSuspend(CoalescingUnit *cu, RegisterFile *rf, uint64_t max_addr, LLVMDisassembler *disasm);
    void execute() override;
    bool is_active() override;
    ~ExecuteSuspend();
private:
    CoalescingUnit *cu;
    ExecutionUnit *eu;
    LLVMDisassembler *disasm;
    uint64_t max_addr;
};