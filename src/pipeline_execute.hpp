#include "utils.hpp"
#include "pipeline.hpp"
#include "pipeline_warp_scheduler.hpp"
#include "register_file.hpp"

typedef struct execute_result {
    bool success;
    bool write_required;
} execute_result;

class ExecutionUnit {
public:
    ExecutionUnit(RegisterFile *rf);
    execute_result execute(Warp *warp, std::vector<size_t> active_threads, cs_insn *insn);
private:
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
    ExecuteSuspend(RegisterFile *rf, uint64_t max_addr);
    void execute() override;
    bool is_active() override;
    ~ExecuteSuspend();
private:
    ExecutionUnit *eu;
    uint64_t max_addr;
};