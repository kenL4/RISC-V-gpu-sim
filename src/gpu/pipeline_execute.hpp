#include "disassembler/llvm_disasm.hpp"
#include "host/host_gpu_control.hpp"
#include "mem/mem_coalesce.hpp"
#include "pipeline.hpp"
#include "trace/trace.hpp"
#include "stats/stats.hpp"
#include "pipeline_warp_scheduler.hpp"
#include "register_file.hpp"
#include "utils.hpp"

typedef struct execute_result {
  bool success;
  bool write_required;
  bool counted;
} execute_result;

/*
 * The Execution Unit is the unit that handles the actual
 * computation and production of side-effects of instructions
 * in the pipeline.
 */
class ExecutionUnit {
public:
  ExecutionUnit(CoalescingUnit *cu, RegisterFile *rf, LLVMDisassembler *disasm,
                HostGPUControl *gpu_controller);
  execute_result execute(Warp *warp, std::vector<size_t> active_threads,
                         llvm::MCInst &inst);

  void set_debug(bool enabled) { debug_enabled = enabled; }
  void log(std::string name, std::string message) {
    if (debug_enabled)
      ::log(name, message);
  }

private:
  CoalescingUnit *cu;
  RegisterFile *rf;
  LLVMDisassembler *disasm;
  HostGPUControl *gpu_controller;
  bool debug_enabled = true;
  bool add(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool addi(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool sub(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool mul(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool and_(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool andi(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool or_(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool ori(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool xor_(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool xori(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool sll(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool slli(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool srl(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool srli(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool sra(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool srai(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool lui(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool auipc(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool lw(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool lh(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool lhu(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool lb(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool lbu(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool sw(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool sh(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool sb(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool amoadd_w(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool jal(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool jalr(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool beq(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool bne(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool blt(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool bltu(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool bge(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool bgeu(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool slti(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool slt(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool sltiu(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool sltu(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool remu(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool divu(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool div_(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool rem_(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool fence(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool ecall(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool ebreak(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool csrrw(Warp *warp, std::vector<size_t> active_threads, llvm::MCInst *in);
  bool noclpush(Warp *warp, std::vector<size_t> active_threads,
                llvm::MCInst *in);
  bool noclpop(Warp *warp, std::vector<size_t> active_threads,
               llvm::MCInst *in);
  bool cache_line_flush(Warp *warp, std::vector<size_t> active_threads,
                        MCInst *in);
};

/*
 * The Execute/Suspend unit executes the instruction and reinserts
 * the warp ID into the warp queue. It also performs the memory access
 * request.
 *
 * We don't need to handle hazards as there will only be one
 * instruction per warp in the pipeline at any given time.
 */
class ExecuteSuspend : public PipelineStage {
public:
  std::function<void(Warp *warp)> insert_warp;
  ExecuteSuspend(CoalescingUnit *cu, RegisterFile *rf, uint64_t max_addr,
                 LLVMDisassembler *disasm, HostGPUControl *gpu_controller);
  void execute() override;
  void set_debug(bool enabled) override {
    PipelineStage::set_debug(enabled);
    eu->set_debug(enabled);
  }
  bool is_active() override;
  ExecutionUnit *get_execution_unit() { return eu; }
  void set_instr_tracer(Tracer *t) { instr_tracer = t; }
  ~ExecuteSuspend();

private:
  CoalescingUnit *cu;
  ExecutionUnit *eu;
  LLVMDisassembler *disasm;
  uint64_t max_addr;
  Tracer *instr_tracer = nullptr;
};