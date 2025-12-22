#include "../utils.hpp"

// Based on the work of michaeljclark, five ways to invoke the LLVM disassembler
// https://gist.github.com/michaeljclark/d94b72fa3d580ea2037e0a4dc5e2fc5b#file-llvmdiscpp-cpp-L15

using namespace llvm;

class LLVMDisassembler {
public:
    LLVMDisassembler(std::string target_id, std::string cpu, std::string features) {
        target = TargetRegistry::lookupTarget(Triple(target_id), err);
        if (!target) {
            std::cout << "Couldn't find " + target_id << std::endl;
        }
        ri.reset(target->createMCRegInfo(Triple(target_id)));
        ai.reset(target->createMCAsmInfo(*ri, Triple(target_id), options));
        si.reset(target->createMCSubtargetInfo(Triple(target_id), cpu, features));
        ii.reset(target->createMCInstrInfo());
        cx.reset(new MCContext(Triple(target_id), ai.get(), ri.get(), si.get()));
        di.reset(target->createMCDisassembler(*si, *cx));
        ip.reset(target->createMCInstPrinter(Triple(target_id), ai->getAssemblerDialect(), *ai, *ii, *ri));
    }

    void format_hex(raw_string_ostream &out, ArrayRef<uint8_t> data, size_t offset, size_t size) {
        int nbytes = size < hexcols ? size : hexcols;
        out << format_hex_no_prefix(offset, 8) << ":"
            << format_bytes(data.slice(offset, nbytes), {}, hexcols, 1);
        out.indent((hexcols - nbytes) * 3 + 8 - (hexcols * 3) % 8);
    }

    std::vector<MCInst> disasm_insts(size_t offset, ArrayRef<uint8_t> data) {
        std::string buf;
        raw_string_ostream out(buf);
        std::vector<MCInst> insts;
        MCInst in;
        uint64_t size;

        while (offset < data.size() && di->getInstruction(in, size, data.slice(offset), offset, out)) {
            format_hex(out, data, offset, size);
            if (size == 0) break;
            // Emplace back a copy of in
            insts.emplace_back(MCInst(in));
            
            buf.clear();
            while (size > hexcols) {
                offset += hexcols; size -= hexcols;
                format_hex(out, data, offset, size);
                std::cout << buf << std::endl;
                buf.clear();
            }
            offset += size;
        }

        return insts;
    }

    MCInst disasm_inst(size_t offset, ArrayRef<uint8_t> data) {
        std::string buf;
        raw_string_ostream out(buf);
        MCInst in;
        uint64_t size;

        di->getInstruction(in, size, data.slice(offset), offset, out);
        format_hex(out, data, offset, size);
        if (size == 0) {
            std::cout << "[WARNING] Instruction has no size?" << std::endl;
        }

        if (getOpcodeName(in.getOpcode()) == "PHI") {
            // Check if we are in a noclPush/Pop case
            std::string buf_substr = buf.substr(0, 14);
            if (buf_substr == "00000000:09 00" || buf_substr == "00000000:09 10") {
                std::string type = buf.substr(12, 1);
                if (type == "0") {
                    in.setOpcode(0xFF);
                } else {
                    in.setOpcode(0xFE);
                }
            } else if (buf_substr == "00000000:08 00") {
                in.setOpcode(0xFD);
                // TODO: Implement cache line flush
            }
        }
        return in;
    }

    void print(const MCInst &inst, uint64_t offset) {
        std::string buf;
        raw_string_ostream out(buf);
        ip->printInst(&inst, offset, "", *si, out);
        std::cout << buf << std::endl;
    }

    std::string getOpcodeName(unsigned int opcode) {
        if (opcode == 0xFF) {
            return "NOCLPUSH";
        } else if (opcode == 0xFE) {
            return "NOCLPOP";
        } else if (opcode == 0xFD) {
            return "CACHE_LINE_FLUSH";
        }
        return ii->getName(opcode).str();
    }
private:
    static const int hexcols = 10;

    const Target *target;
    std::string err;
    MCTargetOptions options;

    std::unique_ptr<MCRegisterInfo> ri;
    std::unique_ptr<MCAsmInfo> ai;
    std::unique_ptr<MCSubtargetInfo> si;
    std::unique_ptr<MCInstrInfo> ii;
    std::unique_ptr<MCContext> cx;
    std::unique_ptr<MCDisassembler> di;
    std::unique_ptr<MCInstPrinter> ip;
};