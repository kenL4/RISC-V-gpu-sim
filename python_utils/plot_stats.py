from matplotlib import pyplot as plt
import numpy as np

def read_simtight_trace(file):
    f = open(file, "r")
    data = {}
    line = f.readline()
    while line:
        parts = line.split(" ")

        if len(parts) == 0 or parts[0] != "Running":
            line = f.readline()
            continue
            
        kernel = parts[2].strip()
        if kernel not in data:
            data[kernel] = {
                "Cycles": 0,
                "Instrs": 0,
                "Susps": 0,
                "Retries": 0,
                "DRAMAccs": 0,
            }
        
        line = f.readline()
        parts = line.split(":")
        
        while parts[0] != "Self test":
            if parts[0] not in ["Detected an error at index", "Expected value", "Computed value"]:
                data[kernel][parts[0]] += int(parts[1], 16)

            # Move to the next line
            line = f.readline()
            parts = line.split(":")

        data[kernel]["IPC"] = float(data[kernel]["Instrs"]) / data[kernel]["Cycles"]
    return data

def plot_gpu_instrs(data, simtight):
    gpu_instrs = [data[name]["Instrs"] for name in data.keys()]
    simtight_gpu_instrs = [simtight[name]["Instrs"] for name in simtight.keys()]
    
    index = np.arange(len(gpu_instrs))
    bar_width = 0.35

    fig, ax = plt.subplots()
    mine = ax.bar(index, gpu_instrs, bar_width, label="Mine")
    baseline = ax.bar(index + bar_width, simtight_gpu_instrs, bar_width, label="SIMTight")

    ax.set_xlabel('Kernel')
    ax.set_ylabel('Instrs')
    ax.set_title('GPU Instrs by GPU, Kernel')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(data.keys())
    ax.legend()

    plt.show()

def plot_gpu_cycles(data, simtight):
    gpu_cycles = [data[name]["Cycles"] for name in data.keys()]
    simtight_gpu_cycles = [simtight[name]["Cycles"] for name in simtight.keys()]
    
    index = np.arange(len(gpu_cycles))
    bar_width = 0.35

    fig, ax = plt.subplots()
    mine = ax.bar(index, gpu_cycles, bar_width, label="Mine")
    baseline = ax.bar(index + bar_width, simtight_gpu_cycles, bar_width, label="SIMTight")

    ax.set_xlabel('Kernel')
    ax.set_ylabel('Cycles')
    ax.set_title('GPU Cycles by GPU, Kernel')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(data.keys())
    ax.legend()

    plt.show()

def plot_dram_accs(data, simtight):
    gpu_dram_accs = [data[name]["DRAMAccs"] for name in data.keys()]
    simtight_gpu_dram_accs = [simtight[name]["DRAMAccs"] for name in simtight.keys()]
    
    index = np.arange(len(gpu_dram_accs))
    bar_width = 0.35

    fig, ax = plt.subplots()
    mine = ax.bar(index, gpu_dram_accs, bar_width, label="Mine")
    baseline = ax.bar(index + bar_width, simtight_gpu_dram_accs, bar_width, label="SIMTight")

    ax.set_xlabel('Kernel')
    ax.set_ylabel('DRAMAccs')
    ax.set_title('GPU DRAMAccs by GPU, Kernel')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(data.keys())
    ax.legend()

    plt.show()

def plot_ipc(data, simtight):
    gpu_ipc = [data[name]["IPC"] for name in data.keys()]
    simtight_gpu_ipc = [simtight[name]["IPC"] for name in simtight.keys()]
    
    index = np.arange(len(gpu_ipc))
    bar_width = 0.35

    fig, ax = plt.subplots()
    mine = ax.bar(index, gpu_ipc, bar_width, label="Mine")
    baseline = ax.bar(index + bar_width, simtight_gpu_ipc, bar_width, label="SIMTight")

    ax.set_xlabel('Kernel')
    ax.set_ylabel('IPC')
    ax.set_title('GPU IPC by GPU, Kernel')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(data.keys())
    ax.legend()

    plt.show()

def plot_gpu_retries(data, simtight):
    gpu_retries = [data[name].get("Retries", 0) for name in data.keys()]
    simtight_gpu_retries = [simtight[name].get("Retries", 0) for name in simtight.keys()]
    
    index = np.arange(len(gpu_retries))
    bar_width = 0.35

    fig, ax = plt.subplots()
    mine = ax.bar(index, gpu_retries, bar_width, label="Mine")
    baseline = ax.bar(index + bar_width, simtight_gpu_retries, bar_width, label="SIMTight")

    ax.set_xlabel('Kernel')
    ax.set_ylabel('Retries')
    ax.set_title('GPU Retries by GPU, Kernel')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(data.keys())
    ax.legend()

    plt.show()

def plot_gpu_all(data, simtight):
    gpu_instrs = [data[name]["Instrs"] for name in data.keys()]
    simtight_gpu_instrs = [simtight[name]["Instrs"] for name in simtight.keys()]
    
    index = np.arange(len(gpu_instrs))
    bar_width = 0.35

    fig, ax = plt.subplots(2, 2)
    mine = ax[1][0].bar(index, gpu_instrs, bar_width, label="Mine")
    baseline = ax[1][0].bar(index + bar_width, simtight_gpu_instrs, bar_width, label="SIMTight")

    ax[1][0].set_xlabel('Kernel')
    ax[1][0].set_ylabel('Instrs')
    ax[1][0].set_title('GPU Instrs by GPU, Kernel')
    ax[1][0].set_xticks(index + bar_width / 2)
    ax[1][0].set_xticklabels(data.keys())
    ax[1][0].legend()

    gpu_cycles = [data[name]["Cycles"] for name in data.keys()]
    simtight_gpu_cycles = [simtight[name]["Cycles"] for name in simtight.keys()]
    
    index = np.arange(len(gpu_cycles))
    bar_width = 0.35

    mine = ax[0][1].bar(index, gpu_cycles, bar_width, label="Mine")
    baseline = ax[0][1].bar(index + bar_width, simtight_gpu_cycles, bar_width, label="SIMTight")

    ax[0][1].set_xlabel('Kernel')
    ax[0][1].set_ylabel('Cycles')
    ax[0][1].set_title('GPU Cycles by GPU, Kernel')
    ax[0][1].set_xticks(index + bar_width / 2)
    ax[0][1].set_xticklabels(data.keys())
    ax[0][1].legend()

    gpu_dram_accs = [data[name]["DRAMAccs"] for name in data.keys()]
    simtight_gpu_dram_accs = [simtight[name]["DRAMAccs"] for name in simtight.keys()]
    
    index = np.arange(len(gpu_dram_accs))
    bar_width = 0.35

    mine = ax[0][0].bar(index, gpu_dram_accs, bar_width, label="Mine")
    baseline = ax[0][0].bar(index + bar_width, simtight_gpu_dram_accs, bar_width, label="SIMTight")

    ax[0][0].set_xlabel('Kernel')
    ax[0][0].set_ylabel('DRAMAccs')
    ax[0][0].set_title('GPU DRAMAccs by GPU, Kernel')
    ax[0][0].set_xticks(index + bar_width / 2)
    ax[0][0].set_xticklabels(data.keys())
    ax[0][0].legend()

    gpu_ipc = [data[name]["IPC"] for name in data.keys()]
    simtight_gpu_ipc = [simtight[name]["IPC"] for name in simtight.keys()]
    
    index = np.arange(len(gpu_ipc))
    bar_width = 0.35

    mine = ax[1][1].bar(index, gpu_ipc, bar_width, label="Mine")
    baseline = ax[1][1].bar(index + bar_width, simtight_gpu_ipc, bar_width, label="SIMTight")

    ax[1][1].set_xlabel('Kernel')
    ax[1][1].set_ylabel('IPC')
    ax[1][1].set_title('GPU IPC by GPU, Kernel')
    ax[1][1].set_xticks(index + bar_width / 2)
    ax[1][1].set_xticklabels(data.keys())
    ax[1][1].legend()

    plt.show()

if __name__ == "__main__":
    data = read_simtight_trace("trace.log")
    simtight = read_simtight_trace("trace_simtight.log")

    # plot_gpu_instrs(data, simtight)
    # plot_gpu_cycles(data, simtight)
    # plot_dram_accs(data, simtight)
    plot_gpu_all(data, simtight)