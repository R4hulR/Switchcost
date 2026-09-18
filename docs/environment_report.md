# Environment Report

Date: 2026-09-17
Host: Windows + WSL2 (Ubuntu), workstation described as Intel Xeon Silver 4210, dual-socket, 64 GB RAM, Quadro P1000.

## 1. Kernel / distro

```
Linux <hostname> 6.6.87.2-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Thu Jun 5 18:30:46 UTC 2025 x86_64
Ubuntu 26.04.1 LTS (Resolute Raccoon)
init: systemd
```

WSL2 (`wsl.exe -l -v`): `Ubuntu` running, WSL version 2. A stopped `docker-desktop` WSL distro also exists but is not running and not used here.

## 2. CPU topology — WSL2 does NOT expose the real 2-socket topology

`lscpu`:

```
CPU(s):                40
Thread(s) per core:    2
Core(s) per socket:    20
Socket(s):             1
NUMA node(s):          1
NUMA node0 CPU(s):     0-39
Model name:            Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz
```

**Finding: WSL2 flattens the physical topology.** The real machine has two Xeon Silver 4210 sockets (10 cores / 20 threads each, per Intel's spec for this SKU) for 20 physical cores / 40 threads total. WSL2 reports this as a single virtual socket with 20 "cores" and 2 threads/core = 40 logical CPUs, and collapses everything into **one NUMA node** (`node0 CPU(s): 0-39`). There is no way, from inside this WSL2 VM, to tell which logical CPUs correspond to which physical socket, or to observe cross-socket memory latency. Any thread/worker placement experiment run here is blind to NUMA effects — a worker "pinned" to CPUs 0-9 might in reality span both physical sockets.

**Consequence for the pilot:** all WSL2-based results must be labeled `[WSL2, virtualized topology, no NUMA visibility]`. Socket-aware or NUMA-aware placement experiments (e.g., "keep a worker within one socket to avoid cross-socket cache/memory traffic") are not meaningful here and must be deferred to native Linux if the project proceeds that far. The pilot as scoped (fixed worker/thread counts, no NUMA-aware placement) is not blocked by this, but any future claim about socket locality needs native-Linux confirmation.

`numactl` is not installed and reports only 1 node when data is available via `/sys/devices/system/node` (`node0` only, `possible: 0`). No independent signal contradicts `lscpu` — WSL2 truly does not surface the second socket.

## 3. CPU affinity — enforced at the guest scheduler, not verified below that

```
$ taskset -c 0,1 python3 -c "import os; print(os.sched_getaffinity(0))"
affinity: {0, 1}
```

`taskset` is present and the **guest (WSL2 VM) scheduler** honors affinity restrictions — a process pinned to logical CPUs 0-1 is confirmed to see only those in its affinity mask. This is real and enforced *at that level*. It is **not** evidence of physical isolation: WSL2 runs as a Hyper-V partition, and the hypervisor is free to schedule the VM's virtual CPUs (and therefore anything pinned "inside" them) onto whatever physical host threads it chooses, time-sliced with the Windows host and any other VMs/processes. So "worker pinned to guest CPUs 0-7" means the worker will only ever run on those 8 guest-visible scheduling slots — it does **not** mean those 8 slots map to 8 dedicated physical threads, and it does not guarantee isolation from Windows background work or from the load generator's guest CPUs at the physical layer. Guest-level pinning is still useful (it fixes *which* virtual CPUs compete for physical time, and keeps a given worker's threads from migrating across guest CPUs mid-run), but affinity numbers in this report should be read as "guest logical CPU N", not as a physical core/socket identity.

Correspondingly, in §2's sibling-pairing check: `thread_siblings_list` reports internally-consistent pairs (0-1, 2-3, ...), but this is the *guest's* view, constructed from whatever CPUID topology leaves Hyper-V chose to expose. It shows the guest's pairing is self-consistent, not that guest CPU 0 and guest CPU 1 are actually the two hardware threads of one physical core. Treat it as a stable guest-level grouping to pin against, not a confirmed physical-core assignment.

## 4. CPU quota / cgroups

```
$ cat /proc/self/cgroup
0::/init.scope
$ cat /sys/fs/cgroup/init.scope/cpu.max
max 100000
```

`init.scope/cpu.max = max` only says there is no quota **at that one cgroup**. Checked the full ancestor chain and the other plausible cgroup an interactive shell might land in:

| cgroup | cpu.max | memory.max |
|---|---|---|
| `init.scope` (where this inspection session's shell runs) | `max 100000` | `max` |
| `user.slice/user-1000.slice` (where an interactive login shell would run) | `max 100000` | `max` |
| root (`/sys/fs/cgroup`) | no `cpu.max` file — cgroup v2 roots don't expose one; there is no cgroup above root to constrain it | n/a |

No quota at either place a benchmark process is likely to run, and nothing above them in the tree can impose one that wouldn't show up here. `cpuset.cpus.effective = 0-39` at both, confirming no cpuset restriction either. This is about **cgroup-level** throttling only — it says nothing about the Hyper-V-level CPU time-sharing described in §3.

## 5. Memory — discrepancy between configured and visible RAM

```
$ free -h
Mem: 31Gi total, 29Gi free, 8.0Gi swap
```

`C:\Users\<user>\.wslconfig` (dated 2023-04-21, the only `.wslconfig` found) specifies:

```
[wsl2]
memory=50GB
```

**Finding: WSL2 is only granting ~31 GiB, not the configured 50 GB — cause not verified.** I have **not** established which of the following (if any) is the actual cause: `.wslconfig` changes requiring a `wsl --shutdown` + restart that hasn't happened; Windows not having 50 GB free to grant at VM-start time; another config/policy overriding this file; or something else. None of these has been checked — they are unconfirmed candidate explanations, not a diagnosis, and should not be treated as established fact in the writeup.

**Decision (user, 2026-09-17): proceed with the currently visible ~31 GiB as-is for this pilot.** No `.wslconfig`/Windows-side change will be made. The pilot's CPU budget (8 logical CPUs) and model size (~22M-param embedding model, a handful of process copies) are small enough that ~29 GB usable headroom is not expected to be a binding constraint — but this is an expectation, not a guarantee, so memory and swap will be actively monitored (RSS per worker, `free -h` before/after each run) rather than assumed safe, per §7 of the pilot config matrix.

8 GB swap is configured and unused at rest. `vm.swappiness = 60` (default) — under memory pressure the kernel will swap fairly eagerly; if benchmark runs show latency cliffs under memory pressure, swappiness is a candidate confound to control for (log it, consider lowering for benchmark runs, but that is a system-setting change and needs your go-ahead).

## 6. GPU (not used in the CPU-only pilot, recorded for completeness)

```
NVIDIA-SMI 570.152   Driver 573.24   CUDA 12.8
GPU: Quadro P1000, 4096 MiB total, 963 MiB already in use (Xwayland), 1% util
```

GPU passthrough into WSL2 is working. Not used for the CPU-only pilot; recorded in case a CPU-vs-GPU comparison becomes relevant later. Note ~1 GB of the 4 GB VRAM is already consumed by the WSLg desktop compositor (`Xwayland`), leaving ~3 GB usable if GPU work happens later.

## 7. Toolchain

```
Python:  3.14.4  (/usr/bin/python3)
gcc:     15.2.0
cmake:   4.2.3
venv:    available
disk:    928 GB free on /
```

`onnxruntime` on PyPI (checked directly against the package index, latest release 1.30.0) **does** publish `cp314-manylinux_2_28_x86_64` wheels, so Python 3.14 is not a blocker — no need to install an older Python via pyenv/deadsnakes. This was worth checking explicitly since 3.14 is very new (released Oct 2025) and some ML packages lag Python releases by months.

## 8. Background load / virtualization caveats to carry into every experiment

- This is a WSL2 VM on a Windows host actively running other things (browser, IDE, etc. on the same physical cores) — CPU time is not exclusively ours the way it would be on a dedicated bare-metal box or a cgroup-isolated container. Document what else is running on the host during each benchmark run.
- No NUMA visibility (Section 2) — cross-socket effects are invisible and unmeasurable here.
- Memory ceiling is ~29 GB in practice, not the nominally configured 50 GB (Section 5), until resolved on the Windows side.
- `hypervisor` flag is set in `/proc/cpuinfo`; timing (`rdtsc`/`BogoMIPS`) inside a VM can have more jitter than bare metal — worth keeping in mind when interpreting very fine-grained (sub-millisecond) latency numbers.
- WSL2's virtual CPU numbering (0-39) is stable within a boot but is not guaranteed to map to the same physical threads across VM restarts — re-verify topology (`lscpu`) at the start of each experiment session, don't assume it's static across days.

## 9. Summary judgment

The pilot as scoped (fixed worker/thread-count configurations, batch size fixed, no NUMA-aware placement, no live topology-dependent decisions) is **not blocked** by WSL2's topology virtualization. CPU affinity works and is sufficient to control which logical CPUs a worker uses. The two open items are: (a) confirm/fix the 50GB vs 31GB memory gap if more headroom is needed, and (b) treat every result from this environment as WSL2-labeled and plan a native-Linux confirmation run before any topology-sensitive claim (e.g., socket-local worker placement) goes into a final evaluation.
