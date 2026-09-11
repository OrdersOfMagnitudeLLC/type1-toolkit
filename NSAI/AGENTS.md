# NSAI — Agent Startup Instructions

## Every session, do this first:
1. Check if /tmp/qwen3b_stream.nsm exists. If not, regenerate it:
   cd $NS_HOME/NSAI/NSQuant
   ./nsquant $NS_HOME/deprecated/NSRun/models/qwen2.5-3b-instruct-q4_k_m.gguf /tmp/qwen3b_stream.nsm

2. Run coherence check before any code changes:
   cd $NS_HOME/NSAI
   ./nsrun_test /tmp/qwen3b_stream.nsm
   Expected: "The answer is 4", prefill ~11 t/s, generate ~2.0 t/s, RAM 615MB

3. Run coherence check again after any code changes.
   If numbers drop >20% or output is wrong, STOP and report. Do not proceed.

## Key paths:
- Source GGUF: $NS_HOME/deprecated/NSRun/models/qwen2.5-3b-instruct-q4_k_m.gguf
- NSM (temp):  /tmp/qwen3b_stream.nsm  ← wiped on every reboot, regenerate as above
- NSQuant:     $NS_HOME/NSAI/NSQuant/nsquant
- Test binary: $NS_HOME/NSAI/nsrun_test
- Build:       cd $NS_HOME/NSAI && make clean && make

## Rules:
- COLD_THR = 0.0f permanent — do not change
- Q6_K panel kernel permanently off-limits (14x regression)
- debug_float32 must be OFF before benchmarks
- make clean && make mandatory after any .h change
- One scoped change per commit. Show diff, wait for go-ahead.
- Two-attempt limit per task. If stuck twice, stop and report.
- Never run two Devin sessions concurrently on same file tree.
