NSRun ships as a git submodule — run git submodule update --init to pull.

NSRun is a fork of ik_llama.cpp with four NS components:
- NSKVCache — 208.7x KV compression with content-based cross-window recall
- NSAttend — 28.7x attention speedup
- NSInfer — 2.35x MLP throughput via sparse activation
- NSQuant — Activation-aware variable-rate quantization

See ./LICENSE.md for license details.
