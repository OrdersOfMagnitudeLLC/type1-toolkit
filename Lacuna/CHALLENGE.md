# The Lacuna Challenge

Produce a valid license key for the binary below without access to source code.

**Binary:** `lacuna_crackme` (included in this directory)
**Input:** A 64-character hex string (32 bytes)
**Success:** Binary prints `LICENSE VALID`

## What you're attacking

Three nested virtual machines with per-build hardware-randomized instruction
sets. License key is hashed via BLAKE2b-256 then used to navigate a
ChaCha20-encrypted byte sequence. Only the correct key finds the valid
positions. Wrong key produces silent failure: no error, no hint, no timing
difference.

Anti-debug is wired into computation, not detection. Running under GDB
corrupts the output. It is not a gate you bypass: it is an input to the
calculation.

The expected validation value is split across three XOR components, none
meaningful in isolation. All strings are runtime-encoded. No symbols.

## Prior attempts

- **angr:** completed in 0.4 seconds, found nothing. ChaCha20 makes the state
  space mathematically intractable.
- **AI reverse engineer with full tooling (30 min):** mapped the full architecture,
  could not generate a valid key.
- **Binary patch:** requires patching multiple branches (build-specific offsets).
  Produces a patched binary that accepts anything: not a valid key.
  Per-customer unique binaries mean each crack is build-specific and
  non-transferable.
- **BLAKE2b preimage:** 2^128 operations. Infeasible.

## Rules

- Binary only. No source code will be provided.
- The crack must generalize: binary patching this specific binary does not
  count. A valid key that the unmodified binary accepts does.
- Submit to: orders@ofmagnitude.com

## Prize

First confirmed valid key: NS Suite commercial license, all products,
founding tier rate, locked forever.

This offer expires 2 years from the date of the HN post. If nobody claims it, that's the point.
