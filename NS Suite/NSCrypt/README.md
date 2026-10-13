# NSCrypt

Schema-aware AEAD + NTT-256 acceleration for post-quantum cryptography.

## Sub-products

### 1. Schema-aware AEAD (ChaCha20-Poly1305)

Standard AEAD treats all payload bytes identically. NSCrypt lets the
caller declare which fields are `AUTH_ONLY` (known/fixed, authenticate
but don't encrypt) vs `ENCRYPT` (private, encrypt + authenticate).

**Win:** fewer bytes encrypted → smaller output, faster throughput.
On a FIX 4.1-shaped 64B payload (~37.5% fixed fields), NSCrypt encrypts
40B vs OpenSSL's 64B — 2.7x throughput, 4.4x per-byte.

**Loss:** random payload with no schema → all bytes ENCRYPT, no savings.
Use standard AEAD. Same pattern as the rest of the suite.

### 2. NTT-256 acceleration (Kyber/ML-KEM, Zq=3329)

The NTT hot path in Kyber (NIST PQC standard) is structurally identical
to FFT but over finite field Zq. Zero hardware acceleration exists.
NSCrypt implements AVX-512 butterfly with two-level Barrett reduction
(all products fit in 32-bit, no 64-bit division).

**Win:** 1.3x over scalar reference on single NTT call. The vectorized
path processes 16 butterflies per instruction at levels where len >= 16
(4 of 8 butterfly levels). Small levels (len=1,2,4,8) fall back to
scalar — the butterfly stride is too small for contiguous vector loads.

**Batch-16 NTT:** transposes 16 independent polynomials so all butterfly
levels are fully vectorized. Registers run packed epi16 (32 lanes: two
coefficient positions × 16 polys) with Montgomery reduction via single-uop
`mullo_epi16`/`mulhi_epi16`. Correctness matches scalar. Current speedup:
~4.8x vs 16× scalar (was 2.67x with epi32 gather/scatter — the epi16
rewrite is ~1.81x faster still).

**Loss:** the 93% of butterflies at small levels are scalar, limiting
single-NTT speedup.

## Architecture

- **ChaCha20:** RFC 8439 IETF variant, 20 rounds, 96-bit nonce.
- **Poly1305:** RFC 8439, 26-bit limb arithmetic, two-level carry.
- **Schema AEAD:** Poly1305 key from ChaCha20 block 0, encrypt with
  counter=1, authenticate AAD (AUTH_ONLY fields + enc_len) || ciphertext.
- **NTT:** Cooley-Tukey forward, Gentleman-Sande inverse, Barrett
  reduction (v=314, shift=20), zeta=17, n=256, q=3329.
- **AVX-512:** 16-wide butterfly with `_mm512_mullo_epi32` + vectorized
  Barrett reduction. Per-level twiddle precomputation. Scalar fallback
  for len < 16 and non-AVX-512 targets.
- **Batch-16 NTT:** 16 polynomials transposed via contiguous-load 16×16
  in-register shuffle network (no gather/scatter), then packed to epi16:
  each ZMM holds 2 coef positions × 16 polys. Montgomery-reduce butterflies
  on `mullo_epi16`+`mulhi_epi16`, twiddles pre-scaled by 2^16 mod q, level
  len=1 handled inside 256-bit register halves (w=1, no multiply). Working
  set halved to 8KB.

## Build

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --target nscrypt -j$(nproc)
./build/nscrypt
```

Or build manually with `g++`:

```sh
g++ -O3 -march=native -DUSE_OPENSSL -o bench_nscrypt bench_nscrypt.cpp -lcrypto
./bench_nscrypt
```

Without OpenSSL: drop `-DUSE_OPENSSL -lcrypto` (NTT benchmark still runs).

## Benchmark

Kings: OpenSSL `EVP_chacha20_poly1305` (AEAD), scalar portable C NTT (NTT).
Correctness: AEAD encrypt+decrypt round-trip, NTT→INTT identity.

Results on Spectre (i7-1165G7, AVX-512):

**Schema-aware AEAD (FIX 4.1, 64B payload, 37.5% fixed):**

| Kernel | Throughput | Per-byte | Encrypts |
|--------|-----------:|---------:|---------:|
| NSCrypt | 2.74x | 4.39x | 40B |
| OpenSSL | 1.00x | 1.00x | 64B |

Output: 56B (40B ciphertext + 16B tag) vs OpenSSL's 80B (64B + 16B tag).

**NTT-256 (Zq=3329, n=256):**

| Kernel | Time/call | Speedup |
|--------|----------:|--------:|
| Scalar | 2.47 µs | 1.00x |
| AVX-512 | 1.92 µs | 1.29x |
| Batch-16 (16 NTTs) | 8.0 µs / 16 (4.47 µs isolated) | ~4.8x vs 16× scalar |

Correctness: NTT→INTT round-trip OK for scalar and AVX-512; batch-16
forward output matches scalar reference.

**Batch-16 progression (isolated, best-of-10):**

| Revision | µs/call | ns/NTT |
|----------|--------:|-------:|
| epi32 gather/scatter | 8.10 | 506.5 |
| in-register transpose + hoisted schedule + x4 unroll | 6.26 | 391.2 |
| epi16 Montgomery (current) | 4.47 | 279.2 |

## Win / Loss

- **Win:** schema-declared payloads with known fixed fields (FIX, HL7,
  protobuf with known tags). PQC NTT hot path.
- **Loss (documented):** random payloads with no schema → no savings.
  Small NTT butterfly levels → scalar fallback limits speedup.
