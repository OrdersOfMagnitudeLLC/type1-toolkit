// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#ifndef NSCRYPT_HPP
#define NSCRYPT_HPP

#include <hwy/highway.h>
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cstddef>

// ===========================================================================
// Sub-product 1: Schema-aware AEAD (ChaCha20-Poly1305)
//
// Standard AEAD treats all payload bytes identically. NSCrypt lets the
// caller declare which fields are AUTH_ONLY (known, don't encrypt) vs
// ENCRYPT (private, encrypt+auth). Fewer bytes encrypted + smaller output.
// ===========================================================================

// --- ChaCha20 (RFC 8439 IETF variant) ---

static inline uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

#define CHACHA_QR(s, a, b, c, d) \
    s[a] += s[b]; s[d] ^= s[a]; s[d] = rotl32(s[d], 16); \
    s[c] += s[d]; s[b] ^= s[c]; s[b] = rotl32(s[b], 12); \
    s[a] += s[b]; s[d] ^= s[a]; s[d] = rotl32(s[d],  8); \
    s[c] += s[d]; s[b] ^= s[c]; s[b] = rotl32(s[b],  7)

static void chacha20_block(const uint8_t key[32], uint32_t counter,
                           const uint8_t nonce[12], uint32_t out[16]) {
    static const uint32_t SIGMA[4] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574};
    uint32_t s[16];
    memcpy(s, SIGMA, 16);
    memcpy(s + 4, key, 32);
    s[12] = counter;
    memcpy(s + 13, nonce, 12);

    uint32_t x[16];
    memcpy(x, s, 64);
    for (int i = 0; i < 10; ++i) {
        CHACHA_QR(x, 0, 4,  8, 12);
        CHACHA_QR(x, 1, 5,  9, 13);
        CHACHA_QR(x, 2, 6, 10, 14);
        CHACHA_QR(x, 3, 7, 11, 15);
        CHACHA_QR(x, 0, 5, 10, 15);
        CHACHA_QR(x, 1, 6, 11, 12);
        CHACHA_QR(x, 2, 7,  8, 13);
        CHACHA_QR(x, 3, 4,  9, 14);
    }
    for (int i = 0; i < 16; ++i) out[i] = x[i] + s[i];
}

static void chacha20_encrypt(const uint8_t key[32], uint32_t counter,
                             const uint8_t nonce[12],
                             const uint8_t* in, uint8_t* out, size_t len) {
    uint32_t block[16];
    size_t off = 0;
    while (off < len) {
        chacha20_block(key, counter++, nonce, block);
        size_t n = len - off < 64 ? len - off : 64;
        for (size_t i = 0; i < n; ++i)
            out[off + i] = in[off + i] ^ ((uint8_t*)block)[i];
        off += n;
    }
}

// --- Poly1305 (RFC 8439) ---

static void poly1305(const uint8_t key[32], const uint8_t* msg, size_t len,
                     uint8_t tag[16]) {
    // r = key[0..15] with clamping, s = key[16..31]
    uint32_t r[4], s[4];
    r[0] = (key[0]) | (key[1] << 8) | (key[2] << 16) | (key[3] << 24);
    r[1] = (key[4]) | (key[5] << 8) | (key[6] << 16) | (key[7] << 24);
    r[2] = (key[8]) | (key[9] << 8) | (key[10] << 16) | (key[11] << 24);
    r[3] = (key[12]) | (key[13] << 8) | (key[14] << 16) | (key[15] << 24);
    memcpy(s, key + 16, 16);

    // Clamp r
    r[0] &= 0x0fffffff;
    r[1] &= 0x0ffffffc;
    r[2] &= 0x0ffffffc;
    r[3] &= 0x0ffffffc;

    // Accumulator (130-bit, stored as 5x 26-bit limbs for easier reduction)
    uint32_t h[5] = {0, 0, 0, 0, 0};

    for (size_t i = 0; i < len; i += 16) {
        // Load block + 1 bit
        size_t n = len - i < 16 ? len - i : 16;
        uint32_t block[5] = {0, 0, 0, 0, 0};
        for (size_t j = 0; j < n; ++j)
            ((uint8_t*)block)[j] = msg[i + j];
        block[n / 4] |= (1u << (8 * (n % 4)));

        // h += block
        for (int j = 0; j < 5; ++j) h[j] += block[j];

        // h = (h * r) mod (2^130 - 5)
        // Full 130-bit multiply h * r, then reduce
        uint64_t t[5];
        t[0] = (uint64_t)h[0] * r[0];
        t[1] = (uint64_t)h[0] * r[1] + (uint64_t)h[1] * r[0];
        t[2] = (uint64_t)h[0] * r[2] + (uint64_t)h[1] * r[1] + (uint64_t)h[2] * r[0];
        t[3] = (uint64_t)h[0] * r[3] + (uint64_t)h[1] * r[2] + (uint64_t)h[2] * r[1] + (uint64_t)h[3] * r[0];
        t[4] = (uint64_t)h[1] * r[3] + (uint64_t)h[2] * r[2] + (uint64_t)h[3] * r[1] + (uint64_t)h[4] * r[0];

        // Carry propagation
        uint64_t c;
        h[0] = (uint32_t)(t[0] & 0x3ffffff); c = t[0] >> 26;
        t[1] += c; h[1] = (uint32_t)(t[1] & 0x3ffffff); c = t[1] >> 26;
        t[2] += c; h[2] = (uint32_t)(t[2] & 0x3ffffff); c = t[2] >> 26;
        t[3] += c; h[3] = (uint32_t)(t[3] & 0x3ffffff); c = t[3] >> 26;
        t[4] += c; h[4] = (uint32_t)(t[4] & 0x3ffffff); c = t[4] >> 26;
        h[0] += (uint32_t)c * 5;
        c = h[0] >> 26; h[0] &= 0x3ffffff;
        h[1] += (uint32_t)c;
    }

    // Final reduction: h = h - p (2^130 - 5)
    uint32_t g[5];
    uint32_t borrow = 0;
    for (int j = 0; j < 5; ++j) {
        g[j] = h[j] - 5 - borrow;
        borrow = (g[j] >> 31) & 1;
        g[j] &= 0x3ffffff;
    }
    uint32_t mask = (h[4] >> 31) ? 0 : 0x3ffffff;
    for (int j = 0; j < 5; ++j)
        h[j] = (h[j] & mask) | (g[j] & ~mask);

    // Add s
    uint32_t carry = 0;
    uint32_t hs[4];
    hs[0] = h[0] | (h[1] << 26);
    carry = hs[0] >> 26; hs[0] &= 0x3ffffff;
    // Actually, let's do it differently — pack h to 4x32 then add s
    uint32_t hp[4];
    uint64_t acc;
    acc = h[0] + ((uint64_t)h[1] << 26); hp[0] = (uint32_t)acc;
    acc = (acc >> 32) + ((uint64_t)h[2] << 20); hp[1] = (uint32_t)acc;
    acc = (acc >> 32) + ((uint64_t)h[3] << 14); hp[2] = (uint32_t)acc;
    acc = (acc >> 32) + ((uint64_t)h[4] << 8);  hp[3] = (uint32_t)acc;

    carry = 0;
    for (int j = 0; j < 4; ++j) {
        uint64_t sum = (uint64_t)hp[j] + s[j] + carry;
        hp[j] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
    }
    memcpy(tag, hp, 16);
}

// --- Schema-aware AEAD ---

enum NSCryptFieldType {
    NS_FIELD_AUTH_ONLY = 0,  // known/fixed: authenticate, don't encrypt
    NS_FIELD_ENCRYPT   = 1,  // private: encrypt + authenticate
};

struct NSCryptField {
    size_t offset;
    size_t length;
    NSCryptFieldType type;
};

struct NSCryptSchema {
    const NSCryptField* fields;
    size_t count;
};

// Encrypt: produces ciphertext (only ENCRYPT fields) + 16-byte tag.
// Output layout: [encrypted_variable_fields] [tag 16B]
// AAD = all AUTH_ONLY fields (in order) + lengths of ENCRYPT fields
// Ciphertext = ChaCha20 of concatenated ENCRYPT fields
// Tag = Poly1305(AAD || ciphertext)
// Returns total output size (ciphertext_len + 16).
static size_t ns_crypt_encrypt(const uint8_t key[32],
                               const uint8_t nonce[12],
                               const NSCryptSchema& schema,
                               const uint8_t* plaintext, size_t pt_len,
                               uint8_t* out, size_t out_cap) {
    // Collect ENCRYPT fields into a compact buffer.
    size_t enc_len = 0;
    for (size_t i = 0; i < schema.count; ++i)
        if (schema.fields[i].type == NS_FIELD_ENCRYPT)
            enc_len += schema.fields[i].length;

    if (enc_len + 16 > out_cap) return 0;

    // Extract ENCRYPT fields into contiguous buffer.
    uint8_t* enc_buf = (uint8_t*)aligned_alloc(64, (enc_len + 63) & ~(size_t)63);
    size_t off = 0;
    for (size_t i = 0; i < schema.count; ++i) {
        if (schema.fields[i].type == NS_FIELD_ENCRYPT) {
            memcpy(enc_buf + off, plaintext + schema.fields[i].offset,
                   schema.fields[i].length);
            off += schema.fields[i].length;
        }
    }

    // Generate Poly1305 key from ChaCha20 block 0.
    uint8_t poly_key[64];
    chacha20_block(key, 0, nonce, (uint32_t*)poly_key);

    // Encrypt ENCRYPT fields with ChaCha20 (counter=1).
    chacha20_encrypt(key, 1, nonce, enc_buf, out, enc_len);

    // Build AAD: AUTH_ONLY fields + enc_len as 8-byte LE.
    size_t aad_len = 0;
    for (size_t i = 0; i < schema.count; ++i)
        if (schema.fields[i].type == NS_FIELD_AUTH_ONLY)
            aad_len += schema.fields[i].length;
    aad_len += 8;  // enc_len prefix
    uint8_t* aad = (uint8_t*)aligned_alloc(64, (aad_len + 63) & ~(size_t)63);
    off = 0;
    for (size_t i = 0; i < schema.count; ++i) {
        if (schema.fields[i].type == NS_FIELD_AUTH_ONLY) {
            memcpy(aad + off, plaintext + schema.fields[i].offset,
                   schema.fields[i].length);
            off += schema.fields[i].length;
        }
    }
    memcpy(aad + off, &enc_len, 8);

    // Authenticate: AAD || ciphertext
    size_t auth_len = aad_len + enc_len;
    uint8_t* auth_buf = (uint8_t*)aligned_alloc(64, (auth_len + 63) & ~(size_t)63);
    memcpy(auth_buf, aad, aad_len);
    memcpy(auth_buf + aad_len, out, enc_len);

    uint8_t tag[16];
    poly1305(poly_key, auth_buf, auth_len, tag);
    memcpy(out + enc_len, tag, 16);

    free(enc_buf); free(aad); free(auth_buf);
    return enc_len + 16;
}

// Decrypt: verifies tag, decrypts ENCRYPT fields, reconstructs plaintext.
// Returns 1 on success, 0 on auth failure.
static int ns_crypt_decrypt(const uint8_t key[32],
                            const uint8_t nonce[12],
                            const NSCryptSchema& schema,
                            const uint8_t* in, size_t in_len,
                            uint8_t* plaintext, size_t pt_len) {
    size_t enc_len = 0;
    for (size_t i = 0; i < schema.count; ++i)
        if (schema.fields[i].type == NS_FIELD_ENCRYPT)
            enc_len += schema.fields[i].length;

    if (in_len != enc_len + 16) return 0;

    // Generate Poly1305 key.
    uint8_t poly_key[64];
    chacha20_block(key, 0, nonce, (uint32_t*)poly_key);

    // Rebuild AAD.
    size_t aad_len = 0;
    for (size_t i = 0; i < schema.count; ++i)
        if (schema.fields[i].type == NS_FIELD_AUTH_ONLY)
            aad_len += schema.fields[i].length;
    aad_len += 8;
    uint8_t* aad = (uint8_t*)aligned_alloc(64, (aad_len + 63) & ~(size_t)63);
    size_t off = 0;
    for (size_t i = 0; i < schema.count; ++i) {
        if (schema.fields[i].type == NS_FIELD_AUTH_ONLY) {
            memcpy(aad + off, plaintext + schema.fields[i].offset,
                   schema.fields[i].length);
            off += schema.fields[i].length;
        }
    }
    memcpy(aad + off, &enc_len, 8);

    // Verify tag.
    size_t auth_len = aad_len + enc_len;
    uint8_t* auth_buf = (uint8_t*)aligned_alloc(64, (auth_len + 63) & ~(size_t)63);
    memcpy(auth_buf, aad, aad_len);
    memcpy(auth_buf + aad_len, in, enc_len);

    uint8_t tag[16];
    poly1305(poly_key, auth_buf, auth_len, tag);
    free(aad); free(auth_buf);

    if (memcmp(tag, in + enc_len, 16) != 0) {
        return 0;
    }

    // Decrypt ENCRYPT fields.
    uint8_t* dec_buf = (uint8_t*)aligned_alloc(64, (enc_len + 63) & ~(size_t)63);
    chacha20_encrypt(key, 1, nonce, in, dec_buf, enc_len);

    // Scatter back to plaintext.
    off = 0;
    for (size_t i = 0; i < schema.count; ++i) {
        if (schema.fields[i].type == NS_FIELD_ENCRYPT) {
            memcpy(plaintext + schema.fields[i].offset, dec_buf + off,
                   schema.fields[i].length);
            off += schema.fields[i].length;
        }
    }
    free(dec_buf);
    return 1;
}

// ===========================================================================
// Sub-product 2: NTT-256 acceleration (Kyber/ML-KEM, Zq=3329)
//
// The NTT hot path is structurally identical to FFT but over Zq. No hardware
// acceleration exists. AVX-512 butterfly with two-level Barrett reduction
// (all products fit in 32-bit, no 64-bit needed).
// ===========================================================================

#define NS_NTT_Q 3329
#define NS_NTT_N 256

// Primitive 256th root of unity mod 3329: zeta = 17
#define NS_NTT_ZETA 17

// n^(-1) mod q = 3316 (since 256 * 3316 = 848896 = 255*3329 + 1)
#define NS_NTT_NINV 3316

// Barrett: v=314, shift=20. Max product 3328*3328=11075584, *314 < 2^32.
// Two levels reduce any product of mod-q values to [0, q).
static inline uint32_t barrett_reduce(uint32_t x) {
    uint32_t t = (x * 314) >> 20;
    uint32_t r = x - t * NS_NTT_Q;
    t = (r * 314) >> 20;
    r = r - t * NS_NTT_Q;
    if (r >= NS_NTT_Q) r -= NS_NTT_Q;
    return r;
}

// Batch-16 epi16 schedule: reg-space butterfly index pairs and Montgomery-
// scaled twiddle pairs (w = w_coef * 2^16 mod q). reg r packs coefs {2r, 2r+1}
// for all 16 polys; levels len>=2 pair whole regs (lanes 0-15 do even-coef
// butterfly, lanes 16-31 the odd-coef one). Populated by ntt_init_twiddles.
static uint16_t ns_ntt16_lo[8][64];
static uint16_t ns_ntt16_hi[8][64];
static uint16_t ns_ntt16_twa[8][64];
static uint16_t ns_ntt16_twb[8][64];
static uint16_t ns_ntt16_qinv;  // q^{-1} mod 2^16, for Montgomery reduce

// Precompute all 256 powers of zeta: zpow[i] = zeta^i mod q
static void ntt_init_twiddles(uint32_t zpow[NS_NTT_N]) {
    zpow[0] = 1;
    for (int i = 1; i < NS_NTT_N; ++i)
        zpow[i] = (uint32_t)((uint64_t)zpow[i-1] * NS_NTT_ZETA % NS_NTT_Q);

    // q^{-1} mod 2^16 via Newton iteration (4 steps: 3 -> 48 bits).
    uint32_t qi = NS_NTT_Q;
    for (int i = 0; i < 4; ++i) qi = qi * (2u - NS_NTT_Q * qi);
    ns_ntt16_qinv = (uint16_t)qi;

    // epi16 reg-space schedule for levels len = 2..128 (lvl 1..7).
    // Level len=1 is handled in-register (w = zpow[0] = 1, no table needed).
    const uint32_t R16 = 65536 % NS_NTT_Q;  // Montgomery scale = 2285
    for (int lvl = 1, len = 2; len < NS_NTT_N; len <<= 1, ++lvl) {
        int len_r = len >> 1, step = NS_NTT_N / (2 * len), b = 0;
        for (int i = 0; i < 128; i += 2 * len_r)
            for (int j = 0; j < len_r; ++j, ++b) {
                ns_ntt16_lo[lvl][b]  = (uint16_t)(i + j);
                ns_ntt16_hi[lvl][b]  = (uint16_t)(i + j + len_r);
                ns_ntt16_twa[lvl][b] = (uint16_t)((uint64_t)zpow[(2*j)*step]     * R16 % NS_NTT_Q);
                ns_ntt16_twb[lvl][b] = (uint16_t)((uint64_t)zpow[(2*j+1)*step] * R16 % NS_NTT_Q);
            }
    }
}

// Scalar forward NTT (reference king). Standard Cooley-Tukey.
// len: 1, 2, 4, ..., 128. Twiddle for butterfly j at level len:
//   w = zeta^(j * n/(2*len)) = zeta^(j * 128/len)
static void ntt256_scalar(uint32_t a[NS_NTT_N], const uint32_t zpow[NS_NTT_N]) {
    for (int len = 1; len < NS_NTT_N; len <<= 1) {
        int step = NS_NTT_N / (2 * len);
        for (int i = 0; i < NS_NTT_N; i += 2 * len) {
            for (int j = 0; j < len; ++j) {
                uint32_t w = zpow[j * step];
                uint32_t u = a[i + j];
                uint32_t t = barrett_reduce(a[i + j + len] * w);
                a[i + j] = u + t;
                if (a[i + j] >= NS_NTT_Q) a[i + j] -= NS_NTT_Q;
                a[i + j + len] = u + NS_NTT_Q - t;
                if (a[i + j + len] >= NS_NTT_Q) a[i + j + len] -= NS_NTT_Q;
            }
        }
    }
}

// Scalar inverse NTT. Gentleman-Sande, then scale by n^(-1).
static void intt256_scalar(uint32_t a[NS_NTT_N], const uint32_t zpow[NS_NTT_N]) {
    // Precompute inverse powers: zeta^(-i) = zeta^(256-i) mod q
    uint32_t izpow[NS_NTT_N];
    izpow[0] = 1;
    for (int i = 1; i < NS_NTT_N; ++i)
        izpow[i] = zpow[NS_NTT_N - i];

    for (int len = NS_NTT_N / 2; len >= 1; len >>= 1) {
        int step = NS_NTT_N / (2 * len);
        for (int i = 0; i < NS_NTT_N; i += 2 * len) {
            for (int j = 0; j < len; ++j) {
                uint32_t w = izpow[j * step];
                uint32_t u = a[i + j];
                uint32_t v = a[i + j + len];
                a[i + j] = u + v;
                if (a[i + j] >= NS_NTT_Q) a[i + j] -= NS_NTT_Q;
                uint32_t diff = (u + NS_NTT_Q - v) % NS_NTT_Q;
                a[i + j + len] = barrett_reduce(diff * w);
            }
        }
    }
    for (int i = 0; i < NS_NTT_N; ++i)
        a[i] = barrett_reduce(a[i] * NS_NTT_NINV);
}

// --- AVX-512 NTT ---

#ifdef __AVX512F__

static inline HWY_ATTR __m512i barrett_reduce_vec(__m512i x) {
    const __m512i v = _mm512_set1_epi32(314);
    const __m512i q = _mm512_set1_epi32(NS_NTT_Q);
    __m512i t = _mm512_mullo_epi32(x, v);
    t = _mm512_srli_epi32(t, 20);
    __m512i r = _mm512_sub_epi32(x, _mm512_mullo_epi32(t, q));
    t = _mm512_mullo_epi32(r, v);
    t = _mm512_srli_epi32(t, 20);
    r = _mm512_sub_epi32(r, _mm512_mullo_epi32(t, q));
    __mmask16 m = _mm512_cmpge_epi32_mask(r, q);
    r = _mm512_mask_sub_epi32(r, m, r, q);
    return r;
}

// Vectorized forward NTT-256.
// For each level, precompute per-block twiddles into a contiguous buffer,
// then process 16 butterflies at once. Levels with len < 16 fall back to
// scalar (too few butterflies per block to fill a vector).
static HWY_ATTR void ntt256_avx512(uint32_t a[NS_NTT_N], const uint32_t zpow[NS_NTT_N]) {
    const __m512i vq = _mm512_set1_epi32(NS_NTT_Q);

    for (int len = 1; len < NS_NTT_N; len <<= 1) {
        int step = NS_NTT_N / (2 * len);

        // Precompute twiddles for this level once (all blocks share them).
        uint32_t tw_buf[NS_NTT_N / 2];
        if (len >= 16) {
            for (int j = 0; j < len; ++j)
                tw_buf[j] = zpow[j * step];
        }

        for (int i = 0; i < NS_NTT_N; i += 2 * len) {
            if (len >= 16) {
                for (int j = 0; j < len; j += 16) {
                    __m512i u = _mm512_loadu_si512((__m512i*)(a + i + j));
                    __m512i v = _mm512_loadu_si512((__m512i*)(a + i + j + len));
                    __m512i w = _mm512_loadu_si512((__m512i*)(tw_buf + j));
                    __m512i t = barrett_reduce_vec(_mm512_mullo_epi32(v, w));
                    __m512i lo = _mm512_add_epi32(u, t);
                    __mmask16 mlo = _mm512_cmpge_epi32_mask(lo, vq);
                    lo = _mm512_mask_sub_epi32(lo, mlo, lo, vq);
                    __m512i hi = _mm512_sub_epi32(_mm512_add_epi32(u, vq), t);
                    __mmask16 mhi = _mm512_cmpge_epi32_mask(hi, vq);
                    hi = _mm512_mask_sub_epi32(hi, mhi, hi, vq);
                    _mm512_storeu_si512((__m512i*)(a + i + j), lo);
                    _mm512_storeu_si512((__m512i*)(a + i + j + len), hi);
                }
            } else {
                for (int j = 0; j < len; ++j) {
                    uint32_t w = zpow[j * step];
                    uint32_t u = a[i + j];
                    uint32_t t = barrett_reduce(a[i + j + len] * w);
                    a[i + j] = u + t;
                    if (a[i + j] >= NS_NTT_Q) a[i + j] -= NS_NTT_Q;
                    a[i + j + len] = u + NS_NTT_Q - t;
                    if (a[i + j + len] >= NS_NTT_Q) a[i + j + len] -= NS_NTT_Q;
                }
            }
        }
    }
}

// Vectorized inverse NTT
static HWY_ATTR void intt256_avx512(uint32_t a[NS_NTT_N], const uint32_t zpow[NS_NTT_N]) {
    const __m512i vq = _mm512_set1_epi32(NS_NTT_Q);
    const __m512i vninv = _mm512_set1_epi32(NS_NTT_NINV);

    uint32_t izpow[NS_NTT_N];
    izpow[0] = 1;
    for (int i = 1; i < NS_NTT_N; ++i)
        izpow[i] = zpow[NS_NTT_N - i];

    for (int len = NS_NTT_N / 2; len >= 1; len >>= 1) {
        int step = NS_NTT_N / (2 * len);

        uint32_t tw_buf[NS_NTT_N / 2];
        if (len >= 16) {
            for (int j = 0; j < len; ++j)
                tw_buf[j] = izpow[j * step];
        }

        for (int i = 0; i < NS_NTT_N; i += 2 * len) {
            if (len >= 16) {
                for (int j = 0; j < len; j += 16) {
                    __m512i u = _mm512_loadu_si512((__m512i*)(a + i + j));
                    __m512i v = _mm512_loadu_si512((__m512i*)(a + i + j + len));
                    __m512i w = _mm512_loadu_si512((__m512i*)(tw_buf + j));
                    __m512i lo = _mm512_add_epi32(u, v);
                    __mmask16 mlo = _mm512_cmpge_epi32_mask(lo, vq);
                    lo = _mm512_mask_sub_epi32(lo, mlo, lo, vq);
                    __m512i diff = _mm512_sub_epi32(_mm512_add_epi32(u, vq), v);
                    __mmask16 md = _mm512_cmpge_epi32_mask(diff, vq);
                    diff = _mm512_mask_sub_epi32(diff, md, diff, vq);
                    __m512i hi = barrett_reduce_vec(_mm512_mullo_epi32(diff, w));
                    _mm512_storeu_si512((__m512i*)(a + i + j), lo);
                    _mm512_storeu_si512((__m512i*)(a + i + j + len), hi);
                }
            } else {
                for (int j = 0; j < len; ++j) {
                    uint32_t w = izpow[j * step];
                    uint32_t u = a[i + j];
                    uint32_t v = a[i + j + len];
                    a[i + j] = u + v;
                    if (a[i + j] >= NS_NTT_Q) a[i + j] -= NS_NTT_Q;
                    uint32_t diff = (u + NS_NTT_Q - v) % NS_NTT_Q;
                    a[i + j + len] = barrett_reduce(diff * w);
                }
            }
        }
    }
    // Vectorized scaling by n^(-1) mod q
    for (int i = 0; i < NS_NTT_N; i += 16) {
        __m512i v = _mm512_loadu_si512((__m512i*)(a + i));
        v = barrett_reduce_vec(_mm512_mullo_epi32(v, vninv));
        _mm512_storeu_si512((__m512i*)(a + i), v);
    }
}

// 16x16 epi32 in-register transpose: r[i] swaps row<->column.
// 4-stage network: unpack32 -> unpack64 -> 2x shuffle_i32x4 rounds.
static inline HWY_ATTR void ns_transpose16(__m512i r[16]) {
    __m512i t[16];
    for (int i = 0; i < 8; ++i) {
        t[2*i]   = _mm512_unpacklo_epi32(r[2*i], r[2*i+1]);
        t[2*i+1] = _mm512_unpackhi_epi32(r[2*i], r[2*i+1]);
    }
    for (int i = 0; i < 4; ++i) {
        r[4*i+0] = _mm512_unpacklo_epi64(t[4*i+0], t[4*i+2]);
        r[4*i+1] = _mm512_unpackhi_epi64(t[4*i+0], t[4*i+2]);
        r[4*i+2] = _mm512_unpacklo_epi64(t[4*i+1], t[4*i+3]);
        r[4*i+3] = _mm512_unpackhi_epi64(t[4*i+1], t[4*i+3]);
    }
    for (int i = 0; i < 4; ++i) {
        t[i]    = _mm512_shuffle_i32x4(r[i],   r[i+4],  0x88);
        t[i+4]  = _mm512_shuffle_i32x4(r[i],   r[i+4],  0xdd);
        t[i+8]  = _mm512_shuffle_i32x4(r[i+8], r[i+12], 0x88);
        t[i+12] = _mm512_shuffle_i32x4(r[i+8], r[i+12], 0xdd);
    }
    for (int i = 0; i < 8; ++i) {
        r[i]   = _mm512_shuffle_i32x4(t[i], t[i+8], 0x88);
        r[i+8] = _mm512_shuffle_i32x4(t[i], t[i+8], 0xdd);
    }
}

// Montgomery multiply mod q in epi16 lanes: r = v*w*R^-1 mod q, canonical [0,q).
// w must be pre-scaled (w*R mod q). mullo+mulhi_epi16 are single-uop on ICL/TGL;
// the classic signed Montgomery: r = hi(v*w) - hi(lo(v*w)*qinv * q), in (-q,q),
// then one mask-add fixes to [0,q).
static inline HWY_ATTR __m512i ns_fqmul16(__m512i v, __m512i w,
                                          __m512i vq, __m512i vqinv) {
    __m512i p_lo = _mm512_mullo_epi16(v, w);
    __m512i p_hi = _mm512_mulhi_epi16(v, w);
    __m512i m = _mm512_mullo_epi16(p_lo, vqinv);
    __m512i r = _mm512_sub_epi16(p_hi, _mm512_mulhi_epi16(m, vq));
    __mmask32 neg = _mm512_cmpgt_epi16_mask(_mm512_setzero_si512(), r);
    return _mm512_mask_add_epi16(r, neg, r, vq);
}

// Batch-16 forward NTT, epi16 edition.
// 16 independent n=256 polynomials, row-major epi32 input (in[m*256+k]).
// Memory side stays epi32 (contiguous loads + 16x16 transpose); registers are
// then packed to epi16: reg[c] lanes 0-15 = coef 2c of polys 0-15, lanes 16-31
// = coef 2c+1. Each ZMM butterfly covers TWO coefficient butterflies, so the
// 16-poly batch uses 128 regs instead of 256. Level len=1 pairs coefs inside
// each reg's 256-bit halves (w = zpow[0] = 1, no multiply needed); len>=2
// levels pair whole regs via the schedule tables. Store path unpacks back to
// epi32 and reuses the same transpose.
static HWY_ATTR void ns_ntt_batch_16(uint32_t* out, const uint32_t* in,
                            const uint32_t zpow[NS_NTT_N]) {
    (void)zpow;  // schedule tables (incl. Montgomery twiddles) come from init
    const __m512i vq = _mm512_set1_epi16(NS_NTT_Q);
    const __m512i vqm1 = _mm512_set1_epi16(NS_NTT_Q - 1);
    const __m512i vqinv = _mm512_set1_epi16((int16_t)ns_ntt16_qinv);

    __m512i reg[NS_NTT_N / 2];  // 8KB stack — epi16 packed regs

    // Transposed load: 16 tiles of 16x16 epi32, pack coef pairs into epi16.
    for (int k0 = 0; k0 < NS_NTT_N; k0 += 16) {
        __m512i t[16];
        for (int m = 0; m < 16; ++m)
            t[m] = _mm512_loadu_si512((const __m512i*)(in + m * NS_NTT_N + k0));
        ns_transpose16(t);
        for (int j = 0; j < 16; j += 2) {
            __m256i e = _mm512_cvtepi32_epi16(t[j]);      // coef k0+j, 16 polys
            __m256i o = _mm512_cvtepi32_epi16(t[j + 1]);  // coef k0+j+1
            reg[(k0 + j) / 2] =
                _mm512_inserti64x4(_mm512_castsi256_si512(e), o, 1);
        }
    }

    // Level len=1: butterfly (coef 2c, coef 2c+1) lives inside reg[c]'s halves.
    // w = zpow[0] = 1 so t = v directly; 256-bit ops suffice.
    {
        const __m256i vq2 = _mm256_set1_epi16(NS_NTT_Q);
        const __m256i vqm12 = _mm256_set1_epi16(NS_NTT_Q - 1);
        for (int c = 0; c < NS_NTT_N / 2; ++c) {
            __m256i u = _mm512_extracti64x4_epi64(reg[c], 0);
            __m256i v = _mm512_extracti64x4_epi64(reg[c], 1);
            __m256i l = _mm256_add_epi16(u, v);
            l = _mm256_mask_sub_epi16(l, _mm256_cmpgt_epi16_mask(l, vqm12), l, vq2);
            __m256i h = _mm256_sub_epi16(_mm256_add_epi16(u, vq2), v);
            h = _mm256_mask_sub_epi16(h, _mm256_cmpgt_epi16_mask(h, vqm12), h, vq2);
            reg[c] = _mm512_inserti64x4(_mm512_castsi256_si512(l), h, 1);
        }
    }

    // Levels len=2..128 (lvl 1..7): whole-reg pairs, 64 butterflies each,
    // unrolled x4. Per reg pair: lanes 0-15 run the even-coef butterfly
    // (twiddle ta), lanes 16-31 the odd-coef one (twiddle tb).
    for (int lvl = 1; lvl < 8; ++lvl) {
        const uint16_t* li = ns_ntt16_lo[lvl];
        const uint16_t* hi = ns_ntt16_hi[lvl];
        const uint16_t* ta = ns_ntt16_twa[lvl];
        const uint16_t* tb = ns_ntt16_twb[lvl];
        for (int b = 0; b < 64; b += 4) {
            __m512i u0 = reg[li[b+0]], u1 = reg[li[b+1]],
                    u2 = reg[li[b+2]], u3 = reg[li[b+3]];
            __m512i v0 = reg[hi[b+0]], v1 = reg[hi[b+1]],
                    v2 = reg[hi[b+2]], v3 = reg[hi[b+3]];
            __m512i w0 = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_set1_epi16((int16_t)ta[b+0])),
                _mm256_set1_epi16((int16_t)tb[b+0]), 1);
            __m512i w1 = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_set1_epi16((int16_t)ta[b+1])),
                _mm256_set1_epi16((int16_t)tb[b+1]), 1);
            __m512i w2 = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_set1_epi16((int16_t)ta[b+2])),
                _mm256_set1_epi16((int16_t)tb[b+2]), 1);
            __m512i w3 = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_set1_epi16((int16_t)ta[b+3])),
                _mm256_set1_epi16((int16_t)tb[b+3]), 1);
            __m512i t0 = ns_fqmul16(v0, w0, vq, vqinv);
            __m512i t1 = ns_fqmul16(v1, w1, vq, vqinv);
            __m512i t2 = ns_fqmul16(v2, w2, vq, vqinv);
            __m512i t3 = ns_fqmul16(v3, w3, vq, vqinv);
            __m512i l0 = _mm512_add_epi16(u0, t0);
            __m512i l1 = _mm512_add_epi16(u1, t1);
            __m512i l2 = _mm512_add_epi16(u2, t2);
            __m512i l3 = _mm512_add_epi16(u3, t3);
            __m512i h0 = _mm512_sub_epi16(_mm512_add_epi16(u0, vq), t0);
            __m512i h1 = _mm512_sub_epi16(_mm512_add_epi16(u1, vq), t1);
            __m512i h2 = _mm512_sub_epi16(_mm512_add_epi16(u2, vq), t2);
            __m512i h3 = _mm512_sub_epi16(_mm512_add_epi16(u3, vq), t3);
            l0 = _mm512_mask_sub_epi16(l0, _mm512_cmpgt_epi16_mask(l0, vqm1), l0, vq);
            l1 = _mm512_mask_sub_epi16(l1, _mm512_cmpgt_epi16_mask(l1, vqm1), l1, vq);
            l2 = _mm512_mask_sub_epi16(l2, _mm512_cmpgt_epi16_mask(l2, vqm1), l2, vq);
            l3 = _mm512_mask_sub_epi16(l3, _mm512_cmpgt_epi16_mask(l3, vqm1), l3, vq);
            h0 = _mm512_mask_sub_epi16(h0, _mm512_cmpgt_epi16_mask(h0, vqm1), h0, vq);
            h1 = _mm512_mask_sub_epi16(h1, _mm512_cmpgt_epi16_mask(h1, vqm1), h1, vq);
            h2 = _mm512_mask_sub_epi16(h2, _mm512_cmpgt_epi16_mask(h2, vqm1), h2, vq);
            h3 = _mm512_mask_sub_epi16(h3, _mm512_cmpgt_epi16_mask(h3, vqm1), h3, vq);
            reg[li[b+0]] = l0; reg[hi[b+0]] = h0;
            reg[li[b+1]] = l1; reg[hi[b+1]] = h1;
            reg[li[b+2]] = l2; reg[hi[b+2]] = h2;
            reg[li[b+3]] = l3; reg[hi[b+3]] = h3;
        }
    }

    // Transposed store: unpack epi16 pairs to epi32 regs, transpose, store.
    for (int k0 = 0; k0 < NS_NTT_N; k0 += 16) {
        __m512i t[16];
        for (int j = 0; j < 16; j += 2) {
            __m512i v = reg[(k0 + j) / 2];
            t[j]     = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v, 0));
            t[j + 1] = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v, 1));
        }
        ns_transpose16(t);
        for (int m = 0; m < 16; ++m)
            _mm512_storeu_si512((__m512i*)(out + m * NS_NTT_N + k0), t[m]);
    }
}

#endif // __AVX512F__

#endif // NSCRYPT_HPP
