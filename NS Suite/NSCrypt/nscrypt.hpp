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

// Precompute all 256 powers of zeta: zpow[i] = zeta^i mod q
static void ntt_init_twiddles(uint32_t zpow[NS_NTT_N]) {
    zpow[0] = 1;
    for (int i = 1; i < NS_NTT_N; ++i)
        zpow[i] = (uint32_t)((uint64_t)zpow[i-1] * NS_NTT_ZETA % NS_NTT_Q);
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

// Batch-16 forward NTT.
// Takes 16 independent n=256 polynomials in row-major order (in[m*NS_NTT_N + k]
// is NTT m, element k). Transposes so each __m512i register holds the k-th
// element of all 16 NTTs. Every butterfly level is then width=16: one vector
// butterfly operates on 16 independent (u, v) pairs simultaneously. Barrett
// reduction is applied across all 16 lanes at once.
static HWY_ATTR void ns_ntt_batch_16(uint32_t* out, const uint32_t* in,
                            const uint32_t zpow[NS_NTT_N]) {
    const __m512i vq = _mm512_set1_epi32(NS_NTT_Q);
    // Lane m holds NTT m; base offsets are m * NS_NTT_N.
    const __m512i base_idx = _mm512_set_epi32(
        15 * NS_NTT_N, 14 * NS_NTT_N, 13 * NS_NTT_N, 12 * NS_NTT_N,
        11 * NS_NTT_N, 10 * NS_NTT_N,  9 * NS_NTT_N,  8 * NS_NTT_N,
         7 * NS_NTT_N,  6 * NS_NTT_N,  5 * NS_NTT_N,  4 * NS_NTT_N,
         3 * NS_NTT_N,  2 * NS_NTT_N,  1 * NS_NTT_N,  0 * NS_NTT_N);

    __m512i reg[NS_NTT_N];  // 16KB stack — transposed registers

    // Gather-transpose: reg[k][m] = in[m][k]
    for (int k = 0; k < NS_NTT_N; ++k) {
        __m512i idx = _mm512_add_epi32(base_idx, _mm512_set1_epi32(k));
        reg[k] = _mm512_i32gather_epi32(idx, in, 4);
    }

    // All 8 butterfly levels, fully vectorized (16 NTTs in parallel).
    for (int len = 1; len < NS_NTT_N; len <<= 1) {
        int step = NS_NTT_N / (2 * len);
        for (int i = 0; i < NS_NTT_N; i += 2 * len) {
            for (int j = 0; j < len; ++j) {
                __m512i u = reg[i + j];
                __m512i v = reg[i + j + len];
                __m512i w = _mm512_set1_epi32(zpow[j * step]);
                __m512i t = barrett_reduce_vec(_mm512_mullo_epi32(v, w));
                __m512i lo = _mm512_add_epi32(u, t);
                __mmask16 mlo = _mm512_cmpge_epi32_mask(lo, vq);
                lo = _mm512_mask_sub_epi32(lo, mlo, lo, vq);
                __m512i hi = _mm512_sub_epi32(_mm512_add_epi32(u, vq), t);
                __mmask16 mhi = _mm512_cmpge_epi32_mask(hi, vq);
                hi = _mm512_mask_sub_epi32(hi, mhi, hi, vq);
                reg[i + j] = lo;
                reg[i + j + len] = hi;
            }
        }
    }

    // Scatter-transpose: out[m][k] = reg[k][m]
    for (int k = 0; k < NS_NTT_N; ++k) {
        __m512i idx = _mm512_add_epi32(base_idx, _mm512_set1_epi32(k));
        _mm512_i32scatter_epi32(out, idx, reg[k], 4);
    }
}

#endif // __AVX512F__

#endif // NSCRYPT_HPP
