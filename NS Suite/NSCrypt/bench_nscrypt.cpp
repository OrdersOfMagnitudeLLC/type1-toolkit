// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

// NSCrypt benchmark: schema-aware AEAD + NTT-256 acceleration.
// Kings: OpenSSL EVP_chacha20_poly1305 (AEAD), scalar NTT (portable C).

#include "nscrypt.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>

#ifdef USE_OPENSSL
#include <openssl/evp.h>
#pragma comment(lib, "crypto")
#endif

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// ---- AEAD benchmark ----

static void bench_aead() {
    printf("== Schema-aware AEAD (ChaCha20-Poly1305) ==\n\n");

    std::mt19937 rng(42);
    uint8_t key[32], nonce[12];
    for (int i = 0; i < 32; ++i) key[i] = (uint8_t)rng();
    for (int i = 0; i < 12; ++i) nonce[i] = (uint8_t)rng();

    // FIX 4.1 message shape: 64 bytes, ~40% fixed fields.
    // Tag=35 (BeginString), Tag=49 (SenderCompID), Tag=56 (TargetCompID)
    // are fixed/known. Tag=34 (MsgSeqNum), Tag=52 (SendingTime), body are variable.
    // Layout: [0..7] BeginString "FIX4.1\0\0" (8B AUTH_ONLY)
    //         [8..15] SenderCompID "SENDER\0\0" (8B AUTH_ONLY)
    //         [16..23] TargetCompID "TARGET\0\0" (8B AUTH_ONLY)
    //         [24..31] MsgType + body part 1 (8B ENCRYPT)
    //         [32..39] MsgSeqNum + SendingTime (8B ENCRYPT)
    //         [40..63] Body part 2 (24B ENCRYPT)
    // Total: 24B AUTH_ONLY (37.5%), 40B ENCRYPT (62.5%)

    NSCryptField fields[] = {
        {0,  8, NS_FIELD_AUTH_ONLY},
        {8,  8, NS_FIELD_AUTH_ONLY},
        {16, 8, NS_FIELD_AUTH_ONLY},
        {24, 8, NS_FIELD_ENCRYPT},
        {32, 8, NS_FIELD_ENCRYPT},
        {40, 24, NS_FIELD_ENCRYPT},
    };
    NSCryptSchema schema = {fields, 6};

    const size_t pt_len = 64;
    uint8_t plaintext[64], plaintext2[64];
    for (int i = 0; i < 64; ++i) plaintext[i] = (uint8_t)rng();

    // Correctness: encrypt + decrypt round-trip
    uint8_t ct[128];
    size_t ct_len = ns_crypt_encrypt(key, nonce, schema, plaintext, pt_len, ct, 128);
    if (ct_len == 0) { printf("ENCRYPT FAILED\n"); return; }

    memcpy(plaintext2, plaintext, pt_len);  // AUTH_ONLY fields must be present
    int ok = ns_crypt_decrypt(key, nonce, schema, ct, ct_len, plaintext2, pt_len);
    if (!ok) { printf("DECRYPT FAILED (auth)\n"); return; }
    if (memcmp(plaintext, plaintext2, pt_len) != 0) {
        printf("ROUNDTRIP MISMATCH\n"); return;
    }
    printf("Round-trip: OK  (ct_len=%zu, plaintext=%zu, saved=%zu bytes)\n",
           ct_len, pt_len, pt_len - ct_len + 16);

    // Throughput: NSCrypt vs OpenSSL (full 64B encrypt)
    const size_t N = 100000;
    uint8_t* pts = (uint8_t*)aligned_alloc(64, N * pt_len);
    uint8_t* cts = (uint8_t*)aligned_alloc(64, N * 80);
    for (size_t i = 0; i < N * pt_len; ++i) pts[i] = (uint8_t)rng();

    // NSCrypt
    double t0 = now_s();
    for (size_t i = 0; i < N; ++i)
        ns_crypt_encrypt(key, nonce, schema, pts + i * pt_len, pt_len,
                         cts + i * 80, 80);
    double ns_s = now_s() - t0;
    double ns_throughput = (double)N / ns_s / 1e6;  // M ops/s
    double ns_gbps = (double)N * pt_len / ns_s / 1e9;

    printf("NSCrypt:  %.0f ops/s  %.2f Gbps  (encrypts %zu of %zu bytes)\n",
           ns_throughput, ns_gbps, ct_len - 16, pt_len);

#ifdef USE_OPENSSL
    // OpenSSL: full encrypt (all 64 bytes)
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    const EVP_CIPHER* cipher = EVP_chacha20_poly1305();
    uint8_t* ossl_ct = (uint8_t*)aligned_alloc(64, N * 80);
    uint8_t ossl_tag[16];

    t0 = now_s();
    for (size_t i = 0; i < N; ++i) {
        int outl;
        EVP_EncryptInit_ex(ctx, cipher, nullptr, key, nonce);
        EVP_EncryptUpdate(ctx, ossl_ct + i * 80, &outl,
                          pts + i * pt_len, (int)pt_len);
        int finl;
        EVP_EncryptFinal_ex(ctx, ossl_ct + i * 80 + outl, &finl);
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_GET_TAG, 16, ossl_tag);
    }
    double ossl_s = now_s() - t0;
    double ossl_throughput = (double)N / ossl_s / 1e6;
    double ossl_gbps = (double)N * pt_len / ossl_s / 1e9;

    printf("OpenSSL:  %.0f ops/s  %.2f Gbps  (encrypts %zu of %zu bytes)\n",
           ossl_throughput, ossl_gbps, pt_len, pt_len);
    printf("Speedup:  %.2fx throughput  (%.2fx per-byte)\n",
           ns_throughput / ossl_throughput,
           (ns_throughput / ossl_throughput) * (double)pt_len / (ct_len - 16));

    EVP_CIPHER_CTX_free(ctx);
    free(ossl_ct);
#else
    printf("OpenSSL:  (not compiled, use -DUSE_OPENSSL -lcrypto)\n");
#endif

    free(pts); free(cts);

    printf("\nLoss case: random payload with no schema — all bytes ENCRYPT,\n");
    printf("no savings. Use standard AEAD. Same pattern as the rest of the suite.\n");
}

// ---- NTT benchmark ----

static void bench_ntt() {
    printf("\n== NTT-256 (Kyber/ML-KEM, Zq=3329) ==\n\n");

    uint32_t zpow[NS_NTT_N];
    ntt_init_twiddles(zpow);

    // Correctness: NTT then INTT should recover original
    uint32_t orig[NS_NTT_N], a[NS_NTT_N], a_ref[NS_NTT_N];
    for (int i = 0; i < NS_NTT_N; ++i)
        orig[i] = a[i] = a_ref[i] = (uint32_t)(i * 7 + 13) % NS_NTT_Q;

    // Scalar reference
    ntt256_scalar(a_ref, zpow);
    intt256_scalar(a_ref, zpow);

    bool ok = true;
    for (int i = 0; i < NS_NTT_N; ++i)
        if (a_ref[i] != orig[i]) { ok = false; break; }
    printf("Scalar NTT->INTT round-trip: %s\n", ok ? "OK" : "FAIL");

#ifdef __AVX512F__
    uint32_t b[NS_NTT_N];
    for (int i = 0; i < NS_NTT_N; ++i)
        b[i] = (uint32_t)(i * 7 + 13) % NS_NTT_Q;

    ntt256_avx512(b, zpow);
    // Compare forward NTT output with scalar
    uint32_t c[NS_NTT_N];
    for (int i = 0; i < NS_NTT_N; ++i)
        c[i] = (uint32_t)(i * 7 + 13) % NS_NTT_Q;
    ntt256_scalar(c, zpow);

    bool ntt_ok = true;
    for (int i = 0; i < NS_NTT_N; ++i)
        if (b[i] != c[i]) { ntt_ok = false; break; }
    printf("AVX-512 NTT vs scalar: %s\n", ntt_ok ? "OK" : "FAIL");

    intt256_avx512(b, zpow);
    bool intt_ok = true;
    for (int i = 0; i < NS_NTT_N; ++i)
        if (b[i] != orig[i]) { intt_ok = false; break; }
    printf("AVX-512 NTT->INTT round-trip: %s\n", intt_ok ? "OK" : "FAIL");
#endif

    // Throughput: single NTT
    const size_t BATCH = 10000;
    uint32_t* buf = (uint32_t*)aligned_alloc(64, BATCH * NS_NTT_N * 4);
    for (size_t i = 0; i < BATCH * NS_NTT_N; ++i)
        buf[i] = (uint32_t)(i * 31 + 17) % NS_NTT_Q;

    // Scalar
    double t0 = now_s();
    for (size_t i = 0; i < BATCH; ++i)
        ntt256_scalar(buf + i * NS_NTT_N, zpow);
    double scalar_s = now_s() - t0;
    double scalar_ops = (double)BATCH / scalar_s / 1e6;

    printf("Scalar:   %.0f NTT/s  (%.2f us/call)\n",
           scalar_ops, scalar_s / BATCH * 1e6);

#ifdef __AVX512F__
    // Reset buffer
    for (size_t i = 0; i < BATCH * NS_NTT_N; ++i)
        buf[i] = (uint32_t)(i * 31 + 17) % NS_NTT_Q;

    t0 = now_s();
    for (size_t i = 0; i < BATCH; ++i)
        ntt256_avx512(buf + i * NS_NTT_N, zpow);
    double avx_s = now_s() - t0;
    double avx_ops = (double)BATCH / avx_s / 1e6;

    printf("AVX-512:  %.0f NTT/s  (%.2f us/call)\n",
           avx_ops, avx_s / BATCH * 1e6);
    printf("Speedup:  %.2fx\n", avx_ops / scalar_ops);

    // Batch-16 NTT benchmark
    const size_t NBATCH = 1000;
    uint32_t* batch_in = (uint32_t*)aligned_alloc(64, NBATCH * 16 * NS_NTT_N * 4);
    uint32_t* batch_out = (uint32_t*)aligned_alloc(64, NBATCH * 16 * NS_NTT_N * 4);
    for (size_t i = 0; i < NBATCH * 16 * NS_NTT_N; ++i)
        batch_in[i] = (uint32_t)(i * 17 + 31) % NS_NTT_Q;

    // Correctness: compare batch-16 to 16 scalar NTTs
    uint32_t tmp[16][NS_NTT_N];
    for (int m = 0; m < 16; ++m)
        for (int k = 0; k < NS_NTT_N; ++k)
            tmp[m][k] = batch_in[m * NS_NTT_N + k];
    for (int m = 0; m < 16; ++m)
        ntt256_scalar(tmp[m], zpow);
    ns_ntt_batch_16(batch_out, batch_in, zpow);
    bool batch_ok = true;
    for (int m = 0; m < 16; ++m)
        for (int k = 0; k < NS_NTT_N; ++k)
            if (tmp[m][k] != batch_out[m * NS_NTT_N + k]) { batch_ok = false; break; }
    printf("Batch-16 NTT vs scalar: %s\n", batch_ok ? "OK" : "FAIL");

    // 16x scalar
    t0 = now_s();
    for (size_t b = 0; b < NBATCH; ++b) {
        for (int m = 0; m < 16; ++m) {
            uint32_t* p = batch_in + b * 16 * NS_NTT_N + m * NS_NTT_N;
            ntt256_scalar(p, zpow);  // in-place
        }
    }
    double scalar16_s = now_s() - t0;
    double scalar16_ntts = (double)NBATCH * 16 / scalar16_s / 1e6;

    // Batch-16
    t0 = now_s();
    for (size_t b = 0; b < NBATCH; ++b) {
        uint32_t* p_in = batch_in + b * 16 * NS_NTT_N;
        uint32_t* p_out = batch_out + b * 16 * NS_NTT_N;
        ns_ntt_batch_16(p_out, p_in, zpow);
    }
    double batch16_s = now_s() - t0;
    double batch16_ntts = (double)NBATCH * 16 / batch16_s / 1e6;

    printf("Scalar 16x: %.0f NTT/s  (%.2f us/16)\n",
           scalar16_ntts, scalar16_s / NBATCH * 1e6);
    printf("Batch-16:   %.0f NTT/s  (%.2f us/16)\n",
           batch16_ntts, batch16_s / NBATCH * 1e6);
    printf("Batch-16 speedup: %.2fx\n", batch16_ntts / scalar16_ntts);

    free(batch_in); free(batch_out);
#else
    printf("AVX-512:  (not available)\n");
#endif

    free(buf);
}

int main() {
    printf("NSCrypt benchmark — schema-aware AEAD + NTT-256\n\n");
    bench_aead();
    bench_ntt();
    return 0;
}
