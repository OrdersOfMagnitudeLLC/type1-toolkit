#include "../include/gguf_parser.h"
#include "../include/tokenizer.h"
#include "../include/transformer.h"
#include "../vendor/ns_dequant.h"
#include "../src/ns_q4k_quant.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <cstdio>
#include <algorithm>
#include <unordered_map>
#include <time.h>

extern "C" void profile_add_sampling(uint64_t ns);
extern "C" void profile_print_report(double total_ms);

size_t get_rss_kb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            size_t val = 0;
            sscanf(line.c_str(), "VmRSS: %zu kB", &val);
            return val;
        }
    }
    return 0;
}

auto now() { return std::chrono::high_resolution_clock::now(); }

template<typename T>
double elapsed_ms(T t0) {
    return std::chrono::duration<double,std::milli>(
      std::chrono::high_resolution_clock::now() - t0).count();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <gguf_file> [-c context_len]" << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    std::string tok_path = model_path;
    int context_len = 2048; // default
    bool debug_layers = false;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "-c") {
            context_len = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--tokenizer") {
            tok_path = argv[++i];
        } else if (std::string(argv[i]) == "--debug-layers") {
            debug_layers = true;
        }
    }

    size_t rss_baseline = get_rss_kb();
    printf("[RAM] Baseline:          %zu MB\n", rss_baseline / 1024);

    GGUFParser tok_parser;
    bool is_nsm = model_path.size() >= 4 && model_path.compare(model_path.size() - 4, 4, ".nsm") == 0;
    bool ok = is_nsm ? tok_parser.load_nsm(model_path) : tok_parser.load(tok_path);
    if (!ok) {
        std::cerr << "Failed to load model metadata: " << tok_path << std::endl;
        return 1;
    }

    size_t rss_after_parser = get_rss_kb();
    printf("[RAM] After parser load: %zu MB  (delta: +%zu MB)\n", rss_after_parser / 1024, (rss_after_parser - rss_baseline) / 1024);

    const ModelParams& params = tok_parser.params();

    // Diagnostic: print all metadata keys starting with "qwen2." or containing "vocab"
    std::cout << "=== Metadata Keys ===" << std::endl;
    std::cout << "Keys starting with 'qwen2.':" << std::endl;
    for (const auto& kv : tok_parser.metadata()) {
        if (kv.first.find("qwen2.") == 0) {
            std::cout << "  " << kv.first << std::endl;
        }
    }
    std::cout << "Keys containing 'vocab':" << std::endl;
    for (const auto& kv : tok_parser.metadata()) {
        if (kv.first.find("vocab") != std::string::npos) {
            std::cout << "  " << kv.first << std::endl;
        }
    }
    std::cout << "Keys containing 'merg':" << std::endl;
    for (const auto& kv : tok_parser.metadata()) {
        if (kv.first.find("merg") != std::string::npos) {
            std::cout << "  " << kv.first << std::endl;
        }
    }
    std::cout << std::endl;

    std::cout << "=== Model Parameters ===" << std::endl;
    std::cout << "Architecture: " << params.architecture << std::endl;
    std::cout << "Key prefix: " << params.key_prefix << std::endl;
    std::cout << "n_layers: " << params.n_layers << std::endl;
    std::cout << "n_heads: " << params.n_heads << std::endl;
    std::cout << "n_kv_heads: " << params.n_kv_heads << std::endl;
    std::cout << "n_embd: " << params.n_embd << std::endl;
    std::cout << "n_ff: " << params.n_ff << std::endl;
    std::cout << "n_vocab: " << params.n_vocab << std::endl;
    std::cout << "head_dim: " << params.head_dim << std::endl;
    std::cout << "max_seq_len: " << params.max_seq_len << std::endl;
    std::cout << "rope_freq_base: " << params.rope_freq_base << std::endl;
    std::cout << "rope_freq_scale: " << params.rope_freq_scale << std::endl;
    std::cout << std::endl;

    const std::vector<GGUFTensor>& tensors = tok_parser.tensors();
    std::cout << "=== Tensors ===" << std::endl;
    std::cout << "Total tensor count: " << tensors.size() << std::endl;
    std::cout << std::endl;

    std::cout << "All tensors:" << std::endl;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const GGUFTensor& tensor = tensors[i];
        std::cout << "  " << tensor.name << " [";
        for (size_t j = 0; j < tensor.shape.size(); ++j) {
            std::cout << tensor.shape[j];
            if (j < tensor.shape.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;

    {
        std::unordered_map<int, int> type_counts;
        for (const auto& t : tensors) type_counts[(int)t.type]++;
        std::cout << "=== Tensor type histogram ===" << std::endl;
        for (const auto& kv : type_counts) {
            std::cout << "  type=" << kv.first << " count=" << kv.second << std::endl;
        }
        std::cout << std::endl;
    }

    // Test tokenizer
    std::cout << "=== Tokenizer Test ===" << std::endl;
    BPETokenizer tokenizer;
    if (!tokenizer.load(tok_parser)) {
        std::cerr << "Failed to load tokenizer" << std::endl;
        return 1;
    }
    if (tok_parser.merges().empty()) {
        size_t last = tok_path.find_last_of("/\\");
        std::string dir = (last == std::string::npos) ? "." : tok_path.substr(0, last + 1);
        std::string merges_path = dir + "merges.txt";
        std::ifstream mf(merges_path);
        if (mf.good()) {
            mf.close();
            tokenizer.load_merges_file(merges_path);
        }
    }

    size_t rss_after_tokenizer = get_rss_kb();
    printf("[RAM] After tokenizer:   %zu MB  (delta: +%zu MB)\n", rss_after_tokenizer / 1024, (rss_after_tokenizer - rss_after_parser) / 1024);

    std::string test_text = "Hello, world! This is NSRun.";
    std::cout << "Test text: " << test_text << std::endl;

    std::vector<int32_t> token_ids = tokenizer.encode(test_text);
    std::cout << "Token ids: ";
    for (size_t i = 0; i < token_ids.size(); ++i) {
        std::cout << token_ids[i];
        if (i < token_ids.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;

    std::string decoded = tokenizer.decode(token_ids);
    std::cout << "Decoded: " << decoded << std::endl;

    std::cout << "Vocab size: " << tokenizer.vocab_size() << std::endl;
    std::cout << "BOS token id: " << tokenizer.bos_token_id() << std::endl;
    std::cout << "EOS token id: " << tokenizer.eos_token_id() << std::endl;
    std::cout << std::endl;

    // Test transformer generation
    std::cout << "=== Transformer Test ===" << std::endl;
    Transformer transformer;
    if (!transformer.load(tok_parser, context_len)) {
        std::cerr << "Failed to load transformer" << std::endl;
        return 1;
    }

    transformer.set_debug_layers(debug_layers);

    size_t rss_after_transformer = get_rss_kb();
    printf("[RAM] After transformer load: %zu MB  (delta: +%zu MB)\n", rss_after_transformer / 1024, (rss_after_transformer - rss_after_tokenizer) / 1024);



    // Optional --repack-ram mode: repack every tensor and print RSS, then exit.
    // This confirms the lazy per-tensor repack memory usage before running the
    // full generation benchmark.
    bool repack_ram = (argc >= 3 && std::string(argv[2]) == "--repack-ram");
    if (repack_ram) {
        size_t rss_before = get_rss_kb();
        transformer.repack_all_weights();
        size_t rss_after = get_rss_kb();
        printf("[RAM] Before repack all: %zu MB\n", rss_before / 1024);
        printf("[RAM] After repack all:  %zu MB  (delta: +%zu MB)\n",
               rss_after / 1024, (rss_after - rss_before) / 1024);
        return 0;
    }

    // Validate the Q4_Kx8 repack layout before any generation.
    if (!transformer.validate_repack_q4k()) {
        std::cerr << "[VALIDATE] repack_q4k validation FAILED - aborting" << std::endl;
        return 1;
    }


    // Diagnostic: compare first 16 values of blk.0.ffn_gate.weight (only for .nsm F16 companion)
    if (tok_parser.data_ptr() == nullptr) {
        const char* name = "blk.0.ffn_gate.weight";
        std::vector<float> nsm_buf;
        const float* nsm = transformer.get_weight_f32(name, nsm_buf);
        const GGUFTensor* ref = nullptr;
        for (const auto& t : tok_parser.tensors()) {
            if (t.name == name) { ref = &t; break; }
        }
        if (nsm && ref) {
            size_t nread = 256;
            std::vector<uint8_t> raw(nread * sizeof(uint16_t));
            tok_parser.read_tensor(*ref, raw.data(), raw.size());
            std::vector<float> f32(nread);
            const uint16_t* raw16 = (const uint16_t*)raw.data();
            for (size_t i = 0; i < nread; ++i) f32[i] = Transformer::f16_to_f32(raw16[i]);
            block_q4_K b;
            quantize_block_q4_K(f32.data(), &b, nread);
            std::vector<float> q4k_f32(nread);
            dequantize_row_q4_K(&b, q4k_f32.data(), (int)nread);
            std::cout << "[VALUE] " << name << " first 16" << std::endl;
            std::cout << "  NSM: ";
            for (size_t i = 0; i < 16; ++i) std::cout << nsm[i] << " ";
            std::cout << std::endl;
            std::cout << "  GGUF Q4K roundtrip: ";
            for (size_t i = 0; i < 16; ++i) std::cout << q4k_f32[i] << " ";
            std::cout << std::endl;
        } else {
            std::cout << "[VALUE] " << name << " not found" << std::endl;
        }
    }

    // Validate vec_dot_q4k correctness before generation
    // if (!transformer.validate_vec_dot_q6k()) {
    //     std::cerr << "[WARN] vec_dot_q6k validation mismatch - continuing anyway" << std::endl;
    // }

    // Validate vec_dot_q4k correctness before generation
    // if (!transformer.validate_vec_dot_q4k()) {
    //     std::cerr << "[VALIDATE] vec_dot_q4k validation FAILED - aborting generation" << std::endl;
    //     return 1;
    // }

    auto run_benchmark = [&](const std::string& benchmark_name, const std::string& user_prompt) {
        std::string full_prompt = user_prompt;

        // Tokenize full prompt and prepend BOS
        auto tokens = tokenizer.encode(full_prompt);
        tokens.insert(tokens.begin(), tokenizer.bos_token_id());
        int prompt_len = tokens.size();

        printf("\n=== %s ===\n", benchmark_name.c_str());
        printf("Prompt tokens: %d\n", prompt_len);

        // Prefill — batched: every weight row is dequantized ONCE and reused
        // against all prompt_len positions, instead of once per token.
        auto t_prefill = now();
        InferenceState state;
        transformer.forward_batch(tokens, 0, state, /*compute_logits_last_only=*/true);
        double prefill_ms = elapsed_ms(t_prefill);

        // Generate 20 tokens
        auto t_gen = now();
        int pos = prompt_len;
        printf("--- Output ---\n");
        for (int i = 0; i < 20; i++) {
            // Argmax sampling: compute_logits() already stored the winning token.
            struct timespec ts0, ts1;
            clock_gettime(CLOCK_MONOTONIC, &ts0);
            int32_t next = state.sampled_token;
            clock_gettime(CLOCK_MONOTONIC, &ts1);
            uint64_t ns = (uint64_t)(ts1.tv_sec - ts0.tv_sec) * 1000000000ULL + (uint64_t)(ts1.tv_nsec - ts0.tv_nsec);
            profile_add_sampling(ns);

            // Check token_id validity
            if (next < 0 || (size_t)next >= (size_t)tokenizer.vocab_size()) {
                fprintf(stderr, "FATAL: Invalid token_id %d (vocab size %zu)\n", next, (size_t)tokenizer.vocab_size());
                break;
            }

            if (tokenizer.is_stop_token(next)) {
                break;
            }

            std::string tok = tokenizer.decode({next});
            printf("%s", tok.c_str()); fflush(stdout);
            transformer.forward(next, pos++, state);
            if (i == 0 && !state.logits.empty()) {
                std::cout << "[LOGIT TOP5] " << benchmark_name
                          << " after token " << next
                          << " \"" << tok << "\"" << std::endl;
                size_t n = state.logits.size();
                std::vector<size_t> idx(n);
                for (size_t j = 0; j < n; ++j) idx[j] = j;
                std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                                  [&](size_t a, size_t b){ return state.logits[a] > state.logits[b]; });
                for (size_t k = 0; k < 5; ++k) {
                    std::cout << "  token=" << idx[k]
                              << " logit=" << state.logits[idx[k]] << std::endl;
                }
            }
        }
        double gen_ms = elapsed_ms(t_gen);

        printf("\n[PERF]   Prefill:  %6.1f ms / %d tokens = %5.1f t/s\n",
               prefill_ms, prompt_len, prompt_len * 1000.0 / prefill_ms);
        printf("[PERF]   Generate: %6.1f ms / 20 tokens = %5.1f t/s\n",
               gen_ms, 20000.0 / gen_ms);
        profile_print_report(gen_ms);
    };

    // Benchmark 1: Short prompt (~48 tokens)
    run_benchmark("Short Prompt Benchmark",
                  "<|im_start|>system\n"
                  "You are a helpful assistant.07\n"
                  "<|im_start|>user\n"
                  "What is 2+2? Answer in English.07\n"
                  "<|im_start|>assistant\n");

    printf("\n[RAM]    After generate: %zu MB\n", get_rss_kb()/1024);

    return 0;
}
