#include "../include/gguf_parser.h"
#include "../include/transformer.h"
#include <cstdint>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstdlib>

using TimePoint = std::chrono::high_resolution_clock::time_point;

auto now() {
    return std::chrono::high_resolution_clock::now();
}

double elapsed_ms(TimePoint t0) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_file.nsm|gguf>" << std::endl;
        return 1;
    }
    const std::string model_path = argv[1];

    GGUFParser parser;
    bool is_nsm = model_path.size() >= 4 && model_path.compare(model_path.size() - 4, 4, ".nsm") == 0;
    if (!(is_nsm ? parser.load_nsm(model_path) : parser.load(model_path))) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return 1;
    }

    Transformer transformer;
    if (!transformer.load(parser, 2048)) {
        std::cerr << "Failed to load transformer" << std::endl;
        return 1;
    }

    const int n_agents = 10;
    const int n_vocab = (int)parser.params().n_vocab;

    // Distinct one-token agents, all at position 0.
    std::vector<std::vector<int32_t>> agent_tokens(n_agents);
    std::vector<int> agent_positions(n_agents, 0);
    for (int i = 0; i < n_agents; ++i) {
        int token_id = (i + 1) % n_vocab;
        agent_tokens[i] = {token_id};
    }

    std::vector<InferenceState> states(n_agents);
    std::vector<InferenceState*> state_ptrs(n_agents);
    for (int i = 0; i < n_agents; ++i) state_ptrs[i] = &states[i];

    std::vector<std::vector<float>> agent_logits;

    // --- HIVE batched pass ---
    auto t0 = now();
    if (!transformer.forward_hive_batch(agent_tokens, agent_positions, state_ptrs, agent_logits)) {
        std::cerr << "forward_hive_batch failed" << std::endl;
        return 1;
    }
    double hive_ms = elapsed_ms(t0);
    double hive_tps = n_agents * 1000.0 / hive_ms;

    // --- Sequential baseline: 10 x one-token forward_batch ---
    std::vector<InferenceState> seq_states(n_agents);
    auto t1 = now();
    for (int i = 0; i < n_agents; ++i) {
        if (!transformer.forward_batch(agent_tokens[i], 0, seq_states[i], true)) {
            std::cerr << "forward_batch failed at agent " << i << std::endl;
            return 1;
        }
    }
    double seq_ms = elapsed_ms(t1);
    double seq_tps = n_agents * 1000.0 / seq_ms;

    std::cout << "\n=== HIVE Benchmark ===" << std::endl;
    std::cout << "HIVE 10-agent:  " << hive_ms << " ms, " << hive_tps << " t/s" << std::endl;
    std::cout << "Sequential 10x: " << seq_ms << " ms, " << seq_tps << " t/s" << std::endl;
    std::cout << "Speed-up:       " << seq_ms / hive_ms << "x" << std::endl;

    std::cout << "\n=== Per-agent top token ===" << std::endl;
    bool distinct = false;
    for (int i = 0; i < n_agents; ++i) {
        if (i >= (int)agent_logits.size() || agent_logits[i].empty()) {
            std::cout << "Agent " << i << ": no logits" << std::endl;
            continue;
        }
        auto it = std::max_element(agent_logits[i].begin(), agent_logits[i].end());
        int top_token = (int)std::distance(agent_logits[i].begin(), it);
        std::cout << "Agent " << i << " top token: " << top_token
                  << " logit: " << *it << std::endl;
        if (i > 0) {
            int prev_top = (int)std::distance(
                agent_logits[i-1].begin(),
                std::max_element(agent_logits[i-1].begin(), agent_logits[i-1].end()));
            if (top_token != prev_top) distinct = true;
        }
    }
    std::cout << "Logits are " << (distinct ? "distinct across agents" : "identical") << std::endl;

    return 0;
}
