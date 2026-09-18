// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

// Compile command:
// g++ -O3 -march=native -std=c++17 -o nsstringindex_bench nsstringindex_bench.cpp

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include "nsstringindex.hpp"

// Generate Apache-style log line using std::string concatenation
// Format: "127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] \"GET /path HTTP/1.1\" STATUS 2326\n"
// STATUS distribution: 80% = "200", 10% = "404", 10% = "500"
std::string generate_log_line(int i) {
    static const char* paths[10] = {
        "/index.html", "/about.html", "/contact.html",
        "/products.html", "/services.html",
        "/blog/post1.html", "/api/users",
        "/images/logo.png", "/css/style.css", "/js/app.js"
    };
    // 8/10 = 200, 1/10 = 404, 1/10 = 500
    const char* status = (i % 10 == 8) ? "404" :
                         (i % 10 == 9) ? "500" : "200";
    return std::string("127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] "
                       "\"GET ") + paths[i % 10] +
          " HTTP/1.1\" " + status + " 2326\n";
}

// Baseline: extract status code field from each line, then search
uint64_t baseline_scan(const char* corpus, size_t len, const char* pattern) {
    uint64_t count = 0;
    const char* pos = corpus;
    const char* end = corpus + len;

    while (pos < end) {
        // Find end of line
        const char* line_end = (const char*)memchr(pos, '\n', end - pos);
        if (!line_end) line_end = end;

        // Find closing quote of HTTP request (second quote in line)
        const char* first_quote = (const char*)memchr(pos, '"', line_end - pos);
        if (first_quote) {
            const char* closing_quote = (const char*)memchr(first_quote + 1, '"', line_end - (first_quote + 1));
            if (closing_quote) {
                // Status code is first whitespace-delimited token after the closing quote
                const char* status_start = closing_quote + 1;
                while (status_start < line_end && *status_start == ' ') status_start++;
                
                const char* status_end = status_start;
                while (status_end < line_end && *status_end != ' ') status_end++;

                // Check if status matches pattern
                if ((size_t)(status_end - status_start) == strlen(pattern) &&
                    strncmp(status_start, pattern, status_end - status_start) == 0) {
                    count++;
                }
            }
        }

        pos = line_end + 1;
    }

    return count;
}

int main() {
    const int NUM_LINES = 5'000'000;
    const int NUM_FIELDS = 9;  // Apache log has 9 fields

    std::cout << "Generating " << NUM_LINES << " Apache log lines..." << std::endl;

    // Generate corpus in memory
    std::string corpus;
    corpus.reserve(NUM_LINES * 150);

    for (int i = 0; i < NUM_LINES; i++) {
        corpus.append(generate_log_line(i));
    }

    size_t corpus_len = corpus.length();
    std::cout << "Corpus generated, size: " << corpus_len << " bytes ("
              << std::fixed << std::setprecision(2) << (corpus_len / 1e9) << " GB)" << std::endl;
    std::cout << std::endl;

    // Expected hit counts
    const int expected_404 = NUM_LINES / 10;  // 10%
    const int expected_200 = (NUM_LINES * 8) / 10;  // 80%
    const int expected_500 = NUM_LINES / 10;  // 10%
    std::cout << "Expected hits: 404=" << expected_404 << ", 200=" << expected_200 << ", 500=" << expected_500 << std::endl;
    std::cout << std::endl;

    // ==================== BUILD TIME BENCHMARK ====================
    std::cout << "=== BUILD TIME BENCHMARK (3 runs) ===" << std::endl;
    double best_build_time = std::numeric_limits<double>::max();
    for (int run = 0; run < 3; run++) {
        NSStringIndex temp_index;
        auto build_start = std::chrono::high_resolution_clock::now();
        temp_index.build(corpus.c_str(), corpus_len, ' ', NUM_FIELDS, {8});
        auto build_end = std::chrono::high_resolution_clock::now();
        double build_time_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
        if (build_time_ms < best_build_time) best_build_time = build_time_ms;
        std::cout << "Run " << (run + 1) << ": " << std::fixed << std::setprecision(2) << build_time_ms << " ms" << std::endl;
    }
    std::cout << "Best build time: " << std::fixed << std::setprecision(2) << best_build_time << " ms" << std::endl;
    std::cout << std::endl;

    // ==================== SCENARIO A: Single Query (Cold) ====================
    std::cout << "=== SCENARIO A: Single Query (Cold) ===" << std::endl;

    // Baseline: one strstr pass
    auto baseline_start = std::chrono::high_resolution_clock::now();
    uint64_t baseline_hits = baseline_scan(corpus.c_str(), corpus_len, "404");
    auto baseline_end = std::chrono::high_resolution_clock::now();
    double baseline_time_ms = std::chrono::duration<double, std::milli>(baseline_end - baseline_start).count();

    std::cout << "Baseline (strstr): " << std::fixed << std::setprecision(2) << baseline_time_ms << " ms, "
              << "hits: " << baseline_hits << std::endl;

    // NS: build index + one query
    NSStringIndex index;
    auto ns_build_start = std::chrono::high_resolution_clock::now();
    index.build(corpus.c_str(), corpus_len, ' ', NUM_FIELDS, {8});
    auto ns_build_end = std::chrono::high_resolution_clock::now();
    double ns_build_time_ms = std::chrono::duration<double, std::milli>(ns_build_end - ns_build_start).count();

    auto ns_query_start = std::chrono::high_resolution_clock::now();
    uint64_t ns_hits = index.query(8, "404");
    auto ns_query_end = std::chrono::high_resolution_clock::now();
    double ns_query_time_ms = std::chrono::duration<double, std::milli>(ns_query_end - ns_query_start).count();
    double ns_total_time_ms = ns_build_time_ms + ns_query_time_ms;

    std::cout << "NSIndex: build " << std::fixed << std::setprecision(2) << ns_build_time_ms << " ms, "
              << "query " << ns_query_time_ms << " ms, "
              << "total " << ns_total_time_ms << " ms, "
              << "hits: " << ns_hits << std::endl;

    std::cout << "Speedup: baseline is " << std::fixed << std::setprecision(1)
              << (ns_total_time_ms / baseline_time_ms) << "x faster" << std::endl;
    std::cout << std::endl;

    // ==================== SCENARIO B: 100 Queries (Amortized) ====================
    std::cout << "=== SCENARIO B: 100 Queries (Amortized) ===" << std::endl;

    // Baseline: 100 strstr passes
    std::vector<std::string> queries;
    for (int i = 0; i < 50; i++) queries.push_back("404");
    for (int i = 0; i < 25; i++) queries.push_back("200");
    for (int i = 0; i < 25; i++) queries.push_back("500");

    auto baseline_100_start = std::chrono::high_resolution_clock::now();
    uint64_t baseline_total_hits = 0;
    for (const auto& q : queries) {
        baseline_total_hits += baseline_scan(corpus.c_str(), corpus_len, q.c_str());
    }
    auto baseline_100_end = std::chrono::high_resolution_clock::now();
    double baseline_100_time_ms = std::chrono::duration<double, std::milli>(baseline_100_end - baseline_100_start).count();
    double baseline_per_query_ms = baseline_100_time_ms / 100;

    std::cout << "Baseline (strstr x100): " << std::fixed << std::setprecision(2) << baseline_100_time_ms << " ms, "
              << "per query: " << baseline_per_query_ms << " ms, "
              << "total hits: " << baseline_total_hits << std::endl;

    // NS: build once, 100 index lookups
    NSStringIndex index2;
    auto ns2_build_start = std::chrono::high_resolution_clock::now();
    index2.build(corpus.c_str(), corpus_len, ' ', NUM_FIELDS, {8});
    auto ns2_build_end = std::chrono::high_resolution_clock::now();
    double ns2_build_time_ms = std::chrono::duration<double, std::milli>(ns2_build_end - ns2_build_start).count();

    auto ns2_query_start = std::chrono::high_resolution_clock::now();
    uint64_t ns2_total_hits = 0;
    for (const auto& q : queries) {
        ns2_total_hits += index2.query(8, q);
    }
    auto ns2_query_end = std::chrono::high_resolution_clock::now();
    double ns2_query_time_ms = std::chrono::duration<double, std::milli>(ns2_query_end - ns2_query_start).count();
    double ns2_total_time_ms = ns2_build_time_ms + ns2_query_time_ms;
    double ns2_per_query_ms = ns2_query_time_ms / 100;

    std::cout << "NSIndex: build " << std::fixed << std::setprecision(2) << ns2_build_time_ms << " ms, "
              << "query x100 " << ns2_query_time_ms << " ms, "
              << "per query " << ns2_per_query_ms << " ms, "
              << "total " << ns2_total_time_ms << " ms, "
              << "total hits: " << ns2_total_hits << std::endl;

    std::cout << "Speedup: NSIndex is " << std::fixed << std::setprecision(1)
              << (baseline_100_time_ms / ns2_total_time_ms) << "x faster" << std::endl;
    std::cout << std::endl;

    // ==================== SCENARIO C: 1000 Queries (Amortized) ====================
    std::cout << "=== SCENARIO C: 1000 Queries (Amortized) ===" << std::endl;

    // Baseline: 1000 strstr passes
    std::vector<std::string> queries_1000;
    for (int i = 0; i < 500; i++) queries_1000.push_back("404");
    for (int i = 0; i < 250; i++) queries_1000.push_back("200");
    for (int i = 0; i < 250; i++) queries_1000.push_back("500");

    auto baseline_1000_start = std::chrono::high_resolution_clock::now();
    uint64_t baseline_1000_total_hits = 0;
    for (const auto& q : queries_1000) {
        baseline_1000_total_hits += baseline_scan(corpus.c_str(), corpus_len, q.c_str());
    }
    auto baseline_1000_end = std::chrono::high_resolution_clock::now();
    double baseline_1000_time_ms = std::chrono::duration<double, std::milli>(baseline_1000_end - baseline_1000_start).count();
    double baseline_1000_per_query_ms = baseline_1000_time_ms / 1000;

    std::cout << "Baseline (strstr x1000): " << std::fixed << std::setprecision(2) << baseline_1000_time_ms << " ms, "
              << "per query: " << baseline_1000_per_query_ms << " ms, "
              << "total hits: " << baseline_1000_total_hits << std::endl;

    // NS: build once, 1000 index lookups
    NSStringIndex index3;
    auto ns3_build_start = std::chrono::high_resolution_clock::now();
    index3.build(corpus.c_str(), corpus_len, ' ', NUM_FIELDS, {8});
    auto ns3_build_end = std::chrono::high_resolution_clock::now();
    double ns3_build_time_ms = std::chrono::duration<double, std::milli>(ns3_build_end - ns3_build_start).count();

    auto ns3_query_start = std::chrono::high_resolution_clock::now();
    uint64_t ns3_total_hits = 0;
    for (const auto& q : queries_1000) {
        ns3_total_hits += index3.query(8, q);
    }
    auto ns3_query_end = std::chrono::high_resolution_clock::now();
    double ns3_query_time_ms = std::chrono::duration<double, std::milli>(ns3_query_end - ns3_query_start).count();
    double ns3_total_time_ms = ns3_build_time_ms + ns3_query_time_ms;
    double ns3_per_query_ms = ns3_query_time_ms / 1000;

    std::cout << "NSIndex: build " << std::fixed << std::setprecision(2) << ns3_build_time_ms << " ms, "
              << "query x1000 " << ns3_query_time_ms << " ms, "
              << "per query " << ns3_per_query_ms << " ms, "
              << "total " << ns3_total_time_ms << " ms, "
              << "total hits: " << ns3_total_hits << std::endl;

    std::cout << "Speedup: NSIndex is " << std::fixed << std::setprecision(1)
              << (baseline_1000_time_ms / ns3_total_time_ms) << "x faster" << std::endl;
    std::cout << std::endl;

    // ==================== CORRECTNESS CHECK ====================
    std::cout << "=== CORRECTNESS CHECK ===" << std::endl;
    uint64_t ns_404 = index2.query(8, "404");
    uint64_t ns_200 = index2.query(8, "200");
    uint64_t ns_500 = index2.query(8, "500");

    std::cout << "Expected: 404=" << expected_404 << ", 200=" << expected_200 << ", 500=" << expected_500 << std::endl;
    std::cout << "NSIndex: 404=" << ns_404 << ", 200=" << ns_200 << ", 500=" << ns_500 << std::endl;

    bool correct = (ns_404 == (uint64_t)expected_404 &&
                    ns_200 == (uint64_t)expected_200 &&
                    ns_500 == (uint64_t)expected_500);
    std::cout << (correct ? "✓ CORRECT" : "✗ INCORRECT") << std::endl;
    std::cout << std::endl;

    // Save results to file
    const char* home_raw = std::getenv("HOME");
    std::string home = home_raw ? home_raw : "";
    std::ofstream results_file(home + "/NS/Testing/NSStringIndex/results.txt");
    if (results_file.is_open()) {
        results_file << "=== NSStringIndex Benchmark Results ===" << std::endl;
        results_file << "NUM_LINES: " << NUM_LINES << std::endl;
        results_file << "Corpus size: " << corpus_len << " bytes (" << std::fixed << std::setprecision(2) << (corpus_len / 1e9) << " GB)" << std::endl;
        results_file << std::endl;
        results_file << "=== SCENARIO A: Single Query (Cold) ===" << std::endl;
        results_file << "Baseline (strstr): " << std::fixed << std::setprecision(2) << baseline_time_ms << " ms, hits: " << baseline_hits << std::endl;
        results_file << "NSIndex: build " << ns_build_time_ms << " ms, query " << ns_query_time_ms << " ms, total " << ns_total_time_ms << " ms, hits: " << ns_hits << std::endl;
        results_file << "Speedup: baseline is " << std::setprecision(1) << (ns_total_time_ms / baseline_time_ms) << "x faster" << std::endl;
        results_file << std::endl;
        results_file << "=== SCENARIO B: 100 Queries (Amortized) ===" << std::endl;
        results_file << "Baseline (strstr x100): " << std::setprecision(2) << baseline_100_time_ms << " ms, per query: " << baseline_per_query_ms << " ms, total hits: " << baseline_total_hits << std::endl;
        results_file << "NSIndex: build " << ns2_build_time_ms << " ms, query x100 " << ns2_query_time_ms << " ms, per query " << ns2_per_query_ms << " ms, total " << ns2_total_time_ms << " ms, total hits: " << ns2_total_hits << std::endl;
        results_file << "Speedup: NSIndex is " << std::setprecision(1) << (baseline_100_time_ms / ns2_total_time_ms) << "x faster" << std::endl;
        results_file << std::endl;
        results_file << "=== SCENARIO C: 1000 Queries (Amortized) ===" << std::endl;
        results_file << "Baseline (strstr x1000): " << std::setprecision(2) << baseline_1000_time_ms << " ms, per query: " << baseline_1000_per_query_ms << " ms, total hits: " << baseline_1000_total_hits << std::endl;
        results_file << "NSIndex: build " << ns3_build_time_ms << " ms, query x1000 " << ns3_query_time_ms << " ms, per query " << ns3_per_query_ms << " ms, total " << ns3_total_time_ms << " ms, total hits: " << ns3_total_hits << std::endl;
        results_file << "Speedup: NSIndex is " << std::setprecision(1) << (baseline_1000_time_ms / ns3_total_time_ms) << "x faster" << std::endl;
        results_file << std::endl;
        results_file << "=== BUILD TIME ===" << std::endl;
        results_file << "Best build time: " << std::setprecision(2) << best_build_time << " ms" << std::endl;
        results_file << "=== CORRECTNESS ===" << std::endl;
        results_file << "Expected: 404=" << expected_404 << ", 200=" << expected_200 << ", 500=" << expected_500 << std::endl;
        results_file << "NSIndex: 404=" << ns_404 << ", 200=" << ns_200 << ", 500=" << ns_500 << std::endl;
        results_file << (correct ? "✓ CORRECT" : "✗ INCORRECT") << std::endl;
        results_file.close();
        std::cout << "Results saved to " << home + "/NS/Testing/NSStringIndex/results.txt" << std::endl;
    } else {
        std::cout << "Failed to open results file for writing" << std::endl;
    }

    return 0;
}
