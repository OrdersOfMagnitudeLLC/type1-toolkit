// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

// nspack_fs_core.hpp - shared filesystem helpers for the NSPack FUSE drivers.
// Included by nspack_fs.cpp (Linux/macOS, FUSE3) and nspack_fs_win.cpp
// (Windows, WinFSP FUSE2). No FUSE headers here; this file is pure
// platform-portable logic. All paths are normalized to forward slashes
// internally; conversion happens only at OS API boundaries.

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <fstream>
#include <atomic>
#include <sys/stat.h>
#include <sys/types.h>

// Platform compat (Win32 vs POSIX): nsp_stat_t, nsp_sys_mkdir, nsp_getpid,
// nsp_temp_dir, nsp_fs_mutex, and the NSPACK_WIN32_REAL detection live here.
#include "nspack_platform.hpp"

// ===== Shared state =====

static std::string g_storagedir;
static nsp_fs_mutex g_fs_mtx;

struct nsp_file {
    std::string path;
    std::vector<uint8_t> buf;
    bool dirty = false;
    bool write_mode = false;
};

// ===== Path helpers (forward slashes internally on every platform) =====

static std::string to_storage_path(const char* path) {
    std::string p = path ? path : "";
    for (auto& c : p) if (c == '\\') c = '/';
    if (p.empty() || p[0] != '/') return g_storagedir + "/" + p;
    return g_storagedir + p;
}

static std::string to_storage_file(const char* path) {
    return to_storage_path(path) + ".nsx";
}

static bool is_hidden_path(const char* path) {
    std::string p = path ? path : "";
    if (p.size() >= 8 && p.substr(p.size() - 8) == ".hdrinfo") return true;
    size_t s = p.find_last_of("/\\");
    if (s != std::string::npos && s + 1 < p.size()) {
        std::string name = p.substr(s + 1);
        if (name.size() >= 8 && name.substr(name.size() - 8) == ".hdrinfo") return true;
    }
    return false;
}

static std::string parent_dir(const std::string& p) {
    size_t s = p.find_last_of("/\\");
    if (s == std::string::npos || s == 0) return "/";
    return p.substr(0, s);
}

static void ensure_dir(const std::string& dir) {
    if (dir.empty() || dir == "/") return;
    size_t pos = 0;
    while (pos < dir.size()) {
        pos = dir.find('/', pos + 1);
        std::string sub = (pos == std::string::npos) ? dir : dir.substr(0, pos);
        if (sub.empty()) continue;
        nsp_sys_mkdir(sub.c_str());
    }
}

// ===== .nsx placeholder (empty NSP1 container) =====

// NspType::PRECOMPRESSED == 12; written as a raw byte so this header does
// not depend on the enum definition inside nspack.cpp.
static bool nsp_create_placeholder(const std::string& spf) {
    std::string dir = parent_dir(spf);
    ensure_dir(dir);
    std::ofstream of(spf, std::ios::binary);
    if (!of) return false;
    const char magic[4] = {'N','S','P','1'};
    of.write(magic, 4);
    uint8_t t = 12; // NspType::PRECOMPRESSED
    of.write((char*)&t, 1);
    uint64_t orig = 0;
    of.write((char*)&orig, 8);
    uint32_t mlen = 0;
    of.write((char*)&mlen, 4);
    of.close();
    return of.good();
}

// ===== Temp file paths =====

static std::string temp_decoded_path(const std::string& base) {
    static std::atomic<int> counter{0};
    int c = counter.fetch_add(1);
    size_t s = base.find_last_of("/\\");
    std::string name = (s == std::string::npos) ? base : base.substr(s + 1);
    return nsp_temp_dir() + "/.nspack_dec_" + std::to_string(nsp_getpid())
         + "_" + std::to_string(c) + "_" + name;
}

static std::string temp_input_path(const char* path) {
    static std::atomic<int> counter{0};
    int c = counter.fetch_add(1);
    std::string p = path ? path : "";
    size_t dot = p.rfind('.');
    std::string ext = (dot == std::string::npos) ? ".tmp" : p.substr(dot);
    std::string t = nsp_temp_dir() + "/.nspack_in_" + std::to_string(nsp_getpid())
                  + "_" + std::to_string(c) + ext;
    // Reserve the name so concurrent opens cannot collide.
    std::ofstream of(t, std::ios::binary | std::ios::trunc);
    return of ? t : std::string();
}
