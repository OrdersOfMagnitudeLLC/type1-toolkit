// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

// NSPack FUSE filesystem -- Windows (WinFSP)
// Requires WinFSP installed: https://github.com/winfsp/winfsp/releases
// Build: cl /O2 /std:c++17 /D_WIN32 nspack_fs_win.cpp NSComp\nscomp.cpp
//        /I"C:\Program Files (x86)\WinFsp\inc"
//        /link "C:\Program Files (x86)\WinFsp\lib\winfsp-x64.lib"
// See BUILD_WINDOWS.txt for full instructions.
//
// This file uses the WinFSP FUSE2 compatibility layer (inc/fuse, API 26).
// All logic is shared with the Linux/macOS driver through
// nspack_fs_core.hpp; only the FUSE glue and OS calls differ.
// Paths are normalized to forward slashes internally and converted to
// backslashes only at Win32 API boundaries.

#ifndef _WIN32
#error "nspack_fs_win.cpp is the Windows (WinFSP) build. Use nspack_fs.cpp on Linux/macOS."
#endif

#define FUSE_USE_VERSION 26

#include "nspack_fs_core.hpp"

#if defined(NSPACK_WIN32_REAL)
// Real Windows build: WinFSP FUSE2 headers. winfsp_fuse.h must come first;
// it supplies the fsp_fuse_env() the fuse_* wrappers call into.
#  include <fuse/winfsp_fuse.h>
#  include <fuse/fuse.h>
#else
// ---------------------------------------------------------------------------
// FUSE2 API declarations for `g++ -D_WIN32 -fsyntax-only` on POSIX hosts.
// These mirror winfsp inc/fuse/fuse.h (FUSE_USE_VERSION 26) closely enough
// to type-check this file without the WinFSP SDK installed.
// ---------------------------------------------------------------------------
extern "C" {
typedef int64_t fuse_off_t;
typedef unsigned int fuse_mode_t;
typedef unsigned int fuse_uid_t;
typedef unsigned int fuse_gid_t;
typedef int fuse_pid_t;
typedef uint64_t fuse_ino_t;
typedef uint64_t fuse_dev_t;

struct fuse_timespec { int64_t tv_sec; int64_t tv_nsec; };

struct fuse_stat {
    fuse_dev_t st_dev;
    fuse_ino_t st_ino;
    fuse_mode_t st_mode;
    uint32_t st_nlink;
    fuse_uid_t st_uid;
    fuse_gid_t st_gid;
    fuse_dev_t st_rdev;
    fuse_off_t st_size;
    struct fuse_timespec st_atim;
    struct fuse_timespec st_mtim;
    struct fuse_timespec st_ctim;
    uint64_t st_blksize;
    uint64_t st_blocks;
    struct fuse_timespec st_birthtim;
};

struct fuse_statvfs {
    uint64_t f_bsize, f_frsize, f_blocks, f_bfree, f_bavail;
    uint64_t f_files, f_ffree, f_favail, f_fsid, f_flag, f_namemax;
};

struct fuse_file_info {
    int flags;
    unsigned long fh_old;
    int writepage;
    unsigned int direct_io : 1;
    unsigned int keep_cache : 1;
    unsigned int flush : 1;
    unsigned int nonseekable : 1;
    unsigned int flock_release : 1;
    unsigned int padding : 27;
    uint64_t fh;
    uint64_t lock_owner;
};

struct fuse_conn_info {
    unsigned proto_major, proto_minor, async_read, max_write, max_readahead;
    unsigned capable, want, max_background, congestion_threshold;
};

struct fuse_context {
    void* fuse;
    fuse_uid_t uid;
    fuse_gid_t gid;
    fuse_pid_t pid;
    void* private_data;
    fuse_mode_t umask;
};

typedef int (*fuse_fill_dir_t)(void* buf, const char* name,
                               const struct fuse_stat* stbuf, fuse_off_t off);

struct fuse_operations {
    int (*getattr)(const char*, struct fuse_stat*);
    void* getdir;
    int (*readlink)(const char*, char*, size_t);
    int (*mknod)(const char*, fuse_mode_t, fuse_dev_t);
    int (*mkdir)(const char*, fuse_mode_t);
    int (*unlink)(const char*);
    int (*rmdir)(const char*);
    int (*symlink)(const char*, const char*);
    int (*rename)(const char*, const char*);
    void* link;
    int (*chmod)(const char*, fuse_mode_t);
    int (*chown)(const char*, fuse_uid_t, fuse_gid_t);
    int (*truncate)(const char*, fuse_off_t);
    void* utime;
    int (*open)(const char*, struct fuse_file_info*);
    int (*read)(const char*, char*, size_t, fuse_off_t, struct fuse_file_info*);
    int (*write)(const char*, const char*, size_t, fuse_off_t, struct fuse_file_info*);
    int (*statfs)(const char*, struct fuse_statvfs*);
    int (*flush)(const char*, struct fuse_file_info*);
    int (*release)(const char*, struct fuse_file_info*);
    int (*fsync)(const char*, int, struct fuse_file_info*);
    void* setxattr; void* getxattr; void* listxattr; void* removexattr;
    int (*opendir)(const char*, struct fuse_file_info*);
    int (*readdir)(const char*, void*, fuse_fill_dir_t, fuse_off_t,
                   struct fuse_file_info*);
    void* releasedir; void* fsyncdir;
    void* (*init)(struct fuse_conn_info*);
    void (*destroy)(void*);
    int (*access)(const char*, int);
    int (*create)(const char*, fuse_mode_t, struct fuse_file_info*);
    int (*ftruncate)(const char*, fuse_off_t, struct fuse_file_info*);
    int (*fgetattr)(const char*, struct fuse_stat*, struct fuse_file_info*);
    void* lock;
    int (*utimens)(const char*, const struct fuse_timespec[2]);
    void* bmap;
    unsigned int flag_nullpath_ok : 1;
    unsigned int flag_nopath : 1;
    unsigned int flag_utime_omit_ok : 1;
    unsigned int flag_reserved : 29;
    // Remaining members (ioctl/poll/write_buf/read_buf/flock/fallocate/
    // getpath/OSXFUSE extensions) are left zeroed; omitted here.
};

struct fuse_args { int argc; char** argv; int allocated; };
#define FUSE_ARGS_INIT(argc, argv) { argc, argv, 0 }

int fuse_opt_add_arg(struct fuse_args* args, const char* arg);
int fuse_main(int argc, char* argv[], const struct fuse_operations* ops, void* data);
struct fuse_context* fuse_get_context(void);
}
#endif // NSPACK_WIN32_REAL

#include <fcntl.h>
#include <cstring>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <system_error>

// Codec, mount records, and CLI helpers (nspack_encode_file,
// nspack_decode_file, nsp_file_size, add_mount_record, do_unmount,
// do_status, do_bench, print_usage). NSPACK_NO_MAIN keeps nspack.cpp's
// own main() out of this binary.
#ifndef NSPACK_NO_MAIN
#define NSPACK_NO_MAIN
#endif
#include "nspack.cpp"

// errno values missing from some MSVC setups.
#ifndef ENOENT
#define ENOENT 2
#endif
#ifndef EIO
#define EIO 5
#endif
#ifndef EBADF
#define EBADF 9
#endif
#ifndef EEXIST
#define EEXIST 17
#endif
#ifndef ENOTEMPTY
#define ENOTEMPTY 41
#endif
// POSIX flag values used by the WinFSP FUSE layer in fi->flags.
#ifndef O_CREAT
#define O_CREAT 0x40
#endif
#ifndef O_TRUNC
#define O_TRUNC 0x200
#endif

namespace fs = std::filesystem;

// ===== FUSE operations (FUSE2 signatures) =====

static void nsp_fill_times(nsp_stat_t* st, struct fuse_stat* stbuf) {
    stbuf->st_atim.tv_sec = (int64_t)st->st_atime; stbuf->st_atim.tv_nsec = 0;
    stbuf->st_mtim.tv_sec = (int64_t)st->st_mtime; stbuf->st_mtim.tv_nsec = 0;
    stbuf->st_ctim.tv_sec = (int64_t)st->st_ctime; stbuf->st_ctim.tv_nsec = 0;
}

static int nsp_getattr(const char *path, struct fuse_stat *stbuf) {
    if (is_hidden_path(path)) return -ENOENT;
    memset(stbuf, 0, sizeof(*stbuf));
    g_fs_mtx.lock();
    std::string spf = to_storage_file(path);
    nsp_stat_t st;
    if (nsp_stat(spf.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
        NspType t; uint64_t orig;
        if (nsp_file_size(spf, t, orig)) {
            stbuf->st_size = (fuse_off_t)orig;
        } else {
            stbuf->st_size = (fuse_off_t)st.st_size;
        }
        stbuf->st_mode = S_IFREG | 0644;
        stbuf->st_nlink = 1;
        struct fuse_context* ctx = fuse_get_context();
        stbuf->st_uid = ctx ? ctx->uid : 0;
        stbuf->st_gid = ctx ? ctx->gid : 0;
        nsp_fill_times(&st, stbuf);
        g_fs_mtx.unlock();
        return 0;
    }
    std::string spd = to_storage_path(path);
    if (nsp_stat(spd.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
        stbuf->st_mode = S_IFDIR | 0755;
        stbuf->st_nlink = 2;
        struct fuse_context* ctx = fuse_get_context();
        stbuf->st_uid = ctx ? ctx->uid : 0;
        stbuf->st_gid = ctx ? ctx->gid : 0;
        nsp_fill_times(&st, stbuf);
        g_fs_mtx.unlock();
        return 0;
    }
    g_fs_mtx.unlock();
    return -ENOENT;
}

static int nsp_fgetattr(const char *path, struct fuse_stat *stbuf,
                        struct fuse_file_info *fi) {
    (void)fi;
    return nsp_getattr(path, stbuf);
}

static int nsp_opendir(const char *path, struct fuse_file_info *fi) {
    (void)fi;
    if (is_hidden_path(path)) return -ENOENT;
    std::string spd = to_storage_path(path);
    std::error_code ec;
    if (!fs::is_directory(fs::path(spd), ec)) return -ENOENT;
    return 0;
}

static int nsp_readdir(const char *path, void *buf, fuse_fill_dir_t filler,
                       fuse_off_t offset, struct fuse_file_info *fi) {
    (void)offset; (void)fi;
    g_fs_mtx.lock();
    std::string spd = to_storage_path(path);
    std::error_code ec;
    if (!fs::is_directory(fs::path(spd), ec)) {
        g_fs_mtx.unlock();
        return -ENOENT;
    }
    filler(buf, ".", NULL, 0);
    filler(buf, "..", NULL, 0);
    for (auto& e : fs::directory_iterator(fs::path(spd), ec)) {
        std::string name = e.path().filename().string();
        if (name == "." || name == ".." || name == ".nspack_tmp") continue;
        if (name.size() > 4 && name.substr(name.size() - 4) == ".nsx") {
            name = name.substr(0, name.size() - 4);
        }
        if (name.size() >= 8 && name.substr(name.size() - 8) == ".hdrinfo") continue;
        if (filler(buf, name.c_str(), NULL, 0) != 0) break;
    }
    g_fs_mtx.unlock();
    return 0;
}

static int nsp_open(const char *path, struct fuse_file_info *fi) {
    if (is_hidden_path(path)) return -ENOENT;
    g_fs_mtx.lock();
    std::string spf = to_storage_file(path);
    bool exists = (nsp_sys_access(spf.c_str(), F_OK) == 0);
    int acc = fi->flags & O_ACCMODE;
    bool creating = (fi->flags & O_CREAT) && !exists;
    bool trunc = (fi->flags & O_TRUNC) != 0;

    nsp_file* nf = new nsp_file();
    nf->path = path;
    nf->write_mode = (acc != O_RDONLY);

    if (nf->write_mode && (creating || trunc)) {
        if (!nsp_create_placeholder(spf)) {
            delete nf;
            g_fs_mtx.unlock();
            return -EIO;
        }
        nf->dirty = true;
    } else if (nf->write_mode && exists) {
        // Open existing for write without truncation: load current content.
        std::string out = temp_decoded_path(path);
        if (!nspack_decode_file(spf, out, nullptr)) {
            delete nf;
            g_fs_mtx.unlock();
            return -EIO;
        }
        std::ifstream f(out, std::ios::binary | std::ios::ate);
        if (f) {
            size_t sz = (size_t)f.tellg();
            f.seekg(0, std::ios::beg);
            nf->buf.resize(sz);
            f.read((char*)nf->buf.data(), sz);
            f.close();
        }
        std::remove(out.c_str());
        nf->dirty = true;
    } else if (!nf->write_mode && exists) {
        std::string out = temp_decoded_path(path);
        if (!nspack_decode_file(spf, out, nullptr)) {
            delete nf;
            g_fs_mtx.unlock();
            return -ENOENT;
        }
        std::ifstream f(out, std::ios::binary | std::ios::ate);
        if (!f) {
            delete nf;
            g_fs_mtx.unlock();
            return -EIO;
        }
        size_t sz = (size_t)f.tellg();
        f.seekg(0, std::ios::beg);
        nf->buf.resize(sz);
        f.read((char*)nf->buf.data(), sz);
        f.close();
        std::remove(out.c_str());
    } else {
        delete nf;
        g_fs_mtx.unlock();
        return -ENOENT;
    }

    fi->fh = (uint64_t)(uintptr_t)nf;
    g_fs_mtx.unlock();
    return 0;
}

static int nsp_create(const char *path, fuse_mode_t mode, struct fuse_file_info *fi) {
    (void)mode;
    if (is_hidden_path(path)) return -ENOENT;
    g_fs_mtx.lock();
    std::string spf = to_storage_file(path);
    if (!nsp_create_placeholder(spf)) {
        g_fs_mtx.unlock();
        return -EIO;
    }
    nsp_file* nf = new nsp_file();
    nf->path = path;
    nf->write_mode = true;
    nf->dirty = true;
    fi->fh = (uint64_t)(uintptr_t)nf;
    g_fs_mtx.unlock();
    return 0;
}

static int nsp_read(const char *path, char *buf, size_t size, fuse_off_t offset,
                    struct fuse_file_info *fi) {
    (void)path;
    nsp_file* nf = (nsp_file*)(uintptr_t)fi->fh;
    if (!nf) return -EBADF;
    if (offset >= (fuse_off_t)nf->buf.size()) return 0;
    size_t n = std::min(size, nf->buf.size() - (size_t)offset);
    memcpy(buf, nf->buf.data() + offset, n);
    return (int)n;
}

static int nsp_write(const char *path, const char *buf, size_t size, fuse_off_t offset,
                     struct fuse_file_info *fi) {
    (void)path;
    nsp_file* nf = (nsp_file*)(uintptr_t)fi->fh;
    if (!nf) return -EBADF;
    g_fs_mtx.lock();
    if ((size_t)offset + size > nf->buf.size()) nf->buf.resize((size_t)offset + size, 0);
    memcpy(nf->buf.data() + offset, buf, size);
    nf->dirty = true;
    g_fs_mtx.unlock();
    return (int)size;
}

static int nsp_truncate_impl(const char *path, fuse_off_t size,
                             struct fuse_file_info *fi) {
    if (is_hidden_path(path)) return -ENOENT;
    g_fs_mtx.lock();
    if (fi && fi->fh) {
        nsp_file* nf = (nsp_file*)(uintptr_t)fi->fh;
        nf->buf.resize((size_t)size, 0);
        nf->dirty = true;
    } else {
        std::string spf = to_storage_file(path);
        std::string out = temp_decoded_path(path);
        if (nspack_decode_file(spf, out, nullptr)) {
            std::vector<uint8_t> data;
            std::ifstream f(out, std::ios::binary | std::ios::ate);
            if (f) {
                size_t sz = (size_t)f.tellg();
                f.seekg(0, std::ios::beg);
                data.resize(sz);
                f.read((char*)data.data(), sz);
                f.close();
            }
            std::remove(out.c_str());
            data.resize((size_t)size, 0);
            std::string in = temp_input_path(path);
            std::ofstream of(in, std::ios::binary);
            of.write((char*)data.data(), data.size());
            of.close();
            nspack_encode_file(in, spf, nullptr);
            std::remove(in.c_str());
        }
    }
    g_fs_mtx.unlock();
    return 0;
}

// FUSE2 splits truncate by path vs. by file handle.
static int nsp_truncate(const char *path, fuse_off_t size) {
    return nsp_truncate_impl(path, size, nullptr);
}

static int nsp_ftruncate(const char *path, fuse_off_t size,
                         struct fuse_file_info *fi) {
    return nsp_truncate_impl(path, size, fi);
}

static int nsp_release(const char *path, struct fuse_file_info *fi) {
    nsp_file* nf = (nsp_file*)(uintptr_t)fi->fh;
    if (!nf) return 0;
    if (nf->dirty) {
        g_fs_mtx.lock();
        std::string in = temp_input_path(path);
        if (!in.empty()) {
            std::ofstream of(in, std::ios::binary);
            of.write((char*)nf->buf.data(), nf->buf.size());
            of.close();
            std::string spf = to_storage_file(path);
            std::string dir = parent_dir(spf);
            ensure_dir(dir);
            // HDR requires a .hdrinfo sidecar. It lives only in storage and is
            // never visible in the mount; copy it next to the temporary input.
            if (std::string(path).size() > 4) {
                std::string p = path;
                std::string ext = p.substr(p.size() - 4);
                if (ext == ".hdr" || ext == ".exr") {
                    std::string sidecar_raw = to_storage_path(path) + ".hdrinfo";
                    std::ifstream sc(sidecar_raw.c_str());
                    if (sc) {
                        std::string sidecar_tmp = in + ".hdrinfo";
                        std::ofstream sct(sidecar_tmp, std::ios::binary);
                        sct << sc.rdbuf();
                        sct.close();
                    }
                }
            }

            double enc_ms = 0;
            bool ok = nspack_encode_file(in, spf, &enc_ms);
            if (!ok) {
                // Schema-aware encoder failed; store raw bytes as PRECOMPRESSED.
                std::string raw = in + ".raw";
                std::ifstream src(in, std::ios::binary);
                std::ofstream dst(raw, std::ios::binary);
                dst << src.rdbuf();
                src.close(); dst.close();
                nspack_encode_file(raw, spf, &enc_ms);
                std::remove(raw.c_str());
            }
            std::remove(in.c_str());
            std::string extra = in + ".nspktmp";
            std::remove(extra.c_str());
        }
        g_fs_mtx.unlock();
    }
    delete nf;
    return 0;
}

static int nsp_unlink(const char *path) {
    if (is_hidden_path(path)) return -ENOENT;
    g_fs_mtx.lock();
    std::string spf = to_storage_file(path);
    std::remove(spf.c_str());
    std::remove((spf + ".nspktmp").c_str());
    g_fs_mtx.unlock();
    return 0;
}

static int nsp_rename(const char *from, const char *to) {
    if (is_hidden_path(from) || is_hidden_path(to)) return -ENOENT;
    g_fs_mtx.lock();
    std::string sf = to_storage_file(from);
    std::string st = to_storage_file(to);
    std::string dt = parent_dir(st);
    ensure_dir(dt);
    // MoveFileExA with REPLACE_EXISTING matches POSIX rename semantics.
    int r = nsp_rename_file(sf.c_str(), st.c_str());
    g_fs_mtx.unlock();
    return (r == 0) ? 0 : -EIO;
}

static int nsp_mkdir(const char *path, fuse_mode_t mode) {
    (void)mode;
    std::string spd = to_storage_path(path);
    if (nsp_sys_mkdir(spd.c_str()) != 0 && errno != EEXIST) return -errno;
    return 0;
}

static int nsp_rmdir(const char *path) {
    std::string spd = to_storage_path(path);
    if (nsp_sys_rmdir(spd.c_str()) != 0) return -errno;
    return 0;
}

static int nsp_access(const char *path, int mask) {
    (void)mask;
    if (is_hidden_path(path)) return -ENOENT;
    return 0;
}

static int nsp_chmod(const char *path, fuse_mode_t mode) {
    (void)path; (void)mode;
    return 0;
}

static int nsp_chown(const char *path, fuse_uid_t uid, fuse_gid_t gid) {
    (void)path; (void)uid; (void)gid;
    return 0;
}

static int nsp_utimens(const char *path, const struct fuse_timespec tv[2]) {
    (void)path; (void)tv;
    return 0;
}

static int nsp_flush(const char *path, struct fuse_file_info *fi) {
    (void)path; (void)fi;
    return 0;
}

static int nsp_fsync(const char *path, int isdatasync, struct fuse_file_info *fi) {
    (void)path; (void)isdatasync; (void)fi;
    return 0;
}

static int nsp_statfs(const char *path, struct fuse_statvfs *stbuf) {
    (void)path;
    memset(stbuf, 0, sizeof(*stbuf));
    stbuf->f_bsize = 4096;
    stbuf->f_frsize = 4096;
    stbuf->f_namemax = 255;
#if defined(NSPACK_WIN32_REAL)
    // Report the storage volume's real free space when available.
    ULARGE_INTEGER free_avail, total, total_free;
    std::string root = g_storagedir;
    for (auto& c : root) if (c == '/') c = '\\';
    if (GetDiskFreeSpaceExA(root.c_str(), &free_avail, &total, &total_free)) {
        stbuf->f_blocks = (uint64_t)(total.QuadPart / 4096);
        stbuf->f_bfree = (uint64_t)(total_free.QuadPart / 4096);
        stbuf->f_bavail = (uint64_t)(free_avail.QuadPart / 4096);
    }
#endif
    return 0;
}

static void* nsp_init(struct fuse_conn_info *conn) {
    (void)conn;
    return NULL;
}

static void nsp_destroy(void* private_data) {
    (void)private_data;
}

static struct fuse_operations nsp_ops;

static void init_ops() {
    memset(&nsp_ops, 0, sizeof(nsp_ops));
    nsp_ops.getattr = nsp_getattr;
    nsp_ops.fgetattr = nsp_fgetattr;
    nsp_ops.opendir = nsp_opendir;
    nsp_ops.readdir = nsp_readdir;
    nsp_ops.open = nsp_open;
    nsp_ops.create = nsp_create;
    nsp_ops.read = nsp_read;
    nsp_ops.write = nsp_write;
    nsp_ops.release = nsp_release;
    nsp_ops.unlink = nsp_unlink;
    nsp_ops.rename = nsp_rename;
    nsp_ops.mkdir = nsp_mkdir;
    nsp_ops.rmdir = nsp_rmdir;
    nsp_ops.truncate = nsp_truncate;
    nsp_ops.ftruncate = nsp_ftruncate;
    nsp_ops.access = nsp_access;
    nsp_ops.chmod = nsp_chmod;
    nsp_ops.chown = nsp_chown;
    nsp_ops.utimens = nsp_utimens;
    nsp_ops.flush = nsp_flush;
    nsp_ops.fsync = nsp_fsync;
    nsp_ops.statfs = nsp_statfs;
    nsp_ops.init = nsp_init;
    nsp_ops.destroy = nsp_destroy;
}

// Mount point on Windows is a drive letter ("Z:") or a directory path.
static int run_fuse(const std::string& mountpoint, const std::string& storagedir) {
    g_storagedir = storagedir;
    for (auto& c : g_storagedir) if (c == '\\') c = '/';
    if (nsp_sys_mkdir(storagedir.c_str()) != 0 && errno != EEXIST) {
        std::cerr << "Cannot create storage dir: " << storagedir << "\n";
        return 1;
    }
    add_mount_record(mountpoint, storagedir, (unsigned long)GetCurrentProcessId());
    init_ops();
    struct fuse_args args = FUSE_ARGS_INIT(0, NULL);
    fuse_opt_add_arg(&args, "nspack_fs");
    fuse_opt_add_arg(&args, mountpoint.c_str());
    int r = fuse_main(args.argc, args.argv, &nsp_ops, NULL);
    return r;
}

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }
    std::string cmd = argv[1];
    if (cmd == "--help" || cmd == "-h") { print_usage(argv[0]); return 0; }

    if (cmd == "mount") {
        if (argc < 4) { print_usage(argv[0]); return 1; }
        return run_fuse(argv[2], argv[3]);
    }

    if (cmd == "unmount") {
        if (argc < 3) { print_usage(argv[0]); return 1; }
        return do_unmount(argv[2]);
    }

    if (cmd == "status") {
        if (argc < 3) { print_usage(argv[0]); return 1; }
        return do_status(argv[2]);
    }

    if (cmd == "bench") {
        if (argc < 3) { print_usage(argv[0]); return 1; }
        std::vector<std::string> files;
        for (int i = 2; i < argc; i++) files.push_back(argv[i]);
        return do_bench(files);
    }

    if (argc == 3) {
        return run_fuse(argv[1], argv[2]);
    }

    std::cerr << "Invalid command: " << cmd << "\n";
    print_usage(argv[0]);
    return 1;
}
