// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

// nspack_fs.cpp - FUSE3 filesystem for NSPack transparent compression

#define FUSE_USE_VERSION 31
#include <fuse3/fuse.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
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

#include "nspack.cpp"
#include "nspack_fs_core.hpp"

static int nsp_getattr(const char *path, struct stat *stbuf, struct fuse_file_info *fi) {
    (void)fi;
    if (is_hidden_path(path)) return -ENOENT;
    memset(stbuf, 0, sizeof(*stbuf));
    g_fs_mtx.lock();
    std::string spf = to_storage_file(path);
    struct stat st;
    if (stat(spf.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
        NspType t; uint64_t orig;
        if (nsp_file_size(spf, t, orig)) {
            stbuf->st_size = (off_t)orig;
        } else {
            stbuf->st_size = st.st_size;
        }
        stbuf->st_mode = S_IFREG | 0644;
        stbuf->st_nlink = 1;
        stbuf->st_uid = getuid();
        stbuf->st_gid = getgid();
        stbuf->st_mtime = st.st_mtime;
        stbuf->st_atime = st.st_atime;
        stbuf->st_ctime = st.st_ctime;
        g_fs_mtx.unlock();
        return 0;
    }
    std::string spd = to_storage_path(path);
    if (stat(spd.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
        stbuf->st_mode = S_IFDIR | 0755;
        stbuf->st_nlink = 2;
        stbuf->st_uid = getuid();
        stbuf->st_gid = getgid();
        stbuf->st_mtime = st.st_mtime;
        stbuf->st_atime = st.st_atime;
        stbuf->st_ctime = st.st_ctime;
        g_fs_mtx.unlock();
        return 0;
    }
    g_fs_mtx.unlock();
    return -ENOENT;
}

static int nsp_readdir(const char *path, void *buf, fuse_fill_dir_t filler,
                       off_t offset, struct fuse_file_info *fi,
                       enum fuse_readdir_flags flags) {
    (void)offset; (void)fi; (void)flags;
    g_fs_mtx.lock();
    std::string spd = to_storage_path(path);
    DIR* d = opendir(spd.c_str());
    if (!d) { g_fs_mtx.unlock(); return -ENOENT; }
    enum fuse_fill_dir_flags zero = (enum fuse_fill_dir_flags)0;
    filler(buf, ".", NULL, 0, zero);
    filler(buf, "..", NULL, 0, zero);
    struct dirent* de;
    while ((de = readdir(d)) != nullptr) {
        std::string name = de->d_name;
        if (name == "." || name == ".." || name == ".nspack_tmp") continue;
        if (name.size() > 4 && name.substr(name.size() - 4) == ".nsx") {
            name = name.substr(0, name.size() - 4);
        }
        if (name.size() >= 8 && name.substr(name.size() - 8) == ".hdrinfo") continue;
        if (filler(buf, name.c_str(), NULL, 0, zero) != 0) break;
    }
    closedir(d);
    g_fs_mtx.unlock();
    return 0;
}

static int nsp_open(const char *path, struct fuse_file_info *fi) {
    if (is_hidden_path(path)) return -ENOENT;
    g_fs_mtx.lock();
    std::string spf = to_storage_file(path);
    bool exists = (access(spf.c_str(), F_OK) == 0);
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

static int nsp_create(const char *path, mode_t mode, struct fuse_file_info *fi) {
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

static int nsp_read(const char *path, char *buf, size_t size, off_t offset,
                    struct fuse_file_info *fi) {
    (void)path;
    nsp_file* nf = (nsp_file*)(uintptr_t)fi->fh;
    if (!nf) return -EBADF;
    if (offset >= (off_t)nf->buf.size()) return 0;
    size_t n = std::min(size, nf->buf.size() - (size_t)offset);
    memcpy(buf, nf->buf.data() + offset, n);
    return (int)n;
}

static int nsp_write(const char *path, const char *buf, size_t size, off_t offset,
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

static int nsp_truncate(const char *path, off_t size, struct fuse_file_info *fi) {
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

static int nsp_rename(const char *from, const char *to, unsigned int flags) {
    (void)flags;
    if (is_hidden_path(from) || is_hidden_path(to)) return -ENOENT;
    g_fs_mtx.lock();
    std::string sf = to_storage_file(from);
    std::string st = to_storage_file(to);
    std::string dt = parent_dir(st);
    ensure_dir(dt);
    int r = rename(sf.c_str(), st.c_str());
    g_fs_mtx.unlock();
    return (r == 0) ? 0 : -errno;
}

static int nsp_mkdir(const char *path, mode_t mode) {
    std::string spd = to_storage_path(path);
    if (mkdir(spd.c_str(), mode) != 0 && errno != EEXIST) return -errno;
    return 0;
}

static int nsp_rmdir(const char *path) {
    std::string spd = to_storage_path(path);
    if (rmdir(spd.c_str()) != 0) return -errno;
    return 0;
}

static int nsp_access(const char *path, int mask) {
    (void)mask;
    if (is_hidden_path(path)) return -ENOENT;
    return 0;
}

static int nsp_chmod(const char *path, mode_t mode, struct fuse_file_info *fi) {
    (void)path; (void)mode; (void)fi;
    return 0;
}

static int nsp_chown(const char *path, uid_t uid, gid_t gid, struct fuse_file_info *fi) {
    (void)path; (void)uid; (void)gid; (void)fi;
    return 0;
}

static int nsp_utimens(const char *path, const struct timespec tv[2],
                       struct fuse_file_info *fi) {
    (void)path; (void)tv; (void)fi;
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

static void* nsp_init(struct fuse_conn_info *conn, struct fuse_config *cfg) {
    (void)conn; (void)cfg;
    return NULL;
}

static void nsp_destroy(void* private_data) {
    (void)private_data;
}

static struct fuse_operations nsp_ops;

static void init_ops() {
    memset(&nsp_ops, 0, sizeof(nsp_ops));
    nsp_ops.getattr = nsp_getattr;
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
    nsp_ops.access = nsp_access;
    nsp_ops.chmod = nsp_chmod;
    nsp_ops.chown = nsp_chown;
    nsp_ops.utimens = nsp_utimens;
    nsp_ops.flush = nsp_flush;
    nsp_ops.fsync = nsp_fsync;
    nsp_ops.init = nsp_init;
    nsp_ops.destroy = nsp_destroy;
}

static int run_fuse(const std::string& mountpoint, const std::string& storagedir) {
    g_storagedir = storagedir;
    if (mkdir(storagedir.c_str(), 0777) != 0 && errno != EEXIST) {
        std::cerr << "Cannot create storage dir: " << storagedir << "\n";
        return 1;
    }
    add_mount_record(mountpoint, storagedir);
    init_ops();
    struct fuse_args args = FUSE_ARGS_INIT(0, NULL);
    fuse_opt_add_arg(&args, "nspack_fs");
    fuse_opt_add_arg(&args, mountpoint.c_str());
    fuse_opt_add_arg(&args, "-o");
    fuse_opt_add_arg(&args, "auto_unmount");
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
