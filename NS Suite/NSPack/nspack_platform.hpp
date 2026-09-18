// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

// nspack_platform.hpp - platform compatibility layer for NSPack.
// Included by nspack.cpp (CLI) and nspack_fs_core.hpp (FUSE drivers).
// Provides a real Win32 path when building with MSVC/MinGW, and minimal
// declarations when a POSIX host runs `-D_WIN32 -fsyntax-only` checks.

#pragma once

#include <string>
#include <cstdint>
#include <cstdio>
#include <cerrno>
#include <sys/stat.h>
#include <sys/types.h>

// Detect a real Windows toolchain vs. a POSIX host doing a -D_WIN32
// syntax check (no <windows.h> available there).
#if defined(_WIN32)
#  if defined(__has_include)
#    if __has_include(<windows.h>)
#      define NSPACK_WIN32_REAL 1
#    endif
#  endif
#  if !defined(NSPACK_WIN32_REAL) && (defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__))
#    define NSPACK_WIN32_REAL 1
#  endif
#endif

#if defined(_WIN32)

#  if defined(NSPACK_WIN32_REAL)
#    include <windows.h>
#    include <direct.h>
#    include <io.h>
#    include <process.h>
#    include <fcntl.h>
#  else
// Minimal declarations so `g++ -D_WIN32 -fsyntax-only` works on POSIX hosts.
// Signatures match the real Win32/MSVC decls closely enough for checking.
extern "C" {
typedef void* HANDLE;
typedef unsigned long DWORD;
typedef unsigned int UINT;
typedef int BOOL;
typedef void* LPVOID;
typedef char* LPSTR;
typedef const char* LPCSTR;
typedef unsigned long* LPDWORD;
typedef struct _NSP_CRITICAL_SECTION { void* opaque[6]; } CRITICAL_SECTION;
typedef CRITICAL_SECTION* LPCRITICAL_SECTION;
void InitializeCriticalSection(LPCRITICAL_SECTION);
void DeleteCriticalSection(LPCRITICAL_SECTION);
void EnterCriticalSection(LPCRITICAL_SECTION);
void LeaveCriticalSection(LPCRITICAL_SECTION);
DWORD GetTempPathA(DWORD nBufferLength, LPSTR lpBuffer);
DWORD GetTempFileNameA(LPCSTR lpPathName, LPCSTR lpPrefixString,
                       UINT uUnique, LPSTR lpTempFileName);
DWORD GetCurrentProcessId(void);
DWORD GetModuleFileNameA(HANDLE hModule, LPSTR lpFilename, DWORD nSize);
void Sleep(DWORD dwMilliseconds);
BOOL CloseHandle(HANDLE hObject);
HANDLE OpenProcess(DWORD dwDesiredAccess, BOOL bInheritHandle, DWORD dwProcessId);
BOOL TerminateProcess(HANDLE hProcess, UINT uExitCode);
BOOL MoveFileExA(LPCSTR lpExistingFileName, LPCSTR lpNewFileName, DWORD dwFlags);
typedef struct _NSP_STARTUPINFOA {
    DWORD cb; LPSTR lpReserved; LPSTR lpDesktop; LPSTR lpTitle;
    DWORD dwX, dwY, dwXSize, dwYSize, dwXCountChars, dwYCountChars;
    DWORD dwFillAttribute, dwFlags; unsigned short wShowWindow, cbReserved2;
    void* lpReserved2; HANDLE hStdInput, hStdOutput, hStdError;
} STARTUPINFOA, *LPSTARTUPINFOA;
typedef struct _NSP_PROCESS_INFORMATION {
    HANDLE hProcess; HANDLE hThread; DWORD dwProcessId; DWORD dwThreadId;
} PROCESS_INFORMATION, *LPPROCESS_INFORMATION;
BOOL CreateProcessA(LPCSTR lpApplicationName, LPSTR lpCommandLine,
                    LPVOID lpProcessAttributes, LPVOID lpThreadAttributes,
                    BOOL bInheritHandles, DWORD dwCreationFlags,
                    LPVOID lpEnvironment, LPCSTR lpCurrentDirectory,
                    LPSTARTUPINFOA lpStartupInfo,
                    LPPROCESS_INFORMATION lpProcessInformation);
typedef struct _NSP_MEMORYSTATUSEX {
    DWORD dwLength; DWORD dwMemoryLoad;
    unsigned long long ullTotalPhys, ullAvailPhys, ullTotalPageFile,
        ullAvailPageFile, ullTotalVirtual, ullAvailVirtual,
        ullAvailExtendedVirtual;
} MEMORYSTATUSEX, *LPMEMORYSTATUSEX;
BOOL GlobalMemoryStatusEx(LPMEMORYSTATUSEX lpBuffer);
int _mkdir(const char*);
int _rmdir(const char*);
int _access(const char*, int);
int _getpid(void);
FILE* _popen(const char*, const char*);
int _pclose(FILE*);
}
// glibc <sys/stat.h> defines st_atime/st_mtime/st_ctime as macros expanding
// to st_atim.tv_sec etc.; drop them so the _stat64 member names survive.
#    ifdef st_atime
#      undef st_atime
#    endif
#    ifdef st_mtime
#      undef st_mtime
#    endif
#    ifdef st_ctime
#      undef st_ctime
#    endif
extern "C" {
struct _stat64 {
    unsigned int st_dev; unsigned short st_ino; unsigned short st_mode;
    short st_nlink; short st_uid; short st_gid; unsigned int st_rdev;
    long long st_size; long long st_atime; long long st_mtime; long long st_ctime;
};
int _stat64(const char*, struct _stat64*);
}
#    ifndef MAX_PATH
#      define MAX_PATH 260
#    endif
#    ifndef FALSE
#      define FALSE 0
#      define TRUE 1
#    endif
#    ifndef DETACHED_PROCESS
#      define DETACHED_PROCESS 0x00000008
#    endif
#    ifndef CREATE_NEW_PROCESS_GROUP
#      define CREATE_NEW_PROCESS_GROUP 0x00000200
#    endif
#    ifndef PROCESS_TERMINATE
#      define PROCESS_TERMINATE 0x0001
#    endif
#    ifndef MOVEFILE_REPLACE_EXISTING
#      define MOVEFILE_REPLACE_EXISTING 0x1
#    endif
#    ifndef STARTF_USESHOWWINDOW
#      define STARTF_USESHOWWINDOW 0x00000001
#    endif
#  endif // NSPACK_WIN32_REAL

// MSVC <sys/stat.h> lacks the POSIX S_IS*/S_IF* spellings.
#  ifndef S_IFREG
#    define S_IFREG _S_IFREG
#  endif
#  ifndef S_IFDIR
#    define S_IFDIR _S_IFDIR
#  endif
#  ifndef S_ISREG
#    define S_ISREG(m) (((m) & _S_IFMT) == _S_IFREG)
#  endif
#  ifndef S_ISDIR
#    define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#  endif
#  ifndef F_OK
#    define F_OK 0
#  endif
// MSVC <fcntl.h> lacks O_ACCMODE; O_RDONLY/O_WRONLY/O_RDWR are 0/1/2 there too.
#  ifndef O_ACCMODE
#    define O_ACCMODE 3
#  endif

typedef struct _stat64 nsp_stat_t;
static inline int nsp_stat(const char* p, nsp_stat_t* s) { return _stat64(p, s); }
static inline int nsp_sys_mkdir(const char* p) { return _mkdir(p); }
static inline int nsp_sys_rmdir(const char* p) { return _rmdir(p); }
static inline int nsp_sys_access(const char* p, int m) { return _access(p, m); }
static inline int nsp_getpid(void) { return (int)GetCurrentProcessId(); }
static inline FILE* nsp_popen(const char* c, const char* m) { return _popen(c, m); }
static inline int nsp_pclose(FILE* f) { return _pclose(f); }
static inline void nsp_sleep_ms(unsigned ms) { Sleep(ms); }
static inline int nsp_rename_file(const char* a, const char* b) {
    return MoveFileExA(a, b, MOVEFILE_REPLACE_EXISTING) ? 0 : -1;
}

static inline std::string nsp_temp_dir() {
    char buf[MAX_PATH + 1];
    DWORD n = GetTempPathA(MAX_PATH, buf);
    if (n == 0 || n > MAX_PATH) return ".";
    std::string t(buf, n);
    for (auto& c : t) if (c == '\\') c = '/';
    while (!t.empty() && t.back() == '/') t.pop_back();
    return t;
}

// CRITICAL_SECTION-backed mutex, same shape as the pthread version.
struct nsp_fs_mutex {
    CRITICAL_SECTION cs;
    nsp_fs_mutex() { InitializeCriticalSection(&cs); }
    ~nsp_fs_mutex() { DeleteCriticalSection(&cs); }
    void lock() { EnterCriticalSection(&cs); }
    void unlock() { LeaveCriticalSection(&cs); }
};

#else // POSIX

#  include <unistd.h>
#  include <pthread.h>

typedef struct stat nsp_stat_t;
static inline int nsp_stat(const char* p, nsp_stat_t* s) { return stat(p, s); }
static inline int nsp_sys_mkdir(const char* p) { return mkdir(p, 0777); }
static inline int nsp_sys_rmdir(const char* p) { return rmdir(p); }
static inline int nsp_sys_access(const char* p, int m) { return access(p, m); }
static inline int nsp_getpid(void) { return (int)getpid(); }
static inline FILE* nsp_popen(const char* c, const char* m) { return popen(c, m); }
static inline int nsp_pclose(FILE* f) { return pclose(f); }
static inline void nsp_sleep_ms(unsigned ms) { usleep(ms * 1000); }
static inline int nsp_rename_file(const char* a, const char* b) { return rename(a, b); }
static inline std::string nsp_temp_dir() { return "/tmp"; }

struct nsp_fs_mutex {
    pthread_mutex_t m;
    nsp_fs_mutex() { pthread_mutex_init(&m, nullptr); }
    ~nsp_fs_mutex() { pthread_mutex_destroy(&m); }
    void lock() { pthread_mutex_lock(&m); }
    void unlock() { pthread_mutex_unlock(&m); }
};

#endif // _WIN32
