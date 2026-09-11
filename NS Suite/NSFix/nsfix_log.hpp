// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace nsfix {

struct LogEntryHeader {
    uint32_t magic;
    uint32_t msg_len;
};

static constexpr uint32_t LOG_MAGIC = 0x4658584C;
static constexpr size_t LOG_FILE_SIZE = 256 * 1024 * 1024;
static constexpr size_t RING_SIZE = 2 * 1024 * 1024;
static constexpr int STAGING_MAX_MSGS = 32;

class FixLog {
public:
    FixLog(const char* path = "nsfix_session.log")
        : path_(path), fd_(-1), ring_(nullptr),
          write_pos_(0), staging_count_(0) {

        ring_ = (char*)mmap(nullptr, RING_SIZE, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ring_ == MAP_FAILED) { ring_ = nullptr; return; }
        memset(ring_, 0, RING_SIZE);

        fd_ = open(path_, O_RDWR | O_CREAT, 0644);
        if (fd_ < 0) return;

        struct stat st;
        if (fstat(fd_, &st) == 0 && st.st_size < (off_t)LOG_FILE_SIZE) {
            if (ftruncate(fd_, LOG_FILE_SIZE) != 0) {}
        }

        void* ro = mmap(nullptr, LOG_FILE_SIZE, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (ro != MAP_FAILED) {
            LogEntryHeader* first = (LogEntryHeader*)ro;
            if (first->magic == LOG_MAGIC) {
                size_t copy_len = LOG_FILE_SIZE < RING_SIZE ? LOG_FILE_SIZE : RING_SIZE;
                memcpy(ring_, ro, copy_len);
                size_t pos = 0;
                while (pos + sizeof(LogEntryHeader) < RING_SIZE) {
                    LogEntryHeader* hdr = (LogEntryHeader*)(ring_ + pos);
                    if (hdr->magic != LOG_MAGIC || hdr->msg_len == 0 ||
                        pos + sizeof(LogEntryHeader) + hdr->msg_len > LOG_FILE_SIZE)
                        break;
                    pos += sizeof(LogEntryHeader) + hdr->msg_len;
                }
                write_pos_ = pos;
            }
            munmap(ro, LOG_FILE_SIZE);
        }
    }

    ~FixLog() {
        flush_staging();
        if (fd_ >= 0 && ring_) {
            if (write_pos_ > 0) pwrite(fd_, ring_, write_pos_, 0);
            fdatasync(fd_);
        }
        if (fd_ >= 0) close(fd_);
        if (ring_) munmap(ring_, RING_SIZE);
    }

    // Hot path: store ptr+len only (16 bytes). Data copied to ring on flush.
    __attribute__((always_inline)) void log_message(const char* buf, size_t len) {
        if (staging_count_ >= STAGING_MAX_MSGS) flush_staging();
        staging_[staging_count_].ptr = buf;
        staging_[staging_count_].len = (uint32_t)len;
        ++staging_count_;
    }

    void flush() { flush_staging(); }

    void reset() {
        flush_staging();
        write_pos_ = 0;
        memset(ring_, 0, RING_SIZE);
    }

    std::vector<std::string> recover_session(std::string_view sender, std::string_view target,
                                              uint32_t gap_start) {
        flush_staging();
        std::vector<std::string> replay_msgs;
        if (!ring_) return replay_msgs;

        size_t pos = 0;
        while (pos + sizeof(LogEntryHeader) < RING_SIZE) {
            LogEntryHeader* hdr = (LogEntryHeader*)(ring_ + pos);
            if (hdr->magic != LOG_MAGIC || hdr->msg_len == 0) break;
            if (pos + sizeof(LogEntryHeader) + hdr->msg_len > RING_SIZE) break;

            const char* msg = ring_ + pos + sizeof(LogEntryHeader);
            if (matches_session(msg, hdr->msg_len, sender, target)) {
                uint32_t seq = extract_seq(msg, hdr->msg_len);
                if (seq > gap_start)
                    replay_msgs.emplace_back(msg, hdr->msg_len);
            }
            pos += sizeof(LogEntryHeader) + hdr->msg_len;
        }
        return replay_msgs;
    }

    size_t write_position() const {
        size_t total = write_pos_;
        for (int i = 0; i < staging_count_; i++)
            total += sizeof(LogEntryHeader) + staging_[i].len;
        return total;
    }

    bool is_open() const { return ring_ != nullptr; }

private:
    const char* path_;
    int fd_;
    char* ring_;
    size_t write_pos_;

    struct StagingEntry {
        const char* ptr;
        uint32_t len;
    };
    StagingEntry staging_[STAGING_MAX_MSGS];
    int staging_count_;

    __attribute__((always_inline)) void flush_staging() {
        if (staging_count_ == 0) return;
        // Calculate total bytes
        size_t total = 0;
        for (int i = 0; i < staging_count_; i++)
            total += sizeof(LogEntryHeader) + staging_[i].len;
        // Wrap if needed
        if (write_pos_ + total > RING_SIZE) write_pos_ = 0;
        // Batch copy all entries to ring
        char* dst = ring_ + write_pos_;
        for (int i = 0; i < staging_count_; i++) {
            *(uint32_t*)(dst) = LOG_MAGIC;
            *(uint32_t*)(dst + 4) = staging_[i].len;
            memcpy(dst + 8, staging_[i].ptr, staging_[i].len);
            dst += sizeof(LogEntryHeader) + staging_[i].len;
        }
        write_pos_ += total;
        staging_count_ = 0;
    }

    static bool matches_session(const char* msg, size_t len,
                                std::string_view sender, std::string_view target) {
        bool sm = false, tm = false;
        size_t i = 0;
        while (i < len) {
            size_t eq = i;
            while (eq < len && msg[eq] != '=') eq++;
            if (eq >= len) break;
            uint32_t tag = 0;
            for (size_t k = i; k < eq; k++)
                if (msg[k] >= '0' && msg[k] <= '9') tag = tag * 10 + (msg[k] - '0');
            size_t soh = eq + 1;
            while (soh < len && msg[soh] != '\x01') soh++;
            std::string_view val(msg + eq + 1, soh - eq - 1);
            if (tag == 49 && val == sender) sm = true;
            if (tag == 56 && val == target) tm = true;
            i = soh + 1;
        }
        return sm && tm;
    }

    static uint32_t extract_seq(const char* msg, size_t len) {
        size_t i = 0;
        while (i < len) {
            size_t eq = i;
            while (eq < len && msg[eq] != '=') eq++;
            if (eq >= len) break;
            uint32_t tag = 0;
            for (size_t k = i; k < eq; k++)
                if (msg[k] >= '0' && msg[k] <= '9') tag = tag * 10 + (msg[k] - '0');
            size_t soh = eq + 1;
            while (soh < len && msg[soh] != '\x01') soh++;
            if (tag == 34) {
                uint32_t seq = 0;
                for (size_t k = eq + 1; k < soh; k++)
                    if (msg[k] >= '0' && msg[k] <= '9') seq = seq * 10 + (msg[k] - '0');
                return seq;
            }
            i = soh + 1;
        }
        return 0;
    }
};

} // namespace nsfix
