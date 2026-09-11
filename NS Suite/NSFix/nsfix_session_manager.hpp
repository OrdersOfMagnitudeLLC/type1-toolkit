// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <chrono>
#include "../NSHash/nshash.hpp"
#include "nsfix.hpp"

namespace nsfix {

// Session state for multi-session management
struct FIXSessionState {
    uint32_t seq_in;
    uint32_t seq_out;
    State state;
    std::chrono::steady_clock::time_point last_heartbeat;
    Session* session;
    char sender[16];
    char target[16];
};

// Session manager using NSHash for O(1) session lookup by composite key
class SessionManager {
public:
    SessionManager(size_t reserve = 1024) : sessions_() {
        sessions_.reserve(reserve);
    }

    ~SessionManager() {
        for (size_t i = 0; i < states_.size(); i++) {
            delete states_[i];
        }
    }

    // Get or create a session by sender/target CompID
    FIXSessionState* get_or_create_session(std::string_view sender, std::string_view target,
                                           Session::Callback cb) {
        uint64_t key = session_key(sender, target);

        FIXSessionState** ptr = sessions_.lookup(key);
        if (ptr) return *ptr;

        // Create new session
        FIXSessionState* st = new FIXSessionState();
        st->seq_in = 1;
        st->seq_out = 1;
        st->state = State::DISCONNECTED;
        st->last_heartbeat = std::chrono::steady_clock::now();
        st->session = new Session(sender, target, cb);

        memset(st->sender, 0, sizeof(st->sender));
        memset(st->target, 0, sizeof(st->target));
        size_t sl = sender.length() < 16 ? sender.length() : 15;
        size_t tl = target.length() < 16 ? target.length() : 15;
        memcpy(st->sender, sender.data(), sl);
        memcpy(st->target, target.data(), tl);

        sessions_.insert(key, st);
        states_.push_back(st);
        return st;
    }

    // Lookup existing session without creating
    FIXSessionState* lookup_session(std::string_view sender, std::string_view target) {
        uint64_t key = session_key(sender, target);
        FIXSessionState** ptr = sessions_.lookup(key);
        return ptr ? *ptr : nullptr;
    }

    // Remove a session
    bool remove_session(std::string_view sender, std::string_view target) {
        uint64_t key = session_key(sender, target);
        FIXSessionState** ptr = sessions_.lookup(key);
        if (!ptr) return false;
        delete *ptr;
        sessions_.erase(key);
        return true;
    }

    size_t session_count() const { return sessions_.size(); }

    // Feed data to a specific session (routes through session manager)
    void feed(std::string_view sender, std::string_view target, const char* buf, size_t len) {
        FIXSessionState* st = lookup_session(sender, target);
        if (st && st->session) {
            st->session->feed(buf, len);
            st->last_heartbeat = std::chrono::steady_clock::now();
        }
    }

private:
    NSHash<uint64_t, FIXSessionState*, WyHash> sessions_;
    std::vector<FIXSessionState*> states_; // for cleanup

    static uint64_t session_key(std::string_view sender, std::string_view target) {
        // Hash concatenated sender+target into a uint64_t key
        char combined[32] = {};
        size_t sl = sender.length() < 16 ? sender.length() : 15;
        size_t tl = target.length() < 16 ? target.length() : 15;
        memcpy(combined, sender.data(), sl);
        memcpy(combined + 16, target.data(), tl);
        return nsh_hash_str(combined, sl + 16);
    }
};

} // namespace nsfix
