// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#pragma once

#include <cstdint>
#include <cstring>
#include <string_view>
#include <array>
#include <immintrin.h>

namespace nsfix {

// Diagnostic counter for MAX_TOTAL cap behavior
static size_t feed_batch_cap_hits = 0;

// Tag numbers for common FIX fields
namespace Tag {
    constexpr uint16_t BeginString = 8;
    constexpr uint16_t BodyLength = 9;
    constexpr uint16_t CheckSum = 10;
    constexpr uint16_t MsgType = 35;
    constexpr uint16_t SenderCompID = 49;
    constexpr uint16_t TargetCompID = 56;
    constexpr uint16_t MsgSeqNum = 34;
    constexpr uint16_t SendingTime = 52;
    constexpr uint16_t Symbol = 55;
    constexpr uint16_t Side = 54;
    constexpr uint16_t OrderQty = 38;
    constexpr uint16_t OrdType = 40;
    constexpr uint16_t Price = 44;
    constexpr uint16_t ClOrdID = 11;
    constexpr uint16_t HeartBtInt = 108;
    constexpr uint16_t TestReqID = 112;
    constexpr uint16_t BeginSeqNo = 7;
    constexpr uint16_t EndSeqNo = 16;
    constexpr uint16_t NewSeqNo = 36;
    constexpr uint16_t GapFillFlag = 123;
    constexpr uint16_t ResetSeqNumFlag = 112;
    constexpr uint16_t Text = 58;
}

// Session state machine
enum class State {
    DISCONNECTED,
    LOGON_SENT,
    ACTIVE,
    LOGOUT_SENT
};

// Message struct: fixed array of string_view slots, zero-copy
struct Message {
    static constexpr size_t MAX_SLOTS = 64;
    std::string_view slots[MAX_SLOTS];
    uint64_t dirty_mask = 0;  // one bit per slot, tracks which slots were set
    
    void reset() {
        uint64_t mask = dirty_mask;
        while (mask) {
            int i = __builtin_ctzll(mask);
            slots[i] = {};
            mask &= mask - 1;
        }
        dirty_mask = 0;
    }
    
    std::string_view get(uint16_t tag) const {
        extern const std::array<uint8_t, 10001> dispatch_array;
        uint8_t slot = dispatch_array[tag];
        if (slot == 0) return {};
        return slots[slot];
    }
    
    int64_t get_int(uint16_t tag) const {
        std::string_view sv = get(tag);
        if (sv.empty()) return 0;
        
        int64_t result = 0;
        bool negative = false;
        const char* p = sv.data();
        size_t len = sv.length();
        
        if (len > 0 && p[0] == '-') {
            negative = true;
            p++;
            len--;
        }
        
        for (size_t i = 0; i < len; i++) {
            if (p[i] >= '0' && p[i] <= '9') {
                result = result * 10 + (p[i] - '0');
            }
        }
        
        return negative ? -result : result;
    }
    
    double get_double(uint16_t tag) const {
        std::string_view sv = get(tag);
        if (sv.empty()) return 0.0;
        
        double result = 0.0;
        bool negative = false;
        const char* p = sv.data();
        size_t len = sv.length();
        size_t i = 0;
        
        if (len > 0 && p[0] == '-') {
            negative = true;
            p++;
            len--;
        }
        
        // Integer part
        while (i < len && p[i] >= '0' && p[i] <= '9') {
            result = result * 10.0 + (p[i] - '0');
            i++;
        }
        
        // Fractional part
        if (i < len && p[i] == '.') {
            i++;
            double divisor = 10.0;
            while (i < len && p[i] >= '0' && p[i] <= '9') {
                result += (p[i] - '0') / divisor;
                divisor *= 10.0;
                i++;
            }
        }
        
        return negative ? -result : result;
    }
};

// Dispatch table: tag number → slot index (0 = not tracked)
// Slot indices start at 1 (0 reserved for "not tracked")
const std::array<uint8_t, 10001> dispatch_array = [] {
    std::array<uint8_t, 10001> table{};
    table.fill(0);
    uint8_t next_slot = 1;
    
    // Map common tags to slots
    auto assign = [&](uint16_t tag) {
        if (table[tag] == 0 && next_slot < Message::MAX_SLOTS) {
            table[tag] = next_slot++;
        }
    };
    
    assign(Tag::BeginString);
    assign(Tag::BodyLength);
    assign(Tag::CheckSum);
    assign(Tag::MsgType);
    assign(Tag::SenderCompID);
    assign(Tag::TargetCompID);
    assign(Tag::MsgSeqNum);
    assign(Tag::SendingTime);
    assign(Tag::Symbol);
    assign(Tag::Side);
    assign(Tag::OrderQty);
    assign(Tag::OrdType);
    assign(Tag::Price);
    assign(Tag::ClOrdID);
    assign(Tag::HeartBtInt);
    assign(Tag::TestReqID);
    assign(Tag::BeginSeqNo);
    assign(Tag::EndSeqNo);
    assign(Tag::NewSeqNo);
    assign(Tag::GapFillFlag);
    assign(Tag::ResetSeqNumFlag);
    assign(Tag::Text);
    
    return table;
}();

// Session class
class Session {
public:
    using Callback = void(*)(Message&);
    
    Session(std::string_view sender, std::string_view target, Callback cb) 
        : callback_(cb), state_(State::DISCONNECTED), 
          seq_in_(1), seq_out_(1) {
        memset(send_buffer_, 0, sizeof(send_buffer_));
        memset(sender_, 0, sizeof(sender_));
        memset(target_, 0, sizeof(target_));
        size_t sender_len = sender.length() < 16 ? sender.length() : 15;
        size_t target_len = target.length() < 16 ? target.length() : 15;
        memcpy(sender_, sender.data(), sender_len);
        memcpy(target_, target.data(), target_len);
    }
    
    // Feed incoming data (hot path)
    void feed(const char* buf, size_t len) {
        static constexpr int MAX_FIELDS = 64;
        uint16_t soh_pos[MAX_FIELDS];
        uint16_t eq_pos[MAX_FIELDS];
        int soh_count = 0, eq_count = 0;

#ifdef __AVX2__
        const __m256i soh_vec = _mm256_set1_epi8('\x01');
        const __m256i eq_vec  = _mm256_set1_epi8('=');

        // PASS 1: Single AVX2 sweep, record all positions
        size_t pos = 0;
        while (pos + 32 <= len && soh_count < MAX_FIELDS - 4) {
            __m256i chunk   = _mm256_loadu_si256(
                                reinterpret_cast<const __m256i*>(buf + pos));
            int soh_mask    = _mm256_movemask_epi8(
                                _mm256_cmpeq_epi8(chunk, soh_vec));
            int eq_mask_val = _mm256_movemask_epi8(
                                _mm256_cmpeq_epi8(chunk, eq_vec));

            while (soh_mask) {
                int bit = __builtin_ctz(soh_mask);
                soh_pos[soh_count++] = static_cast<uint16_t>(pos + bit);
                soh_mask &= soh_mask - 1;
            }
            while (eq_mask_val) {
                int bit = __builtin_ctz(eq_mask_val);
                eq_pos[eq_count++] = static_cast<uint16_t>(pos + bit);
                eq_mask_val &= eq_mask_val - 1;
            }
            pos += 32;
        }
        // Scalar tail
        for (size_t i = pos; i < len; i++) {
            if (buf[i] == '\x01' && soh_count < MAX_FIELDS) soh_pos[soh_count++] = i;
            if (buf[i] == '='    && eq_count  < MAX_FIELDS) eq_pos[eq_count++]   = i;
        }
#else
        // Scalar fallback: plain C++ loop
        for (size_t i = 0; i < len; i++) {
            if (buf[i] == '\x01' && soh_count < MAX_FIELDS) soh_pos[soh_count++] = i;
            if (buf[i] == '='    && eq_count  < MAX_FIELDS) eq_pos[eq_count++]   = i;
        }
#endif

        // PASS 2: Process fields: zero searching, pure computation
        Message msg;
        msg.reset();

        uint16_t field_start = 0;
        int ei = 0;  // index into eq_pos array

        for (int fi = 0; fi < soh_count; fi++) {
            uint16_t soh = soh_pos[fi];

            // Advance eq pointer to first '=' in this field
            while (ei < eq_count && eq_pos[ei] < field_start) ei++;

            if (ei >= eq_count || eq_pos[ei] >= soh) {
                field_start = soh + 1;
                continue;  // malformed field, skip
            }

            uint16_t eq = eq_pos[ei++];
            uint8_t  tag_len   = eq - field_start;
            const char* tp     = buf + field_start;

            // Branchless tag atoi: dynamic, handles ANY tag number
            uint32_t tag = 0;
            switch (tag_len) {
                case 1: tag = tp[0]-'0'; break;
                case 2: tag = (tp[0]-'0')*10  + (tp[1]-'0'); break;
                case 3: tag = (tp[0]-'0')*100 + (tp[1]-'0')*10 + (tp[2]-'0'); break;
                case 4: tag = (tp[0]-'0')*1000+ (tp[1]-'0')*100
                            + (tp[2]-'0')*10  + (tp[3]-'0'); break;
                default:
                    for (uint8_t k = 0; k < tag_len; k++)
                        tag = tag * 10 + (tp[k] - '0');
            }

            uint8_t slot = (tag < 10001) ? dispatch_array[tag] : 0;
            if (slot) {
                msg.slots[slot] = std::string_view(
                    buf + eq + 1,
                    soh - eq - 1);
                msg.dirty_mask |= (1ULL << slot);
            }

            field_start = soh + 1;
        }

        handle_message(msg);
    }
    
    size_t feed_batch(const char* buf, size_t len) {
        static constexpr size_t WINDOW  = 65536;  // bytes per window - fixed optimal size
        static constexpr int    MAX_POS = 16384;   // max SOH/EQ per window

        uint16_t soh_pos[MAX_POS];
        uint16_t eq_pos[MAX_POS];

#ifdef __AVX2__
        const __m256i soh_vec = _mm256_set1_epi8('\x01');
        const __m256i eq_vec  = _mm256_set1_epi8('=');
#endif

        Message msg;
        msg.reset();
        size_t msg_count  = 0;
        size_t field_abs  = 0;  // absolute offset of current field start in buf

        size_t wstart = 0;
        while (wstart < len) {
            size_t wend = std::min(len, wstart + WINDOW);
            if (wend < len) {
                size_t safe = wend;
                while (safe > wstart && buf[safe-1] != '\x01') safe--;
                if (safe > wstart) wend = safe;  // cut at last clean SOH
            }
            int soh_count = 0, eq_count = 0;

            // --- STAGE 1: SIMD sweep this window ---
#ifdef __AVX2__
            size_t pos = wstart;
            while (pos + 32 <= wend) {
                __m256i chunk = _mm256_loadu_si256(
                                  reinterpret_cast<const __m256i*>(buf + pos));
                int sm = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, soh_vec));
                int em = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, eq_vec));
                while (sm && soh_count < MAX_POS) {
                    soh_pos[soh_count++] = static_cast<uint16_t>(pos + __builtin_ctz(sm) - wstart);
                    sm &= sm-1;
                }
                while (em && eq_count < MAX_POS) {
                    eq_pos[eq_count++] = static_cast<uint16_t>(pos + __builtin_ctz(em) - wstart);
                    em &= em-1;
                }
                pos += 32;
            }
            for (size_t i = pos; i < wend; i++) {
                if (buf[i]=='\x01' && soh_count < MAX_POS)
                    soh_pos[soh_count++] = static_cast<uint16_t>(i - wstart);
                if (buf[i]=='='    && eq_count  < MAX_POS)
                    eq_pos[eq_count++]  = static_cast<uint16_t>(i - wstart);
            }
#else
            // Scalar fallback: plain C++ loop
            for (size_t i = wstart; i < wend; i++) {
                if (buf[i]=='\x01' && soh_count < MAX_POS)
                    soh_pos[soh_count++] = static_cast<uint16_t>(i - wstart);
                if (buf[i]=='='    && eq_count  < MAX_POS)
                    eq_pos[eq_count++]  = static_cast<uint16_t>(i - wstart);
            }
#endif

            // --- STAGE 2: Process fields in this window ---
            int ei = 0;
            for (int fi = 0; fi < soh_count; fi++) {
                size_t soh_abs = wstart + soh_pos[fi];

                // Find eq for this field (absolute)
                while (ei < eq_count && (wstart + eq_pos[ei]) < field_abs) ei++;
                if (ei >= eq_count) { field_abs = soh_abs + 1; continue; }
                size_t eq_abs = wstart + eq_pos[ei];
                if (eq_abs >= soh_abs) { field_abs = soh_abs + 1; continue; }
                ei++;

                uint8_t  tag_len = static_cast<uint8_t>(eq_abs - field_abs);
                const char* tp   = buf + field_abs;

                uint32_t tag = 0;
                switch (tag_len) {
                    case 1: tag = tp[0]-'0'; break;
                    case 2: tag = (tp[0]-'0')*10   + (tp[1]-'0'); break;
                    case 3: tag = (tp[0]-'0')*100  + (tp[1]-'0')*10  + (tp[2]-'0'); break;
                    case 4: tag = (tp[0]-'0')*1000 + (tp[1]-'0')*100
                                + (tp[2]-'0')*10   + (tp[3]-'0'); break;
                    default:
                        for (uint8_t k = 0; k < tag_len; k++)
                            tag = tag*10 + (tp[k]-'0');
                }

                if (tag == 10) {
                    // End of message: dispatch and reset
                    handle_message(msg);
                    msg_count++;
                    msg.reset();
                    field_abs = soh_abs + 1;
                    continue;
                }

                uint8_t slot = (tag < 10001) ? dispatch_array[tag] : 0;
                if (slot) {
                    msg.slots[slot] = std::string_view(buf + eq_abs + 1,
                                                       soh_abs - eq_abs - 1);
                    msg.dirty_mask |= (1ULL << slot);
                }
                field_abs = soh_abs + 1;
            }
            wstart = wend;
        }
        seq_in_ += msg_count;  // Bulk increment for all processed messages
        return msg_count;
    }
    
    // Get cap_hits counter for diagnostics
    static size_t get_cap_hits() {
        return feed_batch_cap_hits;
    }
    
    // Reset cap_hits counter
    static void reset_cap_hits() {
        feed_batch_cap_hits = 0;
    }
    
    
    // Send logon message
    std::string_view logon() {
        uint32_t heartbt = 30;
        
        char* p = send_buffer_;
        
        // BeginString
        p = append_field(p, Tag::BeginString, "FIX.4.2");
        
        // BodyLength placeholder (will fill later)
        char* body_len_pos = p;
        p += 3; // "9=XXX"
        *p++ = '\x01';
        
        // MsgType
        p = append_field(p, Tag::MsgType, "A");
        
        // SenderCompID
        p = append_field(p, Tag::SenderCompID, sender_);
        
        // TargetCompID
        p = append_field(p, Tag::TargetCompID, target_);
        
        // MsgSeqNum
        p = append_field(p, Tag::MsgSeqNum, seq_out_++);
        
        // SendingTime (simplified)
        p = append_field(p, Tag::SendingTime, "20240101-12:00:00");
        
        // HeartBtInt
        p = append_field(p, Tag::HeartBtInt, heartbt);
        
        // Calculate body length
        size_t body_len = p - (body_len_pos + 4); // Skip "9=" and SOH
        
        // Write body length
        char len_buf[16];
        int len_digits = 0;
        uint32_t temp = body_len;
        do {
            len_buf[len_digits++] = '0' + (temp % 10);
            temp /= 10;
        } while (temp > 0);
        
        char* len_p = body_len_pos + 2;
        for (int i = len_digits - 1; i >= 0; i--) {
            *len_p++ = len_buf[i];
        }
        *len_p = '\x01';
        
        // Calculate checksum
        uint32_t checksum = 0;
        for (char* q = send_buffer_; q < p; q++) {
            checksum += static_cast<uint8_t>(*q);
        }
        checksum = checksum % 256;
        
        // Append checksum
        p = append_field(p, Tag::CheckSum, checksum);
        
        send_len_ = p - send_buffer_;
        state_ = State::LOGON_SENT;
        return std::string_view(send_buffer_, send_len_);
    }
    
    // Send logout message
    std::string_view logout() {
        char* p = send_buffer_;
        
        // BeginString
        p = append_field(p, Tag::BeginString, "FIX.4.2");
        
        // BodyLength placeholder (will fill later)
        char* body_len_pos = p;
        p += 3; // "9=XXX"
        *p++ = '\x01';
        
        // MsgType
        p = append_field(p, Tag::MsgType, "5");
        
        // SenderCompID
        p = append_field(p, Tag::SenderCompID, sender_);
        
        // TargetCompID
        p = append_field(p, Tag::TargetCompID, target_);
        
        // MsgSeqNum
        p = append_field(p, Tag::MsgSeqNum, seq_out_++);
        
        // SendingTime (simplified)
        p = append_field(p, Tag::SendingTime, "20240101-12:00:00");
        
        // Calculate body length
        size_t body_len = p - (body_len_pos + 4); // Skip "9=" and SOH
        
        // Write body length
        char len_buf[16];
        int len_digits = 0;
        uint32_t temp = body_len;
        do {
            len_buf[len_digits++] = '0' + (temp % 10);
            temp /= 10;
        } while (temp > 0);
        
        char* len_p = body_len_pos + 2;
        for (int i = len_digits - 1; i >= 0; i--) {
            *len_p++ = len_buf[i];
        }
        *len_p = '\x01';
        
        // Calculate checksum
        uint32_t checksum = 0;
        for (char* q = send_buffer_; q < p; q++) {
            checksum += static_cast<uint8_t>(*q);
        }
        checksum = checksum % 256;
        
        // Append checksum
        p = append_field(p, Tag::CheckSum, checksum);
        
        send_len_ = p - send_buffer_;
        state_ = State::LOGOUT_SENT;
        return std::string_view(send_buffer_, send_len_);
    }
    
    State state() const { return state_; }
    uint32_t seq_in() const { return seq_in_; }
    uint32_t seq_out() const { return seq_out_; }
    
private:
    Callback callback_;
    State state_;
    uint32_t seq_in_;
    uint32_t seq_out_;
    char send_buffer_[4096];
    size_t send_len_;
    char sender_[16];
    char target_[16];
    
    // Fast atoi for tag parsing
    static uint16_t fast_atoi(const char* p, size_t len) {
        uint16_t result = 0;
        for (size_t i = 0; i < len && i < 5; i++) {
            if (p[i] >= '0' && p[i] <= '9') {
                result = result * 10 + (p[i] - '0');
            }
        }
        return result;
    }
    
    // Parse a single field
    void parse_field(const char* field, size_t field_len, Message& msg) {
        const char* eq = static_cast<const char*>(memchr(field, '=', field_len));
        if (!eq) return;
        
        size_t tag_len = eq - field;
        
        // SWAR: load exact tag bytes, compare tag in one op
        uint32_t tag_word = 0;
        memcpy(&tag_word, field, tag_len);
        
        uint8_t slot = 0;
        switch (tag_word) {
            case '8':                               slot = dispatch_array[8];  break;
            case '9':                               slot = dispatch_array[9];  break;
            case '1'|('0'<<8):                      slot = dispatch_array[10]; break;
            case '1'|('1'<<8):                      slot = dispatch_array[11]; break;
            case '3'|('4'<<8):                      slot = dispatch_array[34]; break;
            case '3'|('5'<<8):                      slot = dispatch_array[35]; break;
            case '3'|('8'<<8):                      slot = dispatch_array[38]; break;
            case '4'|('0'<<8):                      slot = dispatch_array[40]; break;
            case '4'|('4'<<8):                      slot = dispatch_array[44]; break;
            case '4'|('9'<<8):                      slot = dispatch_array[49]; break;
            case '5'|('2'<<8):                      slot = dispatch_array[52]; break;
            case '5'|('4'<<8):                      slot = dispatch_array[54]; break;
            case '5'|('5'<<8):                      slot = dispatch_array[55]; break;
            case '5'|('6'<<8):                      slot = dispatch_array[56]; break;
            default:
                // Uncommon tag: fall back to atoi
                uint32_t t = 0;
                for (const char* p = field; p < eq; p++)
                    t = t * 10 + (*p - '0');
                slot = (t < 10001) ? dispatch_array[t] : 0;
        }
        
        if (slot == 0) return; // Not tracked
        
        const char* value = eq + 1;
        size_t value_len = field_len - tag_len - 1;
        msg.slots[slot] = std::string_view(value, value_len);
        msg.dirty_mask |= (1ULL << slot);
    }
    
    // Handle message based on type
    void handle_message(Message& msg) {
        std::string_view msg_type = msg.get(Tag::MsgType);
        if (msg_type.empty()) return;
        
        char type = msg_type[0];
        
        switch (type) {
            case 'A': // Logon
                handle_logon(msg);
                break;
            case '5': // Logout
                handle_logout(msg);
                break;
            case '0': // Heartbeat
                handle_heartbeat(msg);
                break;
            case '2': // Resend Request
                handle_resend_request(msg);
                break;
            case '4': // Sequence Reset
                handle_sequence_reset(msg);
                break;
            default:  // Application message
                if (state_ == State::ACTIVE && callback_) {
                    callback_(msg);
                }
                break;
        }
    }
    
    void handle_logon(Message& msg) {
        // Validate sequence number
        uint32_t seq = msg.get_int(Tag::MsgSeqNum);
        if (seq == seq_in_) {
            state_ = State::ACTIVE;
        }
    }
    
    void handle_logout(Message& msg) {
        uint32_t seq = msg.get_int(Tag::MsgSeqNum);
        if (seq == seq_in_) {
            seq_in_++;
            state_ = State::DISCONNECTED;
        }
    }
    
    void handle_heartbeat(Message& msg) {
        uint32_t seq = msg.get_int(Tag::MsgSeqNum);
        if (seq == seq_in_) {
            // seq_in_ will be incremented in bulk at end of feed_batch()
        }
    }
    
    void handle_resend_request(Message& msg) {
        uint32_t seq = msg.get_int(Tag::MsgSeqNum);
        if (seq == seq_in_) {
            seq_in_++;
            
            // Send sequence reset to fill gap
            uint32_t new_seq = seq_out_;
            
            char* p = send_buffer_;
            
            // BeginString
            p = append_field(p, Tag::BeginString, "FIX.4.2");
            
            // BodyLength placeholder (will fill later)
            char* body_len_pos = p;
            p += 3; // "9=XXX"
            *p++ = '\x01';
            
            // MsgType
            p = append_field(p, Tag::MsgType, "4");
            
            // SenderCompID
            p = append_field(p, Tag::SenderCompID, sender_);
            
            // TargetCompID
            p = append_field(p, Tag::TargetCompID, target_);
            
            // MsgSeqNum
            p = append_field(p, Tag::MsgSeqNum, seq_out_++);
            
            // SendingTime (simplified)
            p = append_field(p, Tag::SendingTime, "20240101-12:00:00");
            
            // NewSeqNo
            p = append_field(p, Tag::NewSeqNo, new_seq);
            
            // GapFillFlag
            p = append_field(p, Tag::GapFillFlag, 'Y');
            
            // Calculate body length
            size_t body_len = p - (body_len_pos + 4); // Skip "9=" and SOH
            
            // Write body length
            char len_buf[16];
            int len_digits = 0;
            uint32_t temp = body_len;
            do {
                len_buf[len_digits++] = '0' + (temp % 10);
                temp /= 10;
            } while (temp > 0);
            
            char* len_p = body_len_pos + 2;
            for (int i = len_digits - 1; i >= 0; i--) {
                *len_p++ = len_buf[i];
            }
            *len_p = '\x01';
            
            // Calculate checksum
            uint32_t checksum = 0;
            for (char* q = send_buffer_; q < p; q++) {
                checksum += static_cast<uint8_t>(*q);
            }
            checksum = checksum % 256;
            
            // Append checksum
            p = append_field(p, Tag::CheckSum, checksum);
            
            send_len_ = p - send_buffer_;
        }
    }
    
    void handle_sequence_reset(Message& msg) {
        uint32_t seq = msg.get_int(Tag::MsgSeqNum);
        uint32_t new_seq = msg.get_int(Tag::NewSeqNo);
        
        if (new_seq > seq_in_) {
            seq_in_ = new_seq;
        }
    }
    
    // Append field: tag=value\x01
    char* append_field(char* p, uint16_t tag, uint32_t value) {
        char buf[16];
        int digits = 0;
        uint32_t temp = value;
        do {
            buf[digits++] = '0' + (temp % 10);
            temp /= 10;
        } while (temp > 0);
        
        // Write tag
        uint16_t tag_temp = tag;
        char tag_buf[8];
        int tag_digits = 0;
        do {
            tag_buf[tag_digits++] = '0' + (tag_temp % 10);
            tag_temp /= 10;
        } while (tag_temp > 0);
        
        for (int i = tag_digits - 1; i >= 0; i--) {
            *p++ = tag_buf[i];
        }
        *p++ = '=';
        
        // Write value
        for (int i = digits - 1; i >= 0; i--) {
            *p++ = buf[i];
        }
        *p++ = '\x01';
        
        return p;
    }
    
    char* append_field(char* p, uint16_t tag, const char* value) {
        // Write tag
        uint16_t tag_temp = tag;
        char tag_buf[8];
        int tag_digits = 0;
        do {
            tag_buf[tag_digits++] = '0' + (tag_temp % 10);
            tag_temp /= 10;
        } while (tag_temp > 0);
        
        for (int i = tag_digits - 1; i >= 0; i--) {
            *p++ = tag_buf[i];
        }
        *p++ = '=';
        
        // Write value
        while (*value) {
            *p++ = *value++;
        }
        *p++ = '\x01';
        
        return p;
    }
    
    char* append_field(char* p, uint16_t tag, char value) {
        // Write tag
        uint16_t tag_temp = tag;
        char tag_buf[8];
        int tag_digits = 0;
        do {
            tag_buf[tag_digits++] = '0' + (tag_temp % 10);
            tag_temp /= 10;
        } while (tag_temp > 0);
        
        for (int i = tag_digits - 1; i >= 0; i--) {
            *p++ = tag_buf[i];
        }
        *p++ = '=';
        
        *p++ = value;
        *p++ = '\x01';
        
        return p;
    }
};

} // namespace nsfix
