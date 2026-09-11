#pragma once
#include <cstddef>

#ifdef NS_ATTEND_ENABLED

// Runs NSAttend on one head of the prefill attention.
// Caller is responsible for passing contiguous K/V buffers [seq_len, head_dim].
void ns_attend_head(float* out,
                    const float* q,
                    const float* k_buf,
                    const float* v_buf,
                    int seq_len,
                    int head_dim,
                    float scale);

#endif // NS_ATTEND_ENABLED
