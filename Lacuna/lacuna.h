// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#ifndef LACUNA_H
#define LACUNA_H
#include <stdint.h>
#include <stddef.h>

// Returns 1 if license valid, 0 if invalid, -1 on VM error
// r7_value is optional (set to 0 to not override r7)
int lacuna_check(const uint8_t *license_key_32bytes,
                 const uint8_t *nlang_program,
                 size_t         nlang_num_instructions,
                 uint64_t       r7_value);
#endif
