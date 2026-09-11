// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#ifndef LACUNA_API_H
#define LACUNA_API_H
#include <stdint.h>
// Returns 1 if license valid, 0 if invalid, -1 on error
// license_key: 32 raw bytes (decode your 64-char hex key before passing)
// program/num_instructions: your NLang bytecode
int lacuna_check(const uint8_t *license_key_32bytes,
                 const uint8_t *program,
                 size_t num_instructions);
#endif
