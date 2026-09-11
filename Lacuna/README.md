# Lacuna

**by Orders of Magnitude · ofmagnitude.com**

Encryption and binary hardening layer for NS Suite deployments.

## What Lacuna Does

Lacuna protects compiled NS Suite binaries for commercial distribution:
- AES-256-GCM encryption of binary sections at rest
- License key generation and validation (offline, cryptographic)
- Binary checkpoint patching — inject license checks without recompilation
- Symbol stripping and section hardening

Ships as: `liblacuna.a` + `lacuna.h` + `keygen` binary + `patcher` binary.

## Status

Feature-complete. Pending: Highway integration, CMake cross-platform build, symbol stripping pass.

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

```bash
# Generate a license key
./keygen --product nssort --tier indie --expiry 2027-01-01

# Patch a binary with license check
./patcher --binary libnssort.so --key <license_key>
```

## License

OOM Commercial License v1.0 · https://ofmagnitude.com/license
