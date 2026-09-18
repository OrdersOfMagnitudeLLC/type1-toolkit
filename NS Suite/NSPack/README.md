# NSPack

Filesystem-transparent semantic compressor. Mounts a directory through FUSE
and compresses supported file types on write, decompresses on read. OBJ, PLY,
BVH, WAV, PNG/JPG, HDR, SQLite, scientific arrays, and ELF get schema-aware
codecs; everything else passes through unchanged.

## Build

```bash
cmake -B build && cmake --build build
```

Requires libfuse3 on Linux/macOS, WinFSP on Windows. Highway SIMD is used
when found (see CMakeLists.txt for HWY_ROOT/HWY_LIB hints).

## Run

```bash
./build/nspack mount <dir> <store>     # mount dir, compressed data in store
./build/nspack unmount <dir>
./build/nspack status <dir>            # space saved
./build/nspack bench <files...>        # per-file ratio vs gzip/zstd
```

## Parameters

None.

## Results

Representative ratios (nspack vs gzip vs zstd, `nspack bench`):

| File type | nspack | gzip | zstd |
|-----------|--------|------|------|
| .obj (3D mesh) | 6.9x | 2.8x | 2.8x |
| .tga (image) | 950x | 1.0x | 1.0x |
| .wav (audio) | 1.2x | 1.0x | 1.0x |

## Third-party

- **zstd**: BSD License (system library)
- **LZ4**: BSD License (system library)
- **FUSE3 / WinFSP**: LGPL / GPLv3 (filesystem interface)
- **Highway**: Apache 2.0 (optional SIMD backend)
- **stb_image / stb_image_write**: public domain (vendored)
