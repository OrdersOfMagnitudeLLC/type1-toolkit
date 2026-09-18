// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

// nspack_extra.hpp - Additional file type compressors
#ifndef _WIN32
#include <unistd.h>
#endif
#include <unordered_map>
#include <cstdint>
#include "nspack_hwy.hpp"
#include <cmath>
#include <chrono>
#include <iomanip>

using Clock = std::chrono::high_resolution_clock;

// ===== Shannon Entropy Estimation =====
static double shannon_entropy_bytes(const uint8_t* data, size_t n) {
    if (n == 0) return 0.0;
    uint64_t counts[256] = {0};
    for (size_t i = 0; i < n; i++) counts[data[i]]++;
    double h = 0.0;
    for (int i = 0; i < 256; i++) {
        if (counts[i] == 0) continue;
        double p = (double)counts[i] / n;
        h -= p * log2(p);
    }
    return h;
}

template<typename T>
static double shannon_entropy_typed(const uint8_t* data, size_t n) {
    size_t elem_size = sizeof(T);
    if (n < elem_size) return 0.0;
    size_t N = n / elem_size;
    std::unordered_map<T, uint64_t> counts;
    for (size_t i = 0; i < N; i++) {
        T v;
        memcpy(&v, data + i * elem_size, elem_size);
        counts[v]++;
    }
    double h = 0.0;
    for (const auto& kv : counts) {
        double p = (double)kv.second / N;
        h -= p * log2(p);
    }
    return h;
}

// Shannon limit of T(X): honest first-order byte entropy, no clamping, no multi-byte trick.
static size_t shannon_limit_data(const std::vector<uint8_t>& data) {
    if (data.empty()) return 0;
    size_t n = data.size();
    double h = shannon_entropy_bytes(data.data(), n);
    return (size_t)ceil(h * n / 8.0);
}

static size_t shannon_limit_bytes(const std::vector<uint8_t>& data) {
    return shannon_limit_data(data);
}

static size_t shannon_limit_file(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) return 0;
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
    return shannon_limit_data(data);
}


// ===== Utility functions =====
static uint64_t split3_64(uint64_t x) {
    x &= 0x1fffff;
    x = (x | (x << 32)) & 0x1f00000000ffff;
    x = (x | (x << 16)) & 0x1f0000ff0000ff;
    x = (x | (x << 8))  & 0x100f00f00f00f00f;
    x = (x | (x << 4))  & 0x10c30c30c30c30c3;
    x = (x | (x << 2))  & 0x1249249249249249;
    return x;
}

static uint64_t morton3_64(uint64_t x, uint64_t y, uint64_t z) {
    return split3_64(x) | (split3_64(y) << 1) | (split3_64(z) << 2);
}

static std::vector<int16_t> quantize_floats(const std::vector<float>& vals, float& vmin, float& vmax) {
    vmin = 1e30f; vmax = -1e30f;
    for (float v : vals) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
    float range = vmax - vmin;
    if (range < 1e-10f) range = 1.0f;
    std::vector<int16_t> q(vals.size());
    for (size_t i = 0; i < vals.size(); i++)
        q[i] = (int16_t)(((vals[i] - vmin) / range) * 65535.0f - 32768.0f);
    return q;
}

static std::vector<int32_t> quantize_floats_i32(const std::vector<float>& vals, float& vmin, float& vmax) {
    vmin = 1e30f; vmax = -1e30f;
    for (float v : vals) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
    double range = (double)vmax - (double)vmin;
    if (range < 1e-12) range = 1.0;
    std::vector<int32_t> q(vals.size());
    for (size_t i = 0; i < vals.size(); i++) {
        double t = ((double)vals[i] - (double)vmin) / range;   // [0,1]
        int64_t qi = (int64_t)std::llround(t * 4294967295.0 - 2147483648.0);
        if (qi < -2147483648LL) qi = -2147483648LL;
        if (qi > 2147483647LL) qi = 2147483647LL;
        q[i] = (int32_t)qi;
    }
    return q;
}

template<typename T>
static std::vector<float> dequantize_floats(const std::vector<T>& q, float vmin, float vmax) {
    float range = vmax - vmin;
    if (range < 1e-10f) range = 1.0f;
    std::vector<float> vals(q.size());
    for (size_t i = 0; i < q.size(); i++)
        vals[i] = vmin + ((q[i] + 32768.0f) / 65535.0f) * range;
    return vals;
}

static std::vector<uint8_t> pack_int16_zigzag(const std::vector<int32_t>& v) {
    int64_t max_abs = 0;
    for (int32_t x : v) { int64_t a = x >= 0 ? (int64_t)x : -(int64_t)x; if (a > max_abs) max_abs = a; }
    if (max_abs <= 32767) {
        std::vector<uint8_t> r(1 + v.size() * 2);
        r[0] = 0;
        for (size_t i = 0; i < v.size(); i++) {
            uint32_t z = (uint32_t)(v[i] << 1) ^ (v[i] >> 31);
            r[1 + i*2] = z & 0xff; r[1 + i*2 + 1] = (z >> 8) & 0xff;
        }
        return r;
    } else {
        std::vector<uint8_t> r(1 + v.size() * 4);
        r[0] = 1;
        for (size_t i = 0; i < v.size(); i++) {
            uint32_t z = (uint32_t)(v[i] << 1) ^ (v[i] >> 31);
            r[1 + i*4] = z & 0xff; r[1 + i*4 + 1] = (z >> 8) & 0xff;
            r[1 + i*4 + 2] = (z >> 16) & 0xff; r[1 + i*4 + 3] = (z >> 24) & 0xff;
        }
        return r;
    }
}

static std::vector<uint8_t> pack_int16_zigzag(const std::vector<int16_t>& v) {
    std::vector<int32_t> v32(v.begin(), v.end());
    return pack_int16_zigzag(v32);
}

static std::vector<int32_t> unpack_int16_zigzag(const std::vector<uint8_t>& r) {
    if (r.empty()) return {};
    uint8_t mode = r[0];
    if (mode == 0) {
        size_t n = (r.size() - 1) / 2;
        std::vector<int32_t> v(n);
        for (size_t i = 0; i < n; i++) {
            uint32_t z = r[1 + i*2] | (r[1 + i*2 + 1] << 8);
            v[i] = (int32_t)(int16_t)((z >> 1) ^ (~(z & 1) + 1));
        }
        return v;
    } else {
        size_t n = (r.size() - 1) / 4;
        std::vector<int32_t> v(n);
        for (size_t i = 0; i < n; i++) {
            uint32_t z = r[1 + i*4] | (r[1 + i*4 + 1] << 8) | (r[1 + i*4 + 2] << 16) | (r[1 + i*4 + 3] << 24);
            v[i] = (int32_t)((z >> 1) ^ (~(z & 1) + 1));
        }
        return v;
    }
}

// ===== Audio PCM WAV =====
static const char WAV_MAGIC[4] = {'N','S','A','1'};

static bool parse_wav(const std::string& filename, int& sample_rate, int& channels, int& bits, std::vector<int16_t>& samples) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) return false;
    char riff[4], wave[4];
    f.read(riff, 4);
    uint32_t filesize; f.read((char*)&filesize, 4);
    f.read(wave, 4);
    if (memcmp(riff, "RIFF", 4) != 0 || memcmp(wave, "WAVE", 4) != 0) return false;
    while (f) {
        char chunk[4]; f.read(chunk, 4);
        uint32_t sz; f.read((char*)&sz, 4);
        if (f.eof()) break;
        if (memcmp(chunk, "fmt ", 4) == 0) {
            uint16_t fmt, nch, bps; uint32_t sr, bps2;
            uint16_t block_align;
            f.read((char*)&fmt, 2);
            f.read((char*)&nch, 2);
            f.read((char*)&sr, 4);
            f.read((char*)&bps2, 4);
            f.read((char*)&block_align, 2);
            f.read((char*)&bps, 2);
            sample_rate = sr; channels = nch; bits = bps;
            if (fmt != 1) { std::cerr << "WAV not PCM\n"; return false; }
        } else if (memcmp(chunk, "data", 4) == 0) {
            int nsamp = sz / (channels * (bits/8));
            if (bits == 16) {
                samples.resize(nsamp * channels);
                f.read((char*)samples.data(), sz);
            } else {
                std::cerr << "WAV bits not 16\n"; return false;
            }
        } else {
            f.seekg(sz, std::ios::cur);
        }
    }
    return sample_rate > 0 && channels > 0 && !samples.empty();
}

static bool encode_wav(const std::string& infile, const std::string& outfile, double* enc_ms = nullptr) {
    auto t0 = Clock::now();
    int sr, ch, bits;
    std::vector<int16_t> samples;
    if (!parse_wav(infile, sr, ch, bits, samples)) return false;
    int n = samples.size() / ch;
    std::vector<int32_t> col(samples.size());
    for (int c = 0; c < ch; c++) {
#if defined(NSPACK_HIGHWAY)
        nspack_hwy::deinterleave_i16_i32(samples.data(), col.data() + c*n, n, ch, c);
#else
        for (int i = 0; i < n; i++)
            col[c*n + i] = samples[i*ch + c];
#endif
    }
    for (int c = 0; c < ch; c++) {
#if defined(NSPACK_HIGHWAY)
        nspack_hwy::delta_encode_i32(col.data() + c*n, n);
#else
        for (int i = n-1; i > 0; i--)
            col[c*n + i] -= col[c*n + i-1];
#endif
    }
    std::vector<uint8_t> packed = pack_int16_zigzag(col);
    size_t zb = ZSTD_compressBound(packed.size());
    std::vector<uint8_t> zdata(zb);
    size_t zs = ZSTD_compress(zdata.data(), zb, packed.data(), packed.size(), 19);
    if (ZSTD_isError(zs)) return false;
    std::ofstream out(outfile, std::ios::binary);
    out.write(WAV_MAGIC, 4);
    out.write((char*)&sr, 4); out.write((char*)&ch, 4); out.write((char*)&bits, 4);
    out.write((char*)&n, 4);
    uint32_t zs32 = (uint32_t)zs;
    out.write((char*)&zs32, 4);
    out.write((char*)zdata.data(), zs);
    out.close();
    if (enc_ms) { auto t1 = Clock::now(); *enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(); }
    return true;
}

static bool decode_wav(const std::string& infile, const std::string& outfile, double* dec_ms = nullptr, double* shannon_T = nullptr) {
    auto t0 = Clock::now();
    std::ifstream f(infile, std::ios::binary);
    char magic[4]; f.read(magic, 4);
    if (memcmp(magic, WAV_MAGIC, 4) != 0) return false;
    int sr, ch, bits, n; uint32_t zs;
    f.read((char*)&sr, 4); f.read((char*)&ch, 4); f.read((char*)&bits, 4); f.read((char*)&n, 4); f.read((char*)&zs, 4);
    std::vector<uint8_t> zdata(zs); f.read((char*)zdata.data(), zs);
    unsigned long long psize = ZSTD_getFrameContentSize(zdata.data(), zs);
    if (psize == ZSTD_CONTENTSIZE_ERROR || psize == ZSTD_CONTENTSIZE_UNKNOWN) return false;
    std::vector<uint8_t> packed(psize);
    if (ZSTD_isError(ZSTD_decompress(packed.data(), packed.size(), zdata.data(), zs))) return false;
    if (shannon_T) *shannon_T = (double)shannon_limit_bytes(packed);
    std::vector<int32_t> col = unpack_int16_zigzag(packed);
    for (int c = 0; c < ch; c++) {
#if defined(NSPACK_HIGHWAY)
        nspack_hwy::delta_decode_i32(col.data() + c*n, n);
#else
        for (int i = 1; i < n; i++)
            col[c*n + i] += col[c*n + i-1];
#endif
    }
    std::vector<int16_t> samples(n * ch);
    for (int c = 0; c < ch; c++) {
#if defined(NSPACK_HIGHWAY)
        nspack_hwy::interleave_i32_i16(col.data() + c*n, samples.data(), n, ch, c);
#else
        for (int i = 0; i < n; i++)
            samples[i*ch + c] = (int16_t)col[c*n + i];
#endif
    }
    std::ofstream out(outfile, std::ios::binary);
    out.write("RIFF", 4);
    uint32_t fsz = 36 + n * ch * (bits/8);
    out.write((char*)&fsz, 4);
    out.write("WAVE", 4);
    out.write("fmt ", 4); uint32_t sz16 = 16; out.write((char*)&sz16, 4);
    uint16_t fmt = 1, nch = ch, bps = bits; out.write((char*)&fmt, 2); out.write((char*)&nch, 2);
    out.write((char*)&sr, 4);
    uint32_t bps2 = sr * ch * (bits/8); out.write((char*)&bps2, 4);
    uint16_t align = ch * (bits/8); out.write((char*)&align, 2); out.write((char*)&bps, 2);
    out.write("data", 4); uint32_t dsz = n * ch * (bits/8); out.write((char*)&dsz, 4);
    out.write((char*)samples.data(), dsz);
    out.close();
    if (dec_ms) { auto t1 = Clock::now(); *dec_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(); }
    return true;
}

// ===== Source Code AST =====
static const char AST_MAGIC[4] = {'N','S','C','1'};

static bool encode_code_ast(const std::string& infile, const std::string& outfile, double* enc_ms = nullptr) {
    auto t0 = Clock::now();
    std::string tmp = nsp_temp_dir() + "/nspack_ast_" + std::to_string(nsp_getpid()) + ".bin";
    std::string py = R"PY(import sys, ast, struct
path = sys.argv[1]; out = sys.argv[2]
with open(path) as f: src = f.read()
tree = ast.parse(src)
types = []; depths = []; values = []
def walk(node, d):
    types.append(type(node).__name__); depths.append(d)
    if isinstance(node, ast.Constant):
        v = getattr(node, 'value', '')
        values.append(str(v)[:64])
    elif isinstance(node, ast.Name):
        values.append(getattr(node, 'id', ''))
    else:
        values.append('')
    for child in ast.iter_child_nodes(node): walk(child, d+1)
walk(tree, 0)
type_map = {t:i for i,t in enumerate(sorted(set(types)))}
type_ids = [type_map[t] for t in types]
with open(out, 'wb') as f:
    f.write(struct.pack('<III', len(type_ids), max(1, len(type_map)), len(type_map)))
    for t in sorted(type_map, key=lambda x: type_map[x]):
        f.write(struct.pack('<H', len(t)) + t.encode())
    for i in range(len(type_ids)):
        f.write(struct.pack('<HH', type_ids[i], depths[i]))
        v = values[i].encode()[:255]; f.write(struct.pack('<B', len(v)) + v)
)PY";
    std::string full_cmd = "python3 -c \"" + py + "\" '" + infile + "' '" + tmp + "'";
    if (system(full_cmd.c_str()) != 0) {
        std::cerr << "AST parse failed for " << infile << "\n";
        return false;
    }
    std::ifstream rf(tmp, std::ios::binary);
    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(rf)), std::istreambuf_iterator<char>());
    std::remove(tmp.c_str());
    size_t zb = ZSTD_compressBound(raw.size());
    std::vector<uint8_t> zdata(zb);
    size_t zs = ZSTD_compress(zdata.data(), zb, raw.data(), raw.size(), 19);
    if (ZSTD_isError(zs)) return false;
    std::ofstream out(outfile, std::ios::binary);
    out.write(AST_MAGIC, 4);
    uint32_t zs32 = (uint32_t)zs;
    out.write((char*)&zs32, 4);
    out.write((char*)zdata.data(), zs);
    out.close();
    if (enc_ms) { auto t1 = Clock::now(); *enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(); }
    return true;
}

static bool decode_code_ast(const std::string& infile, const std::string& outfile, double* dec_ms = nullptr, double* shannon_T = nullptr) {
    auto t0 = Clock::now();
    std::ifstream f(infile, std::ios::binary);
    char magic[4]; f.read(magic, 4);
    if (memcmp(magic, AST_MAGIC, 4) != 0) return false;
    uint32_t zs; f.read((char*)&zs, 4);
    std::vector<uint8_t> zdata(zs); f.read((char*)zdata.data(), zs);
    unsigned long long dsz = ZSTD_getFrameContentSize(zdata.data(), zs);
    if (dsz == ZSTD_CONTENTSIZE_ERROR || dsz == ZSTD_CONTENTSIZE_UNKNOWN) { std::cerr << "Unknown decompressed size\n"; return false; }
    std::vector<uint8_t> raw((size_t)dsz);
    size_t ds = ZSTD_decompress(raw.data(), raw.size(), zdata.data(), zs);
    if (shannon_T) *shannon_T = (double)shannon_limit_bytes(raw);
    if (ZSTD_isError(ds)) return false;
    raw.resize(ds);
    std::ofstream out(outfile);
    out << "# AST representation (decompressed " << ds << " bytes)\n";
    out << "# Round-trip source reconstruction not implemented; see original source.\n";
    out.close();
    if (dec_ms) { auto t1 = Clock::now(); *dec_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(); }
    return true;
}


// ===== PLY Point Clouds =====
static const char PLY_MAGIC[4] = {'N','S','P','C'};

static bool parse_ply(const std::string& filename, int& npts, std::vector<float>& xyz, std::vector<float>& intensity) {
    std::ifstream f(filename);
    if (!f) return false;
    std::string line;
    bool in_header = true;
    npts = 0;
    bool has_intensity = false;
    while (in_header && std::getline(f, line)) {
        std::istringstream ss(line);
        std::string tok; ss >> tok;
        if (tok == "element" && (ss >> tok) && tok == "vertex") ss >> npts;
        else if (tok == "property" && (ss >> tok) && tok == "float" && (ss >> tok) && tok == "intensity") has_intensity = true;
        else if (tok == "end_header") in_header = false;
    }
    if (npts == 0) return false;
    xyz.resize(npts * 3);
    intensity.resize(npts);
    for (int i = 0; i < npts; i++) {
        if (!std::getline(f, line)) return false;
        std::istringstream ss(line);
        ss >> xyz[i*3] >> xyz[i*3+1] >> xyz[i*3+2];
        if (has_intensity) ss >> intensity[i];
        else intensity[i] = 0;
    }
    return true;
}

static bool encode_ply(const std::string& infile, const std::string& outfile, double* enc_ms = nullptr) {
    auto t0 = Clock::now();
    int npts;
    std::vector<float> xyz, intensity;
    if (!parse_ply(infile, npts, xyz, intensity)) return false;
    std::vector<float> xslice(npts), yslice(npts), zslice(npts);
    for (int i = 0; i < npts; i++) { xslice[i] = xyz[i*3]; yslice[i] = xyz[i*3+1]; zslice[i] = xyz[i*3+2]; }
    float xmin, xmax, ymn, ymx, zmn, zmx, imn, imx;
    std::vector<int16_t> qx = quantize_floats(xslice, xmin, xmax);
    std::vector<int16_t> qy = quantize_floats(yslice, ymn, ymx);
    std::vector<int16_t> qz = quantize_floats(zslice, zmn, zmx);
    std::vector<int16_t> qi = quantize_floats(intensity, imn, imx);
    std::vector<uint64_t> keys(npts);
    const uint64_t M21 = (1ULL << 21) - 1;
#if defined(NSPACK_HIGHWAY)
    {
        std::vector<uint64_t> ax(npts), ay(npts), az(npts);
        for (int i = 0; i < npts; i++) {
            ax[i] = (uint64_t)(qx[i] + 32768) * M21 / 65535ULL;
            ay[i] = (uint64_t)(qy[i] + 32768) * M21 / 65535ULL;
            az[i] = (uint64_t)(qz[i] + 32768) * M21 / 65535ULL;
        }
        nspack_hwy::morton3_64_batch(ax.data(), ay.data(), az.data(), keys.data(), npts);
    }
#else
    for (int i = 0; i < npts; i++) {
        uint64_t ux = (uint64_t)(qx[i] + 32768) * M21 / 65535ULL;
        uint64_t uy = (uint64_t)(qy[i] + 32768) * M21 / 65535ULL;
        uint64_t uz = (uint64_t)(qz[i] + 32768) * M21 / 65535ULL;
        keys[i] = morton3_64(ux, uy, uz);
    }
#endif
    std::vector<int> idx(npts);
    for (int i = 0; i < npts; i++) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return keys[a] < keys[b]; });
    std::vector<int32_t> sx(npts), sy(npts), sz(npts), si(npts);
    for (int i = 0; i < npts; i++) { sx[i] = qx[idx[i]]; sy[i] = qy[idx[i]]; sz[i] = qz[idx[i]]; si[i] = qi[idx[i]]; }
#if defined(NSPACK_HIGHWAY)
    nspack_hwy::delta_encode_i32(sx.data(), npts);
    nspack_hwy::delta_encode_i32(sy.data(), npts);
    nspack_hwy::delta_encode_i32(sz.data(), npts);
    nspack_hwy::delta_encode_i32(si.data(), npts);
#else
    for (int i = npts-1; i > 0; i--) { sx[i] -= sx[i-1]; sy[i] -= sy[i-1]; sz[i] -= sz[i-1]; si[i] -= si[i-1]; }
#endif
    std::vector<uint8_t> packed = pack_int16_zigzag(sx);
    std::vector<uint8_t> p2 = pack_int16_zigzag(sy);
    std::vector<uint8_t> p3 = pack_int16_zigzag(sz);
    std::vector<uint8_t> p4 = pack_int16_zigzag(si);
    packed.insert(packed.end(), p2.begin(), p2.end());
    packed.insert(packed.end(), p3.begin(), p3.end());
    packed.insert(packed.end(), p4.begin(), p4.end());
    size_t zb = ZSTD_compressBound(packed.size());
    std::vector<uint8_t> zdata(zb);
    size_t zs = ZSTD_compress(zdata.data(), zb, packed.data(), packed.size(), 19);
    if (ZSTD_isError(zs)) return false;
    std::ofstream out(outfile, std::ios::binary);
    out.write(PLY_MAGIC, 4);
    out.write((char*)&npts, 4);
    out.write((char*)&xmin, 4); out.write((char*)&xmax, 4);
    out.write((char*)&ymn, 4); out.write((char*)&ymx, 4);
    out.write((char*)&zmn, 4); out.write((char*)&zmx, 4);
    out.write((char*)&imn, 4); out.write((char*)&imx, 4);
    uint32_t zs32 = (uint32_t)zs;
    out.write((char*)&zs32, 4);
    out.write((char*)zdata.data(), zs);
    out.close();
    if (enc_ms) { auto t1 = Clock::now(); *enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(); }
    return true;
}

static bool decode_ply(const std::string& infile, const std::string& outfile, double* dec_ms = nullptr, double* shannon_T = nullptr) {
    auto t0 = Clock::now();
    std::ifstream f(infile, std::ios::binary);
    char magic[4]; f.read(magic, 4);
    if (memcmp(magic, PLY_MAGIC, 4) != 0) return false;
    int npts; f.read((char*)&npts, 4);
    float xmin, xmax, ymn, ymx, zmn, zmx, imn, imx;
    f.read((char*)&xmin, 4); f.read((char*)&xmax, 4);
    f.read((char*)&ymn, 4); f.read((char*)&ymx, 4);
    f.read((char*)&zmn, 4); f.read((char*)&zmx, 4);
    f.read((char*)&imn, 4); f.read((char*)&imx, 4);
    uint32_t zs; f.read((char*)&zs, 4);
    std::vector<uint8_t> zdata(zs); f.read((char*)zdata.data(), zs);
    unsigned long long psize = ZSTD_getFrameContentSize(zdata.data(), zs);
    if (psize == ZSTD_CONTENTSIZE_ERROR || psize == ZSTD_CONTENTSIZE_UNKNOWN) return false;
    std::vector<uint8_t> packed((size_t)psize);
    if (ZSTD_isError(ZSTD_decompress(packed.data(), packed.size(), zdata.data(), zs))) return false;
    if (shannon_T) *shannon_T = (double)shannon_limit_bytes(packed);
    // packed holds 4 concatenated pack_int16_zigzag streams (x, y, z, intensity).
    // Each stream carries its own mode byte; parse them sequentially.
    std::vector<int32_t> qx, qy, qz, qi;
    {
        size_t off = 0;
        std::vector<int32_t>* cols[4] = {&qx, &qy, &qz, &qi};
        for (int c = 0; c < 4; c++) {
            if (off >= packed.size()) return false;
            size_t nbytes = (packed[off] == 0) ? (size_t)npts * 2 : (size_t)npts * 4;
            if (off + 1 + nbytes > packed.size()) return false;
            std::vector<uint8_t> sub(packed.begin() + off, packed.begin() + off + 1 + nbytes);
            *cols[c] = unpack_int16_zigzag(sub);
            if (cols[c]->size() != (size_t)npts) return false;
            off += 1 + nbytes;
        }
    }
#if defined(NSPACK_HIGHWAY)
    nspack_hwy::delta_decode_i32(qx.data(), npts);
    nspack_hwy::delta_decode_i32(qy.data(), npts);
    nspack_hwy::delta_decode_i32(qz.data(), npts);
    nspack_hwy::delta_decode_i32(qi.data(), npts);
#else
    for (int i = 1; i < npts; i++) { qx[i] += qx[i-1]; qy[i] += qy[i-1]; qz[i] += qz[i-1]; qi[i] += qi[i-1]; }
#endif
    std::vector<float> x = dequantize_floats(qx, xmin, xmax);
    std::vector<float> y = dequantize_floats(qy, ymn, ymx);
    std::vector<float> z = dequantize_floats(qz, zmn, zmx);
    std::vector<float> it = dequantize_floats(qi, imn, imx);
    std::ofstream out(outfile);
    out << "ply\nformat ascii 1.0\nelement vertex " << npts << "\nproperty float x\nproperty float y\nproperty float z\nproperty float intensity\nend_header\n";
    for (int i = 0; i < npts; i++)
        out << x[i] << " " << y[i] << " " << z[i] << " " << it[i] << "\n";
    out.close();
    if (dec_ms) { auto t1 = Clock::now(); *dec_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(); }
    return true;
}


// ===== Scientific Data =====
static const char SCI_MAGIC[4] = {'N','S','S','1'};

struct SciHeader { uint8_t type; uint32_t rows; uint32_t cols; uint8_t dtype; };

static bool encode_scientific(const std::string& infile, const std::string& outfile, double* enc_ms = nullptr) {
    auto t0 = Clock::now();
    std::ifstream f(infile, std::ios::binary);
    if (!f) return false;
    SciHeader h;
    f.read((char*)&h.type, 1); f.read((char*)&h.rows, 4); f.read((char*)&h.cols, 4); f.read((char*)&h.dtype, 1);
    std::ofstream out(outfile, std::ios::binary);
    out.write(SCI_MAGIC, 4);
    out.write((char*)&h.type, 1); out.write((char*)&h.rows, 4); out.write((char*)&h.cols, 4); out.write((char*)&h.dtype, 1);
    std::vector<uint8_t> packed;
    if (h.type == 1) { // 2D dense float64 array
        int n = h.rows * h.cols;
        std::vector<double> d(n);
        f.read((char*)d.data(), n * 8);
        std::vector<float> vals(n);
        for (int i = 0; i < n; i++) vals[i] = (float)d[i];
        for (int r = 0; r < h.rows; r++) {
#if defined(NSPACK_HIGHWAY)
            nspack_hwy::delta_encode_f32(vals.data() + r*h.cols, h.cols);
#else
            for (int c = h.cols-1; c > 0; c--)
                vals[r*h.cols + c] -= vals[r*h.cols + c-1];
#endif
        }
        std::vector<float> col(n);
        for (int c = 0; c < h.cols; c++)
            for (int r = 0; r < h.rows; r++)
                col[c*h.rows + r] = vals[r*h.cols + c];
        float mn, mx;
        std::vector<int32_t> q = quantize_floats_i32(col, mn, mx);
        packed = pack_int16_zigzag(q);
        out.write((char*)&mn, 4); out.write((char*)&mx, 4);
    } else if (h.type == 2) { // time series float32
        int n = h.rows;
        std::vector<float> vals(n);
        f.read((char*)vals.data(), n * 4);
#if defined(NSPACK_HIGHWAY)
        nspack_hwy::delta_encode_f32(vals.data(), n);
#else
        for (int i = n-1; i > 0; i--) vals[i] -= vals[i-1];
#endif
        float mn, mx;
        std::vector<int16_t> q16 = quantize_floats(vals, mn, mx);
        std::vector<int32_t> q(q16.begin(), q16.end());
        packed = pack_int16_zigzag(q);
        out.write((char*)&mn, 4); out.write((char*)&mx, 4);
    } else if (h.type == 3) { // sparse (row, col, float32) sorted
        int nnz = h.rows;
        std::vector<uint32_t> rows(nnz), cols(nnz);
        std::vector<float> vals(nnz);
        for (int i = 0; i < nnz; i++) {
            f.read((char*)&rows[i], 4); f.read((char*)&cols[i], 4); f.read((char*)&vals[i], 4);
        }
#if defined(NSPACK_HIGHWAY)
        nspack_hwy::delta_encode_u32(rows.data(), nnz);
        nspack_hwy::delta_encode_u32(cols.data(), nnz);
#else
        for (int i = nnz-1; i > 0; i--) { rows[i] -= rows[i-1]; cols[i] -= cols[i-1]; }
#endif
        float mn, mx;
        std::vector<int16_t> q16 = quantize_floats(vals, mn, mx);
        std::vector<int32_t> q(q16.begin(), q16.end());
        out.write((char*)&mn, 4); out.write((char*)&mx, 4);
        // zigzag encode uint32 deltas
        for (int i = 0; i < nnz; i++) {
            uint32_t zr = (uint32_t)((int32_t)rows[i] << 1) ^ (uint32_t)((int32_t)rows[i] >> 31);
            uint32_t zc = (uint32_t)((int32_t)cols[i] << 1) ^ (uint32_t)((int32_t)cols[i] >> 31);
            packed.push_back(zr & 0xff); packed.push_back((zr >> 8) & 0xff); packed.push_back((zr >> 16) & 0xff); packed.push_back((zr >> 24) & 0xff);
            packed.push_back(zc & 0xff); packed.push_back((zc >> 8) & 0xff); packed.push_back((zc >> 16) & 0xff); packed.push_back((zc >> 24) & 0xff);
        }
        std::vector<uint8_t> pv = pack_int16_zigzag(q);
        packed.insert(packed.end(), pv.begin(), pv.end());
    } else {
        return false;
    }
    size_t zb = ZSTD_compressBound(packed.size());
    std::vector<uint8_t> zdata(zb);
    size_t zs = ZSTD_compress(zdata.data(), zb, packed.data(), packed.size(), 19);
    if (ZSTD_isError(zs)) return false;
    uint32_t zs32 = (uint32_t)zs;
    out.write((char*)&zs32, 4);
    out.write((char*)zdata.data(), zs);
    out.close();
    if (enc_ms) { auto t1 = Clock::now(); *enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(); }
    return true;
}

static bool decode_scientific(const std::string& infile, const std::string& outfile, double* dec_ms = nullptr, double* shannon_T = nullptr) {
    auto t0 = Clock::now();
    std::ifstream f(infile, std::ios::binary);
    char magic[4]; f.read(magic, 4);
    if (memcmp(magic, SCI_MAGIC, 4) != 0) return false;
    SciHeader h;
    f.read((char*)&h.type, 1); f.read((char*)&h.rows, 4); f.read((char*)&h.cols, 4); f.read((char*)&h.dtype, 1);
    float mn, mx; f.read((char*)&mn, 4); f.read((char*)&mx, 4);
    std::ofstream out(outfile, std::ios::binary);
    out.write((char*)&h.type, 1); out.write((char*)&h.rows, 4); out.write((char*)&h.cols, 4); out.write((char*)&h.dtype, 1);
    if (h.type == 1) {
        int n = h.rows * h.cols;
        uint32_t zs; f.read((char*)&zs, 4);
        std::vector<uint8_t> zdata(zs); f.read((char*)zdata.data(), zs);
        unsigned long long psize = ZSTD_getFrameContentSize(zdata.data(), zs);
        if (psize == ZSTD_CONTENTSIZE_ERROR || psize == ZSTD_CONTENTSIZE_UNKNOWN) return false;
        std::vector<uint8_t> packed((size_t)psize);
        if (ZSTD_isError(ZSTD_decompress(packed.data(), packed.size(), zdata.data(), zs))) return false;
    if (shannon_T) *shannon_T = (double)shannon_limit_bytes(packed);
        std::vector<int32_t> q = unpack_int16_zigzag(packed);
        // Encode delta-encoded along c (within each row) then stored column-major.
        // Reconstruct by accumulating the dequantized int32 deltas along c per row.
        double inv_range = ((double)mx - (double)mn) / 4294967295.0;
        for (int r = 0; r < h.rows; r++) {
            double acc = 0.0;
            for (int c = 0; c < h.cols; c++) {
                acc += (double)mn + (((double)q[c*h.rows + r] + 2147483648.0) * inv_range);
                double d = acc;
                out.write((char*)&d, 8);
            }
        }
    } else if (h.type == 2) {
        int n = h.rows;
        uint32_t zs; f.read((char*)&zs, 4);
        std::vector<uint8_t> zdata(zs); f.read((char*)zdata.data(), zs);
        unsigned long long psize = ZSTD_getFrameContentSize(zdata.data(), zs);
        if (psize == ZSTD_CONTENTSIZE_ERROR || psize == ZSTD_CONTENTSIZE_UNKNOWN) return false;
        std::vector<uint8_t> packed((size_t)psize);
        if (ZSTD_isError(ZSTD_decompress(packed.data(), packed.size(), zdata.data(), zs))) return false;
    if (shannon_T) *shannon_T = (double)shannon_limit_bytes(packed);
        std::vector<int32_t> q = unpack_int16_zigzag(packed);
#if defined(NSPACK_HIGHWAY)
        nspack_hwy::delta_decode_i32(q.data(), n);
#else
        for (int i = 1; i < n; i++) q[i] += q[i-1];
#endif
        for (int i = 0; i < n; i++) {
            float v = mn + ((q[i] + 32768.0f) / 65535.0f) * (mx - mn);
            out.write((char*)&v, 4);
        }
    } else if (h.type == 3) {
        int nnz = h.rows;
        uint32_t zs; f.read((char*)&zs, 4);
        std::vector<uint8_t> zdata(zs); f.read((char*)zdata.data(), zs);
        unsigned long long dsz = ZSTD_getFrameContentSize(zdata.data(), zs);
        std::vector<uint8_t> packed((size_t)dsz);
        ZSTD_decompress(packed.data(), packed.size(), zdata.data(), zs);
    if (shannon_T) *shannon_T = (double)shannon_limit_bytes(packed);
        size_t p = 0;
        std::vector<uint32_t> rows(nnz), cols(nnz);
        for (int i = 0; i < nnz; i++) {
            uint32_t zr = packed[p] | (packed[p+1] << 8) | (packed[p+2] << 16) | (packed[p+3] << 24); p += 4;
            uint32_t zc = packed[p] | (packed[p+1] << 8) | (packed[p+2] << 16) | (packed[p+3] << 24); p += 4;
            rows[i] = (zr >> 1) ^ (~(zr & 1) + 1);
            cols[i] = (zc >> 1) ^ (~(zc & 1) + 1);
        }
        std::vector<uint8_t> pv(packed.begin() + p, packed.end());
        std::vector<int32_t> q = unpack_int16_zigzag(pv);
#if defined(NSPACK_HIGHWAY)
        nspack_hwy::delta_decode_u32(rows.data(), nnz);
        nspack_hwy::delta_decode_u32(cols.data(), nnz);
#else
        for (int i = 1; i < nnz; i++) { rows[i] += rows[i-1]; cols[i] += cols[i-1]; }
#endif
        for (int i = 0; i < nnz; i++) {
            float v = mn + ((q[i] + 32768.0f) / 65535.0f) * (mx - mn);
            out.write((char*)&rows[i], 4); out.write((char*)&cols[i], 4); out.write((char*)&v, 4);
        }
    }
    out.close();
    if (dec_ms) { auto t1 = Clock::now(); *dec_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(); }
    return true;
}


// ===== HDR / EXR images (float32 RGBA) =====
static const char HDR_MAGIC[4] = {'N','S','H','1'};

static bool read_hdr_info(const std::string& infile, int& w, int& h) {
    std::string inf = infile + ".hdrinfo";
    std::ifstream f(inf);
    if (f) {
        f >> w >> h;
        return w > 0 && h > 0;
    }
    // No sidecar: derive dimensions from raw float32 RGBA file size.
    std::ifstream sf(infile, std::ios::binary | std::ios::ate);
    if (!sf) return false;
    uint64_t sz = (uint64_t)sf.tellg();
    if (sz % 16 != 0) return false;
    uint64_t n = sz / 16;
    if (n == 0) return false;
    // Pick the factor pair closest to square with w >= h.
    uint64_t best_w = n, best_h = 1, best_diff = n - 1;
    for (uint64_t h2 = 1; h2 * h2 <= n; h2++) {
        if (n % h2 == 0) {
            uint64_t w2 = n / h2;
            uint64_t diff = w2 - h2;
            if (diff < best_diff) { best_diff = diff; best_w = w2; best_h = h2; }
        }
    }
    w = (int)best_w; h = (int)best_h;
    return w > 0 && h > 0;
}

static bool encode_hdr(const std::string& infile, const std::string& outfile, double* enc_ms = nullptr) {
    auto t0 = Clock::now();
    int w, h;
    if (!read_hdr_info(infile, w, h)) { std::cerr << "HDR info read failed for " << infile << "\n"; return false; }
    size_t n = (size_t)w * h * 4;
    std::vector<float> rgba(n);
    std::ifstream f(infile, std::ios::binary);
    f.read((char*)rgba.data(), n * 4);
    float mn[4] = {1e30f, 1e30f, 1e30f, 1e30f};
    float mx[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    std::vector<float> logvals(n);
    for (size_t i = 0; i < n; i++) {
        float v = rgba[i];
        if (v <= 0) v = 1e-6f;
        float lv = logf(v);
        logvals[i] = lv;
        int c = i % 4;
        if (lv < mn[c]) mn[c] = lv;
        if (lv > mx[c]) mx[c] = lv;
    }
    std::vector<float> col(n);
    for (int c = 0; c < 4; c++)
        for (int i = 0; i < w * h; i++)
            col[c * w * h + i] = logvals[i * 4 + c];
    std::vector<int32_t> q(n);
    for (int c = 0; c < 4; c++) {
        float range = mx[c] - mn[c];
        if (range < 1e-10f) range = 1.0f;
        for (int i = 0; i < w * h; i++)
            q[c * w * h + i] = (int16_t)(((col[c * w * h + i] - mn[c]) / range) * 65535.0f - 32768.0f);
    }
    for (int c = 0; c < 4; c++)
        for (int i = w * h - 1; i > 0; i--)
            q[c * w * h + i] -= q[c * w * h + i - 1];
    std::vector<uint8_t> packed = pack_int16_zigzag(q);
    size_t zb = ZSTD_compressBound(packed.size());
    std::vector<uint8_t> zdata(zb);
    size_t zs = ZSTD_compress(zdata.data(), zb, packed.data(), packed.size(), 19);
    if (ZSTD_isError(zs)) return false;
    std::ofstream out(outfile, std::ios::binary);
    out.write(HDR_MAGIC, 4);
    out.write((char*)&w, 4); out.write((char*)&h, 4);
    out.write((char*)mn, 4 * 4); out.write((char*)mx, 4 * 4);
    uint32_t zs32 = (uint32_t)zs;
    out.write((char*)&zs32, 4);
    out.write((char*)zdata.data(), zs);
    out.close();
    if (enc_ms) { auto t1 = Clock::now(); *enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(); }
    return true;
}

static bool decode_hdr(const std::string& infile, const std::string& outfile, double* dec_ms = nullptr, double* shannon_T = nullptr) {
    auto t0 = Clock::now();
    std::ifstream f(infile, std::ios::binary);
    char magic[4]; f.read(magic, 4);
    if (memcmp(magic, HDR_MAGIC, 4) != 0) return false;
    int w, h; f.read((char*)&w, 4); f.read((char*)&h, 4);
    float mn[4], mx[4]; f.read((char*)mn, 16); f.read((char*)mx, 16);
    uint32_t zs; f.read((char*)&zs, 4);
    std::vector<uint8_t> zdata(zs); f.read((char*)zdata.data(), zs);
    unsigned long long psize = ZSTD_getFrameContentSize(zdata.data(), zs);
    if (psize == ZSTD_CONTENTSIZE_ERROR || psize == ZSTD_CONTENTSIZE_UNKNOWN) return false;
    std::vector<uint8_t> packed((size_t)psize);
    if (ZSTD_isError(ZSTD_decompress(packed.data(), packed.size(), zdata.data(), zs))) return false;
    if (shannon_T) *shannon_T = (double)shannon_limit_bytes(packed);
    std::vector<int32_t> q = unpack_int16_zigzag(packed);
    for (int c = 0; c < 4; c++) {
        float range = mx[c] - mn[c];
        if (range < 1e-10f) range = 1.0f;
        for (int i = 1; i < w * h; i++) q[c * w * h + i] += q[c * w * h + i - 1];
    }
    std::vector<float> rgba(w * h * 4);
    for (int c = 0; c < 4; c++) {
        float range = mx[c] - mn[c];
        if (range < 1e-10f) range = 1.0f;
        for (int i = 0; i < w * h; i++) {
            float lv = mn[c] + ((q[c * w * h + i] + 32768.0f) / 65535.0f) * range;
            rgba[i * 4 + c] = expf(lv);
        }
    }
    std::ofstream out(outfile, std::ios::binary);
    out.write((char*)rgba.data(), w * h * 4 * 4);
    out.close();
    if (dec_ms) { auto t1 = Clock::now(); *dec_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(); }
    return true;
}


// ===== SQLite Databases =====
static const char SQL_MAGIC[4] = {'N','S','Q','1'};

static bool encode_sqlite(const std::string& infile, const std::string& outfile, double* enc_ms = nullptr) {
    auto t0 = Clock::now();
    std::string schema = infile + ".schema";
    std::string data = infile + ".data.csv";
    std::string cmd1 = "sqlite3 '" + infile + "' '.schema' > '" + schema + "'";
    std::string cmd2 = "sqlite3 -header -csv '" + infile + "' 'SELECT * FROM data' > '" + data + "'";
    if (system(cmd1.c_str()) != 0 || system(cmd2.c_str()) != 0) return false;
    std::ifstream df(data);
    std::string line;
    std::getline(df, line);
    std::istringstream hss(line);
    std::vector<std::string> headers;
    std::string tok;
    while (std::getline(hss, tok, ',')) headers.push_back(tok);
    int ncols = headers.size();
    std::vector<std::vector<std::string>> cols(ncols);
    int nrows = 0;
    while (std::getline(df, line)) {
        std::istringstream ss(line);
        std::string val;
        int c = 0;
        while (std::getline(ss, val, ',')) {
            if (c < ncols) cols[c].push_back(val);
            c++;
        }
        nrows++;
    }
    std::remove(schema.c_str());
    std::remove(data.c_str());
    std::vector<int> ctype(ncols, 0);
    std::vector<float> cmin(ncols, 1e30f), cmax(ncols, -1e30f);
    for (int c = 0; c < ncols; c++) {
        bool all_int = true, all_float = true;
        for (const auto& s : cols[c]) {
            char* end;
            long l = strtol(s.c_str(), &end, 10);
            if (*end != 0) all_int = false;
            float f = strtof(s.c_str(), &end);
            if (*end != 0) all_float = false;
            if (all_float || all_int) {
                if (f < cmin[c]) cmin[c] = f;
                if (f > cmax[c]) cmax[c] = f;
            }
        }
        if (all_int) ctype[c] = 0;
        else if (all_float) ctype[c] = 1;
        else ctype[c] = 2;
    }
    std::vector<uint8_t> outdata;
    outdata.push_back(ncols & 0xff); outdata.push_back((ncols >> 8) & 0xff);
    for (int c = 0; c < ncols; c++) {
        outdata.push_back(ctype[c]);
        uint16_t hlen = headers[c].size(); outdata.push_back(hlen & 0xff); outdata.push_back((hlen >> 8) & 0xff);
        outdata.insert(outdata.end(), headers[c].begin(), headers[c].end());
    }
    for (int c = 0; c < ncols; c++) {
        if (ctype[c] == 2) {
            std::vector<std::string> dict;
            std::unordered_map<std::string, uint32_t> mp;
            std::vector<uint32_t> ids(nrows);
            for (int i = 0; i < nrows; i++) {
                auto it = mp.find(cols[c][i]);
                if (it == mp.end()) { uint32_t id = dict.size(); dict.push_back(cols[c][i]); mp[cols[c][i]] = id; ids[i] = id; }
                else ids[i] = it->second;
            }
            uint32_t nd = dict.size(); outdata.insert(outdata.end(), (uint8_t*)&nd, (uint8_t*)&nd + 4);
            for (auto& s : dict) { uint16_t l = s.size(); outdata.insert(outdata.end(), (uint8_t*)&l, (uint8_t*)&l + 2); outdata.insert(outdata.end(), s.begin(), s.end()); }
#if defined(NSPACK_HIGHWAY)
            nspack_hwy::delta_encode_u32(ids.data(), nrows);
#else
            for (int i = nrows-1; i > 0; i--) ids[i] -= ids[i-1];
#endif
            for (uint32_t v : ids) {
                uint32_t z = (uint32_t)((int32_t)v << 1) ^ (uint32_t)((int32_t)v >> 31);
                outdata.insert(outdata.end(), (uint8_t*)&z, (uint8_t*)&z + 4);
            }
        } else if (ctype[c] == 0) {
            // Integer column: exact int64 delta + zigzag (lossless, no quantization)
            std::vector<int64_t> iv(nrows);
            for (int i = 0; i < nrows; i++) iv[i] = strtoll(cols[c][i].c_str(), nullptr, 10);
            for (int i = nrows-1; i > 0; i--) iv[i] -= iv[i-1];
            for (int64_t v : iv) {
                uint64_t z = ((uint64_t)v << 1) ^ (uint64_t)(v >> 63);
                for (int b = 0; b < 8; b++) outdata.push_back((uint8_t)((z >> (b*8)) & 0xff));
            }
        } else {
            // Float column: int32 quantization for fine precision
            double range = (double)cmax[c] - (double)cmin[c]; if (range < 1e-12) range = 1.0;
            std::vector<int32_t> q(nrows);
            for (int i = 0; i < nrows; i++) {
                float f = strtof(cols[c][i].c_str(), nullptr);
                double t = ((double)f - (double)cmin[c]) / range;
                int64_t qi = (int64_t)std::llround(t * 4294967295.0 - 2147483648.0);
                if (qi < -2147483648LL) qi = -2147483648LL;
                if (qi > 2147483647LL) qi = 2147483647LL;
                q[i] = (int32_t)qi;
            }
            for (int i = nrows-1; i > 0; i--) q[i] -= q[i-1];
            std::vector<uint8_t> p = pack_int16_zigzag(q);
            outdata.insert(outdata.end(), p.begin(), p.end());
            outdata.insert(outdata.end(), (uint8_t*)&cmin[c], (uint8_t*)&cmin[c] + 4);
            outdata.insert(outdata.end(), (uint8_t*)&cmax[c], (uint8_t*)&cmax[c] + 4);
        }
    }
    size_t zb = ZSTD_compressBound(outdata.size());
    std::vector<uint8_t> zdata(zb);
    size_t zs = ZSTD_compress(zdata.data(), zb, outdata.data(), outdata.size(), 19);
    if (ZSTD_isError(zs)) return false;
    std::ofstream out(outfile, std::ios::binary);
    out.write(SQL_MAGIC, 4);
    out.write((char*)&nrows, 4);
    uint32_t zs32 = (uint32_t)zs;
    out.write((char*)&zs32, 4);
    out.write((char*)zdata.data(), zs);
    out.close();
    if (enc_ms) { auto t1 = Clock::now(); *enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(); }
    return true;
}

static bool decode_sqlite(const std::string& infile, const std::string& outfile, double* dec_ms = nullptr, double* shannon_T = nullptr) {
auto t0 = Clock::now();
    std::ifstream f(infile, std::ios::binary);
    char magic[4]; f.read(magic, 4);
    if (memcmp(magic, SQL_MAGIC, 4) != 0) return false;
    int nrows; f.read((char*)&nrows, 4);
    uint32_t zs; f.read((char*)&zs, 4);
    std::vector<uint8_t> zdata(zs); f.read((char*)zdata.data(), zs);
    unsigned long long dsz = ZSTD_getFrameContentSize(zdata.data(), zs);
    if (dsz == ZSTD_CONTENTSIZE_ERROR || dsz == ZSTD_CONTENTSIZE_UNKNOWN) { std::cerr << "Unknown decompressed size\n"; return false; }
    std::vector<uint8_t> raw((size_t)dsz);
    size_t ds = ZSTD_decompress(raw.data(), raw.size(), zdata.data(), zs);
    if (shannon_T) *shannon_T = (double)shannon_limit_bytes(raw);
    if (ZSTD_isError(ds)) return false;
    raw.resize(ds);
    size_t p = 0;
    auto r8 = [&]() -> uint8_t { if (p >= raw.size()) { std::cerr << "SQLite decode out of bounds at p=" << p << " size=" << raw.size() << "\n"; throw std::runtime_error("oob"); } return raw[p++]; };
    uint16_t ncols = r8() | (r8() << 8);
    std::vector<int> ctype(ncols);
    std::vector<std::string> headers(ncols);
    for (int c = 0; c < ncols; c++) {
        ctype[c] = r8();
        uint16_t hlen = r8() | (r8() << 8);
        headers[c].assign(raw.begin() + p, raw.begin() + p + hlen); p += hlen;
    }
    std::vector<std::vector<std::string>> cols(ncols, std::vector<std::string>(nrows));
    for (int c = 0; c < ncols; c++) {
        if (ctype[c] == 2) {
            uint32_t nd = *(uint32_t*)(raw.data() + p); p += 4;
            std::vector<std::string> dict(nd);
            for (uint32_t i = 0; i < nd; i++) {
                uint16_t l = *(uint16_t*)(raw.data() + p); p += 2;
                dict[i].assign(raw.begin() + p, raw.begin() + p + l); p += l;
            }
            std::vector<uint32_t> ids(nrows);
            for (int i = 0; i < nrows; i++) { uint32_t z = *(uint32_t*)(raw.data() + p); p += 4; ids[i] = (z >> 1) ^ (~(z & 1) + 1); }
#if defined(NSPACK_HIGHWAY)
            nspack_hwy::delta_decode_u32(ids.data(), nrows);
#else
            for (int i = 1; i < nrows; i++) ids[i] += ids[i-1];
#endif
            for (int i = 0; i < nrows; i++) cols[c][i] = dict[ids[i]];
        } else if (ctype[c] == 0) {
            // Integer column: exact int64 delta + zigzag
            if (p + (size_t)nrows * 8 > raw.size()) return false;
            std::vector<int64_t> iv(nrows);
            for (int i = 0; i < nrows; i++) {
                uint64_t z = 0;
                for (int b = 0; b < 8; b++) z |= (uint64_t)raw[p++] << (b*8);
                iv[i] = (int64_t)((z >> 1) ^ (~(z & 1) + 1));
            }
            for (int i = 1; i < nrows; i++) iv[i] += iv[i-1];
            for (int i = 0; i < nrows; i++) cols[c][i] = std::to_string(iv[i]);
        } else {
            if (p >= raw.size()) return false;
            size_t nbytes = (raw[p] == 0) ? (size_t)nrows * 2 : (size_t)nrows * 4;
            if (p + 1 + nbytes > raw.size()) return false;
            std::vector<uint8_t> packed(raw.data() + p, raw.data() + p + 1 + nbytes); p += 1 + nbytes;
            std::vector<int32_t> q = unpack_int16_zigzag(packed);
            if (q.size() != (size_t)nrows) return false;
            for (int i = 1; i < nrows; i++) q[i] += q[i-1];
            float cmin = *(float*)(raw.data() + p); p += 4;
            float cmax = *(float*)(raw.data() + p); p += 4;
            double range = (double)cmax - (double)cmin; if (range < 1e-12) range = 1.0;
            for (int i = 0; i < nrows; i++) {
                double v = (double)cmin + (((double)q[i] + 2147483648.0) / 4294967295.0) * range;
                cols[c][i] = std::to_string(v);
            }
        }
    }
    std::ofstream out(outfile);
    for (int c = 0; c < ncols; c++) { out << headers[c]; if (c < ncols-1) out << ","; }
    out << "\n";
    for (int i = 0; i < nrows; i++) {
        for (int c = 0; c < ncols; c++) { out << cols[c][i]; if (c < ncols-1) out << ","; }
        out << "\n";
    }
    out.close();
    if (dec_ms) { auto t1 = Clock::now(); *dec_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(); }
    return true;
}


// ===== ELF Binary Executables =====
static const char ELF_MAGIC[4] = {'N','S','E','1'};

struct Elf64_Hdr {
    uint8_t e_ident[16];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint64_t e_entry;
    uint64_t e_phoff;
    uint64_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
};

struct Elf64_Shdr {
    uint32_t sh_name;
    uint32_t sh_type;
    uint64_t sh_flags;
    uint64_t sh_addr;
    uint64_t sh_offset;
    uint64_t sh_size;
    uint32_t sh_link;
    uint32_t sh_info;
    uint64_t sh_addralign;
    uint64_t sh_entsize;
};

static bool encode_elf(const std::string& infile, const std::string& outfile, double* enc_ms = nullptr) {
    auto t0 = Clock::now();
    std::ifstream f(infile, std::ios::binary);
    if (!f) return false;
    std::vector<uint8_t> file((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (file.size() < 64 || file[0] != 0x7f || file[1] != 'E' || file[2] != 'L' || file[3] != 'F') {
        std::cerr << "Not an ELF file\n"; return false;
    }
    Elf64_Hdr* h = (Elf64_Hdr*)file.data();
    if (h->e_ident[4] != 2) { std::cerr << "Not 64-bit ELF\n"; return false; }
    Elf64_Shdr* shdrs = (Elf64_Shdr*)(file.data() + h->e_shoff);
    const char* shstr = (char*)file.data() + shdrs[h->e_shstrndx].sh_offset;
    std::vector<uint8_t> outbuf;
    outbuf.insert(outbuf.end(), (uint8_t*)h, (uint8_t*)h + 64);
    uint32_t nsec = h->e_shnum;
    outbuf.insert(outbuf.end(), (uint8_t*)&nsec, (uint8_t*)&nsec + 4);
    for (uint16_t i = 0; i < h->e_shnum; i++) {
        std::string name(shstr + shdrs[i].sh_name);
        const char* data = (char*)file.data() + shdrs[i].sh_offset;
        size_t sz = shdrs[i].sh_size;
        bool is_data = (name == ".data" || name == ".rodata" || name == ".data.rel.ro");
        bool is_text = (name == ".text");
        uint8_t flags = (is_data ? 1 : is_text ? 2 : 0);
        outbuf.push_back(flags);
        uint16_t nlen = name.size(); outbuf.insert(outbuf.end(), (uint8_t*)&nlen, (uint8_t*)&nlen + 2);
        outbuf.insert(outbuf.end(), name.begin(), name.end());
        uint64_t ssz = sz; outbuf.insert(outbuf.end(), (uint8_t*)&ssz, (uint8_t*)&ssz + 8);
        if ((is_data || is_text) && sz > 0) {
            std::vector<uint8_t> d(data, data + sz);
            if (is_data)
                for (size_t j = sz-1; j > 0; j--) d[j] -= d[j-1];
            size_t zb = ZSTD_compressBound(sz);
            std::vector<uint8_t> zd(zb);
            size_t zs = ZSTD_compress(zd.data(), zb, d.data(), sz, 19);
            uint32_t zsz = (uint32_t)zs;
            outbuf.insert(outbuf.end(), (uint8_t*)&zsz, (uint8_t*)&zsz + 4);
            outbuf.insert(outbuf.end(), zd.data(), zd.data() + zs);
        } else {
            outbuf.insert(outbuf.end(), data, data + sz);
        }
    }
    size_t zb = ZSTD_compressBound(outbuf.size());
    std::vector<uint8_t> zd(zb);
    size_t zs = ZSTD_compress(zd.data(), zb, outbuf.data(), outbuf.size(), 19);
    if (ZSTD_isError(zs)) return false;
    std::ofstream out(outfile, std::ios::binary);
    out.write(ELF_MAGIC, 4);
    uint32_t zs32 = (uint32_t)zs;
    out.write((char*)&zs32, 4);
    out.write((char*)zd.data(), zs);
    out.close();
    if (enc_ms) { auto t1 = Clock::now(); *enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(); }
    return true;
}

static bool decode_elf(const std::string& infile, const std::string& outfile, double* dec_ms = nullptr, double* shannon_T = nullptr) {
    auto t0 = Clock::now();
    std::ifstream f(infile, std::ios::binary);
    char magic[4]; f.read(magic, 4);
    if (memcmp(magic, ELF_MAGIC, 4) != 0) return false;
    uint32_t zs; f.read((char*)&zs, 4);
    std::vector<uint8_t> zdata(zs); f.read((char*)zdata.data(), zs);
    unsigned long long dsz = ZSTD_getFrameContentSize(zdata.data(), zs);
    if (dsz == ZSTD_CONTENTSIZE_ERROR || dsz == ZSTD_CONTENTSIZE_UNKNOWN) { std::cerr << "Unknown decompressed size\n"; return false; }
    std::vector<uint8_t> raw((size_t)dsz);
    size_t ds = ZSTD_decompress(raw.data(), raw.size(), zdata.data(), zs);
    if (shannon_T) *shannon_T = (double)shannon_limit_bytes(raw);
    if (ZSTD_isError(ds)) return false;
    raw.resize(ds);
    std::ofstream out(outfile, std::ios::binary);
    out.write((char*)raw.data(), raw.size());
    out.close();
    if (dec_ms) { auto t1 = Clock::now(); *dec_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(); }
    return true;
}

