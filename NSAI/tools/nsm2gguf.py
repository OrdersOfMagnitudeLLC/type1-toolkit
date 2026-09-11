#!/usr/bin/env python3
"""NSM2-to-GGUF converter.

Reads an NSM2 file, decompresses per-cluster data, unpacks ns_q4_Kx8 panels
back to standard block_q4_K, and writes a valid GGUF file loadable by ik_llama.
"""

import struct
import sys
import zlib
import os
import numpy as np

CLUSTER_SIZE = 256
PANEL_SIZE = 8 * CLUSTER_SIZE
BLOCK_Q4_K = 144
ALIGN = 32

GGUF_SCALAR_SIZES = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}

# Only norm weights and biases must be F32 for ik_llama's RMS norm handler
F32_SUFFIXES = ('attn_norm.weight', 'ffn_norm.weight', 'output_norm.weight',
                '.bias')

def should_be_f32(name):
    return any(name.endswith(s) for s in F32_SUFFIXES)

def read_exact(f, n):
    data = b''
    while len(data) < n:
        chunk = f.read(n - len(data))
        if not chunk:
            raise EOFError(f"Expected {n} bytes, got {len(data)}")
        data += chunk
    return data

def zlib_decompress(data):
    if not data:
        return b''
    d = zlib.decompressobj()
    return d.decompress(data) + d.flush()

def gguf_type_size(gguf_type, dims):
    n = 1
    for d in dims:
        n *= d
    if gguf_type == 0: return n * 4
    elif gguf_type == 1: return n * 2
    elif gguf_type == 8: return (n // 32) * 34
    elif gguf_type == 12: return (n // 256) * BLOCK_Q4_K
    elif gguf_type == 14: return (n // 256) * 210
    else:
        print(f"Unknown GGUF type {gguf_type}")
        sys.exit(1)

def unpack_q4_k_panel(panel):
    """Unpack ns_q4_Kx8 (1152 bytes) to 8 standard block_q4_K (144 bytes each).

    ns_q4_Kx8 layout:
      d[8]      : 8 x uint16 super-block scales
      dmin[8]   : 8 x uint16 super-block mins
      scales[96]: 8 groups of 12 bytes, each packing 6-bit scale/min for 8 rows
      qs[1024]  : interleaved quantized values (blck_size_interleave=8)
    """
    assert len(panel) == 1152
    blocks = [bytearray(144) for _ in range(8)]

    # Extract d and dmin
    for i in range(8):
        d = struct.unpack_from('<H', panel, i * 2)[0]
        dmin = struct.unpack_from('<H', panel, 16 + i * 2)[0]
        struct.pack_into('<H', blocks[i], 0, d)
        struct.pack_into('<H', blocks[i], 2, dmin)

    # Unpack scales: 8 groups of 12 bytes
    # Groups 0-3: sub-blocks 0-3, Groups 4-7: sub-blocks 4-7
    sc = [[0]*8 for _ in range(8)]  # sc[sub_block][row]
    mn = [[0]*8 for _ in range(8)]  # mn[sub_block][row]

    for g in range(8):
        base = 32 + g * 12  # scales start at offset 32 (after d[8] + dmin[8])
        b = panel[base:base+12]
        s = [0]*8
        m = [0]*8
        s[0] = b[0] & 0x3F
        s[1] = b[1] & 0x3F
        s[2] = b[2] & 0x3F
        s[3] = b[3] & 0x3F
        s[4] = ((b[0] >> 6) << 4) | (b[8] & 0x0F)
        s[5] = ((b[1] >> 6) << 4) | (b[9] & 0x0F)
        s[6] = ((b[2] >> 6) << 4) | (b[10] & 0x0F)
        s[7] = ((b[3] >> 6) << 4) | (b[11] & 0x0F)
        m[0] = b[4] & 0x3F
        m[1] = b[5] & 0x3F
        m[2] = b[6] & 0x3F
        m[3] = b[7] & 0x3F
        m[4] = ((b[4] >> 6) << 4) | (b[8] >> 4)
        m[5] = ((b[5] >> 6) << 4) | (b[9] >> 4)
        m[6] = ((b[6] >> 6) << 4) | (b[10] >> 4)
        m[7] = ((b[7] >> 6) << 4) | (b[11] >> 4)

        for j in range(8):
            sc[g][j] = s[j]
            mn[g][j] = m[j]

    # Reconstruct scales[12] for each row
    for j in range(8):
        scales = bytearray(12)
        # Sub-blocks 0-3: scales[0..3] = sc0..sc3, scales[4..7] = mn0..mn3
        # Sub-blocks 4-7: packed into scales[8..11] + high bits in scales[0..3] and scales[4..7]
        for i in range(4):
            scales[i] = sc[i][j] | ((sc[i+4][j] >> 4) << 6)
            scales[i+4] = mn[i][j] | ((mn[i+4][j] >> 4) << 6)
        scales[8] = (sc[4][j] & 0x0F) | ((mn[4][j] & 0x0F) << 4)
        scales[9] = (sc[5][j] & 0x0F) | ((mn[5][j] & 0x0F) << 4)
        scales[10] = (sc[6][j] & 0x0F) | ((mn[6][j] & 0x0F) << 4)
        scales[11] = (sc[7][j] & 0x0F) | ((mn[7][j] & 0x0F) << 4)
        blocks[j][4:16] = scales

    # Unpack qs: blck_size_interleave=8
    # Packing: for i in 0..127: memcpy(out.qs[i*8], in[src_id].qs[(i/8)*8], 8)
    # where src_id = i % 8
    qs_data = panel[128:]  # qs starts at offset 128 (32 + 96)
    assert len(qs_data) == 1024
    for j in range(8):
        blocks[j][16:144] = b'\x00' * 128

    for i in range(128):
        src_id = i % 8
        src_offset = (i // 8) * 8
        dst_offset = i * 8
        chunk = qs_data[dst_offset:dst_offset+8]
        blocks[src_id][16+src_offset:16+src_offset+8] = chunk

    return [bytes(b) for b in blocks]

def dequantize_q4_k_block(block_data):
    """Dequantize a single Q4_K block (144 bytes) to 256 float32 values."""
    d = np.float32(struct.unpack_from('<e', block_data, 0)[0])
    dmin = np.float32(struct.unpack_from('<e', block_data, 2)[0])
    scales = block_data[4:16]
    qs = block_data[16:144]

    out = np.zeros(256, dtype=np.float32)

    def get_sc_min(j):
        if j < 4:
            return scales[j] & 63, scales[j+4] & 63
        else:
            sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
            mn = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
            return sc, mn

    q_idx = 0
    for j in range(0, 256, 64):
        sc, mn = get_sc_min(j // 64 * 2)
        d1 = d * sc
        m1 = dmin * mn
        sc, mn = get_sc_min(j // 64 * 2 + 1)
        d2 = d * sc
        m2 = dmin * mn

        for l in range(32):
            out[j + l] = d1 * (qs[q_idx + l] & 0x0F) - m1
        for l in range(32):
            out[j + 32 + l] = d2 * (qs[q_idx + l] >> 4) - m2
        q_idx += 32

    return out

def dequantize_q4_k(data, n_elements):
    """Dequantize Q4_K packed data to F32 numpy array."""
    n_blocks = (n_elements + 255) // 256
    out = np.zeros(n_blocks * 256, dtype=np.float32)
    for b in range(n_blocks):
        block_data = data[b * 144:(b + 1) * 144]
        if len(block_data) < 144:
            break
        dequant = dequantize_q4_k_block(block_data)
        out[b * 256:b * 256 + 256] = dequant
    return out[:n_elements]

def parse_gguf_kv(buf, offset):
    key_len = struct.unpack_from('<Q', buf, offset)[0]; offset += 8
    key = buf[offset:offset+key_len].decode('utf-8'); offset += key_len
    val_type = struct.unpack_from('<I', buf, offset)[0]; offset += 4
    val_start = offset
    if val_type == 8:
        slen = struct.unpack_from('<Q', buf, offset)[0]; offset += 8 + slen
    elif val_type == 9:
        arr_type = struct.unpack_from('<I', buf, offset)[0]; offset += 4
        arr_len = struct.unpack_from('<Q', buf, offset)[0]; offset += 8
        for _ in range(arr_len):
            if arr_type == 8:
                slen = struct.unpack_from('<Q', buf, offset)[0]; offset += 8 + slen
            elif arr_type in GGUF_SCALAR_SIZES:
                offset += GGUF_SCALAR_SIZES[arr_type]
            else:
                print(f"Unknown array element type {arr_type}"); sys.exit(1)
    elif val_type in GGUF_SCALAR_SIZES:
        offset += GGUF_SCALAR_SIZES[val_type]
    else:
        print(f"Unknown GGUF value type {val_type}"); sys.exit(1)
    return key, val_type, buf[val_start:offset], offset

def write_gguf_kv(out, key, val_type, val_bytes):
    out += struct.pack('<Q', len(key))
    out += key.encode('utf-8')
    out += struct.pack('<I', val_type)
    out += val_bytes
    return out

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.nsm> <output.gguf>")
        sys.exit(1)

    nsm_path = sys.argv[1]
    out_path = sys.argv[2]

    with open(nsm_path, 'rb') as f:
        # --- NSM header ---
        hdr_fmt = '<IIIIQQQQ'
        hdr_data = read_exact(f, struct.calcsize(hdr_fmt))
        magic, version, n_layers, n_tensors, data_offset, _, gguf_meta_off, gguf_meta_bytes = struct.unpack(hdr_fmt, hdr_data)
        print(f"NSM: version={version}, n_tensors={n_tensors}, n_layers={n_layers}")
        if magic != 0x314D534E or version != 2:
            print("ERROR: bad NSM header"); sys.exit(1)

        # --- NSM tensor entries ---
        tensor_fmt = '<64sIIQQ'
        tensor_size = struct.calcsize(tensor_fmt)
        nsm_tensors = []
        for i in range(n_tensors):
            data = read_exact(f, tensor_size)
            name_raw, n_clusters, quant_type, tdata_off, tdata_bytes = struct.unpack(tensor_fmt, data)
            name = name_raw.split(b'\x00')[0].decode('utf-8')
            nsm_tensors.append({'name': name, 'n_clusters': n_clusters, 'quant_type': quant_type,
                               'data_offset': tdata_off, 'data_bytes': tdata_bytes})

        # --- Cluster maps ---
        cluster_size = 24
        for t in nsm_tensors:
            t['clusters'] = []
            for _ in range(t['n_clusters']):
                data = read_exact(f, cluster_size)
                t['clusters'].append({
                    'quant_bits': data[4],
                    'data_offset': struct.unpack_from('<I', data, 8)[0],
                    'data_bytes': struct.unpack_from('<I', data, 12)[0],
                    'scale': struct.unpack_from('<f', data, 16)[0],
                    'compressed': data[20],
                })

        # --- Embedded GGUF metadata ---
        f.seek(gguf_meta_off)
        gguf_buf = read_exact(f, gguf_meta_bytes)

        g_off = 0
        gguf_magic = struct.unpack_from('<I', gguf_buf, g_off)[0]; g_off += 4
        gguf_version = struct.unpack_from('<I', gguf_buf, g_off)[0]; g_off += 4
        tensor_count = struct.unpack_from('<Q', gguf_buf, g_off)[0]; g_off += 8
        kv_count = struct.unpack_from('<Q', gguf_buf, g_off)[0]; g_off += 8
        print(f"GGUF: version={gguf_version}, tensors={tensor_count}, kv={kv_count}")

        kv_pairs = []
        for _ in range(kv_count):
            key, val_type, val_bytes, g_off = parse_gguf_kv(gguf_buf, g_off)
            kv_pairs.append((key, val_type, val_bytes))

        gguf_tensors = []
        for i in range(tensor_count):
            name_len = struct.unpack_from('<Q', gguf_buf, g_off)[0]; g_off += 8
            tname = gguf_buf[g_off:g_off+name_len].decode('utf-8'); g_off += name_len
            n_dims = struct.unpack_from('<I', gguf_buf, g_off)[0]; g_off += 4
            dims = []
            for _ in range(n_dims):
                dims.append(struct.unpack_from('<Q', gguf_buf, g_off)[0]); g_off += 8
            ttype = struct.unpack_from('<I', gguf_buf, g_off)[0]; g_off += 4
            toff = struct.unpack_from('<Q', gguf_buf, g_off)[0]; g_off += 8
            gguf_tensors.append({'name': tname, 'n_dims': n_dims, 'dims': dims, 'type': ttype, 'offset': toff})

        print(f"Parsed {len(kv_pairs)} KV pairs, {len(gguf_tensors)} tensors")

        nsm_map = {t['name']: t for t in nsm_tensors}

        # --- Override types and recompute offsets ---
        new_offsets = {}
        current_offset = 0
        for gt in gguf_tensors:
            tname = gt['name']
            nsm_t = nsm_map.get(tname)
            if nsm_t:
                if should_be_f32(tname) and nsm_t['quant_type'] == 12:
                    gt['type'] = 0  # F32 (dequantize from Q4_K)
                else:
                    gt['type'] = nsm_t['quant_type']  # Keep NSM type (F16 or Q4_K)
            raw_size = gguf_type_size(gt['type'], gt['dims'])
            current_offset = (current_offset + ALIGN - 1) // ALIGN * ALIGN
            new_offsets[tname] = current_offset
            current_offset += raw_size

        # --- Decompress and unpack tensors ---
        print("Decompressing tensors...")
        tensor_data = {}

        for i, gt in enumerate(gguf_tensors):
            tname = gt['name']
            nsm_t = nsm_map.get(tname)
            if not nsm_t:
                continue

            qt = nsm_t['quant_type']
            nc = nsm_t['n_clusters']
            n = 1
            for d in gt['dims']:
                n *= d

            is_2d = len(gt['dims']) >= 2
            n_cols = gt['dims'][0] if is_2d else n
            n_rows = gt['dims'][1] if is_2d else 1
            full_blocks = (n + CLUSTER_SIZE - 1) // CLUSTER_SIZE
            n_blocks_per_row = (n_cols + CLUSTER_SIZE - 1) // CLUSTER_SIZE
            n_row_groups = (n_rows + 7) // 8
            n_panels = n_row_groups * n_blocks_per_row

            use_panels = (nc == n_panels and n_panels != full_blocks)
            is_f32 = should_be_f32(tname) and qt == 12
            raw_size = gguf_type_size(gt['type'], gt['dims'])

            buf = bytearray(raw_size)

            if nc == 1 and nsm_t['clusters'][0]['data_bytes'] == nsm_t['data_bytes']:
                # Passthrough
                cl = nsm_t['clusters'][0]
                f.seek(nsm_t['data_offset'] + cl['data_offset'])
                raw = read_exact(f, cl['data_bytes'])
                if cl['compressed']:
                    raw = zlib_decompress(raw)
                buf[:min(len(raw), raw_size)] = raw[:raw_size]
            else:
                # Per-cluster
                for c in range(nc):
                    cl = nsm_t['clusters'][c]
                    f.seek(nsm_t['data_offset'] + cl['data_offset'])
                    packed = read_exact(f, cl['data_bytes'])
                    if cl['compressed']:
                        raw = zlib_decompress(packed)
                    else:
                        raw = packed

                    if use_panels:
                        # Unpack ns_q4_Kx8 panel to 8 standard Q4_K blocks
                        blocks = unpack_q4_k_panel(raw)
                        row_group = c // n_blocks_per_row
                        col_block = c % n_blocks_per_row
                        for r in range(8):
                            actual_row = row_group * 8 + r
                            if actual_row >= n_rows:
                                break
                            block_idx = actual_row * n_blocks_per_row + col_block
                            off = block_idx * BLOCK_Q4_K
                            if off + BLOCK_Q4_K <= raw_size:
                                buf[off:off+BLOCK_Q4_K] = blocks[r]
                    else:
                        # Non-panel: direct copy
                        if qt == 0 or qt == 1:
                            elem_size = 4 if qt == 0 else 2
                            cluster_len = min(CLUSTER_SIZE, n - c * CLUSTER_SIZE)
                            expected = cluster_len * elem_size
                            if len(raw) >= expected:
                                off = c * CLUSTER_SIZE * elem_size
                                buf[off:off+expected] = raw[:expected]
                        else:
                            off = c * BLOCK_Q4_K
                            copy_len = min(BLOCK_Q4_K, raw_size - off)
                            if copy_len > 0 and len(raw) >= copy_len:
                                buf[off:off+copy_len] = raw[:copy_len]

            # Dequantize F32 tensors from Q4_K
            if is_f32 and qt == 12:
                f32_data = dequantize_q4_k(bytes(buf), n)
                tensor_data[tname] = f32_data.tobytes()
            else:
                tensor_data[tname] = bytes(buf)

            if tname == 'blk.0.attn_q.weight':
                import struct as _s
                _d = _s.unpack_from('<e', buf, 0)[0]
                print(f"  DEBUG: {tname} buf[0] d={_d}, scales={buf[4:16].hex()}, use_panels={use_panels}, nc={nc}, n_panels={n_panels}, n_cols={n_cols}, n_rows={n_rows}")
                _td = tensor_data[tname]
                _d2 = _s.unpack_from('<e', _td, 0)[0]
                print(f"  DEBUG: tensor_data d={_d2}, scales={_td[4:16].hex()}, len={len(_td)}")

            if (i + 1) % 50 == 0 or i == len(gguf_tensors) - 1:
                print(f"  [{i+1}/{len(gguf_tensors)}] {tname} ({raw_size} bytes, {'panel' if use_panels else 'block'}, {'F32' if is_f32 else 'Q4K'})")

        # --- Write GGUF ---
        print(f"Writing {out_path}...")
        header = bytearray()
        header += struct.pack('<I', 0x46554747)
        header += struct.pack('<I', gguf_version)
        header += struct.pack('<Q', tensor_count)
        header += struct.pack('<Q', kv_count)
        for key, val_type, val_bytes in kv_pairs:
            header = write_gguf_kv(header, key, val_type, val_bytes)
        for gt in gguf_tensors:
            tname = gt['name']
            header += struct.pack('<Q', len(tname))
            header += tname.encode('utf-8')
            header += struct.pack('<I', gt['n_dims'])
            for d in gt['dims']:
                header += struct.pack('<Q', d)
            header += struct.pack('<I', gt['type'])
            header += struct.pack('<Q', new_offsets[tname])

        while len(header) % ALIGN != 0:
            header += b'\x00'

        with open(out_path, 'wb') as out:
            out.write(header)
            current_pos = 0
            for gt in gguf_tensors:
                tname = gt['name']
                tdata = tensor_data.get(tname)
                if tdata is None:
                    raw_size = gguf_type_size(gt['type'], gt['dims'])
                    tdata = b'\x00' * raw_size
                target = new_offsets[tname]
                if current_pos < target:
                    out.write(b'\x00' * (target - current_pos))
                    current_pos = target
                out.write(tdata)
                current_pos += len(tdata)

        out_size = os.path.getsize(out_path)
        in_size = os.path.getsize(nsm_path)
        type_counts = {}
        for gt in gguf_tensors:
            type_counts[gt['type']] = type_counts.get(gt['type'], 0) + 1
        print(f"\nConversion complete:")
        print(f"  NSM input:    {in_size / 1e9:.2f} GB")
        print(f"  GGUF output:  {out_size / 1e9:.2f} GB")
        print(f"  Type distribution: {type_counts}")

if __name__ == '__main__':
    main()
