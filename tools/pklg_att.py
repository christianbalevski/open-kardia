#!/usr/bin/env python
"""Parse an Apple PacketLogger .pklg capture and print the ATT conversation
(reads/writes/notifications/indications) — no Wireshark needed.

Usage:
  python tools/pklg_att.py capture.pklg [--handle 0x12] [--out captures/app_session.jsonl]

The JSONL output uses the same shape as `open_kardia sniff` ({t, uuid, len, hex}) with
uuid = "h<handle>" (or the resolved UUID when a Find-Information/Read-By-Type response
is present in the trace), so `open_kardia analyze` can score it directly. Writes are logged
as {"t", "dir": "write", "handle", "hex"} lines so the start sequence can be replayed.

pklg record: u32 BE length (of the rest), u32 BE ts_sec, u32 BE ts_usec, u8 type, payload
  type 0x00 HCI command, 0x01 HCI event, 0x02 ACL host->ctrl (sent), 0x03 ACL ctrl->host (recv)
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

ATT_OPS = {
    0x01: "ERROR", 0x02: "MTU_REQ", 0x03: "MTU_RSP", 0x04: "FIND_INFO_REQ", 0x05: "FIND_INFO_RSP",
    0x06: "FIND_BY_TYPE_REQ", 0x07: "FIND_BY_TYPE_RSP", 0x08: "READ_BY_TYPE_REQ", 0x09: "READ_BY_TYPE_RSP",
    0x0a: "READ_REQ", 0x0b: "READ_RSP", 0x0c: "READ_BLOB_REQ", 0x0d: "READ_BLOB_RSP",
    0x10: "READ_BY_GROUP_REQ", 0x11: "READ_BY_GROUP_RSP", 0x12: "WRITE_REQ", 0x13: "WRITE_RSP",
    0x16: "PREP_WRITE_REQ", 0x17: "PREP_WRITE_RSP", 0x18: "EXEC_WRITE_REQ", 0x19: "EXEC_WRITE_RSP",
    0x1b: "NOTIFY", 0x1d: "INDICATE", 0x1e: "CONFIRM", 0x52: "WRITE_CMD", 0xd2: "SIGNED_WRITE",
}


def iter_pklg(path):
    data = Path(path).read_bytes()
    off = 0
    n = len(data)
    # Two variants exist: big-endian (classic) and little-endian (newer PacketLogger).
    ln_be = struct.unpack(">I", data[:4])[0]
    ln_le = struct.unpack("<I", data[:4])[0]
    fmt = ">IIIB" if 9 <= ln_be <= 0x10000 and not (9 <= ln_le <= 0x10000 and ln_le < ln_be) else "<IIIB"
    while off + 13 <= n:
        ln, sec, usec, typ = struct.unpack(fmt, data[off: off + 13])
        if ln < 9 or off + 4 + ln > n:
            break
        payload = data[off + 13: off + 4 + ln]
        off += 4 + ln
        yield sec + usec / 1e6, typ, payload


def uuid_str(b: bytes) -> str:
    if len(b) == 2:
        return f"0000{b[::-1].hex()}-0000-1000-8000-00805f9b34fb"
    if len(b) == 16:
        h = b[::-1].hex()
        return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"
    return b.hex()


def parse(path, only_handle=None, out=None, quiet=False, silent=False):
    frag = {}                       # (conn, dir) -> (expected_len, buffer)
    handle_uuid: dict[int, str] = {}
    counts = defaultdict(int)
    t0 = None
    fh = open(out, "w") if out else None
    for t, typ, p in iter_pklg(path):
        if typ not in (2, 3) or len(p) < 4:
            continue
        hdr, alen = struct.unpack("<HH", p[:4])
        conn = hdr & 0x0FFF
        pb = (hdr >> 12) & 0x3
        acl = p[4: 4 + alen]
        key = (conn, typ)
        if pb in (0, 2):            # first fragment
            if len(acl) < 4:
                continue
            l2len, cid = struct.unpack("<HH", acl[:4])
            frag[key] = [l2len, cid, bytearray(acl[4:])]
        elif pb == 1 and key in frag:
            frag[key][2] += acl
        else:
            continue
        l2len, cid, buf = frag[key]
        if len(buf) < l2len:
            continue
        del frag[key]
        if cid != 4 or not buf:       # ATT only
            continue
        att = bytes(buf[:l2len])
        op = att[0]
        name = ATT_OPS.get(op, f"OP{op:02x}")
        direction = "->" if typ == 2 else "<-"
        if t0 is None:
            t0 = t
        rel = t - t0
        counts[name] += 1
        # learn handle -> uuid from discovery responses
        if op == 0x05 and len(att) >= 2:            # FIND_INFO_RSP
            fmt = att[1]; sz = 4 if fmt == 1 else 18
            for i in range(2, len(att) - sz + 1, sz):
                h = struct.unpack("<H", att[i:i + 2])[0]
                handle_uuid[h] = uuid_str(att[i + 2:i + sz])
        elif op == 0x09 and len(att) >= 2:          # READ_BY_TYPE_RSP (char declarations)
            sz = att[1]
            for i in range(2, len(att) - sz + 1, sz):
                rec = att[i:i + sz]
                if sz in (7, 21):                    # decl: props(1) value_handle(2) uuid
                    vh = struct.unpack("<H", rec[3:5])[0]
                    handle_uuid[vh] = uuid_str(rec[5:])
        line = None
        if op in (0x12, 0x52, 0x1b, 0x1d, 0x0a, 0x0b, 0x13):
            if op in (0x12, 0x52, 0x1b, 0x1d, 0x0a):
                h = struct.unpack("<H", att[1:3])[0]
                val = att[3:]
            else:
                h, val = None, att[1:]
            if only_handle is not None and h is not None and h != only_handle:
                continue
            hu = handle_uuid.get(h, f"h{h:#06x}" if h is not None else "?")
            line = f"{rel:9.3f} {direction} {name:14s} h={h:#06x} {hu[:36]:36s} len={len(val):3d} {val.hex()}" if h is not None \
                else f"{rel:9.3f} {direction} {name:14s} len={len(val):3d} {val.hex()}"
            if fh:
                if op in (0x1b, 0x1d):
                    fh.write(json.dumps({"t": t, "uuid": hu, "handle": h, "len": len(val), "hex": val.hex(),
                                         "kind": name.lower()}) + "\n")
                elif op in (0x12, 0x52):
                    fh.write(json.dumps({"t": t, "dir": "write", "uuid": hu, "handle": h, "hex": val.hex(),
                                         "kind": name.lower()}) + "\n")
                elif op == 0x0b:
                    fh.write(json.dumps({"t": t, "dir": "read_rsp", "hex": val.hex()}) + "\n")
        elif not quiet:
            line = f"{rel:9.3f} {direction} {name:14s} {att[1:].hex()}"
        if line and not silent:
            print(line)
    if not silent:
        print("\nATT opcode counts:", dict(counts), file=sys.stderr)
    if handle_uuid and not silent:
        print("handle -> uuid map:", file=sys.stderr)
        for h in sorted(handle_uuid):
            print(f"  {h:#06x} {handle_uuid[h]}", file=sys.stderr)
    if fh:
        fh.close()
        print(f"wrote {out}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pklg")
    ap.add_argument("--handle", type=lambda s: int(s, 0))
    ap.add_argument("--out")
    ap.add_argument("--quiet", action="store_true", help="only data ops")
    a = ap.parse_args()
    parse(a.pklg, a.handle, a.out, a.quiet)


if __name__ == "__main__":
    main()
