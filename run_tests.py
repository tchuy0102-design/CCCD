#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_tests.py - Tu tao anh thu (nen khac nhau, xoay, nghieng, mo, toi, loa...) tu the mau,
chay cccd_scanner va do do chinh xac so voi dap an dung (4 goc that + the dung chieu).

    python run_tests.py            # chay 20 truong hop, ghi ket qua vao results/synthetic
"""
import math
import os
import sys
import json

import cv2
import numpy as np

from cccd_scanner import CARD_ASPECT, imread, imwrite, scan_card, draw_debug

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "synthetic")
os.makedirs(OUT, exist_ok=True)

REF = imread(os.path.join(HERE, "reference_card.png"))
REF_GRAY = cv2.cvtColor(cv2.resize(REF, (449, 283)), cv2.COLOR_BGR2GRAY).astype(np.float32)


# ------------------------------------------------------------------ nen
def _smooth(h, w, scale, rng):
    small = rng.random((max(2, h // scale + 2), max(2, w // scale + 2))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _lerp(c0, c1, t):
    c0, c1 = np.array(c0, np.float32), np.array(c1, np.float32)
    return c0[None, None] + (c1 - c0)[None, None] * t[:, :, None]


def bg_wood(W, H, rng):
    n1 = _smooth(H, W, 140, rng)
    streak = cv2.resize(rng.random((H, 7)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    streak = cv2.GaussianBlur(streak, (0, 0), 1.2)
    rings = 0.5 + 0.5 * np.sin(np.arange(H, dtype=np.float32)[:, None] / 7.0 + 9 * n1)
    t = np.clip(0.45 * streak + 0.55 * rings, 0, 1)
    img = _lerp((48, 88, 135), (98, 152, 205), t)
    return img + rng.normal(0, 3, img.shape).astype(np.float32)


def bg_white(W, H, rng):
    img = np.full((H, W, 3), 232, np.float32) + _smooth(H, W, 220, rng)[:, :, None] * 10
    return img + rng.normal(0, 2.5, img.shape).astype(np.float32)


def bg_dark(W, H, rng):
    img = np.full((H, W, 3), 30, np.float32) + _smooth(H, W, 160, rng)[:, :, None] * 14
    return img + rng.normal(0, 2.5, img.shape).astype(np.float32)


def bg_green(W, H, rng):
    t = _smooth(H, W, 90, rng)
    img = _lerp((55, 98, 60), (80, 125, 85), t)
    return img + rng.normal(0, 4, img.shape).astype(np.float32)


def bg_marble(W, H, rng):
    n = _smooth(H, W, 200, rng) * 0.6 + _smooth(H, W, 60, rng) * 0.3 + _smooth(H, W, 20, rng) * 0.1
    veins = np.abs(np.sin(n * 22)) ** 6
    g = 150 + 55 * n - 50 * veins
    return np.repeat(g[:, :, None], 3, 2).astype(np.float32)


def bg_clutter(W, H, rng):
    img = np.full((H, W, 3), 205, np.uint8)
    for _ in range(28):
        col = tuple(int(v) for v in rng.integers(30, 250, 3))
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        k = rng.integers(0, 4)
        if k == 0:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(40, 260)), y + int(rng.integers(30, 200))), col, -1)
        elif k == 1:
            cv2.circle(img, (x, y), int(rng.integers(15, 90)), col, -1)
        elif k == 2:
            cv2.line(img, (x, y), (int(rng.integers(0, W)), int(rng.integers(0, H))), col, int(rng.integers(2, 9)))
        else:
            cv2.putText(img, "ABC 123 hoa don", (x, y), cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(.6, 1.8)), col, 2)
    img = img.astype(np.float32)
    return img + rng.normal(0, 3, img.shape).astype(np.float32)


def bg_bluesim(W, H, rng):
    t = _smooth(H, W, 120, rng)
    img = _lerp((205, 178, 120), (228, 200, 150), t)       # nen xanh nhat gan mau the
    return img + rng.normal(0, 3, img.shape).astype(np.float32)


def bg_jeans(W, H, rng):
    xs = np.arange(W, dtype=np.float32)[None]
    ys = np.arange(H, dtype=np.float32)[:, None]
    t = 0.5 + 0.5 * np.sin((xs + ys) / 2.2)
    t = 0.7 * t + 0.3 * _smooth(H, W, 40, rng)
    img = _lerp((95, 55, 30), (140, 90, 55), t)
    return img + rng.normal(0, 5, img.shape).astype(np.float32)


def bg_keys(W, H, rng):
    img = np.full((H, W, 3), 75, np.uint8)
    s = 70
    for y in range(8, H, s):
        for x in range(8, W, s):
            cv2.rectangle(img, (x, y), (x + s - 12, y + s - 12), (150, 150, 150), -1)
    img = img.astype(np.float32)
    return img + rng.normal(0, 3, img.shape).astype(np.float32)


BG = dict(wood=bg_wood, white=bg_white, dark=bg_dark, green=bg_green, marble=bg_marble,
          clutter=bg_clutter, bluesim=bg_bluesim, jeans=bg_jeans, keys=bg_keys)


# ------------------------------------------------------------------ the
def card_rgba(width):
    h0, w0 = REF.shape[:2]
    c = REF[1:h0 - 1, 1:w0 - 1]
    h0, w0 = c.shape[:2]
    S = 4
    mask = np.zeros((h0 * S, w0 * S), np.uint8)
    r = int(0.055 * h0 * S)
    cv2.rectangle(mask, (r, 0), (w0 * S - r, h0 * S), 255, -1)
    cv2.rectangle(mask, (0, r), (w0 * S, h0 * S - r), 255, -1)
    for cx, cy in ((r, r), (w0 * S - r, r), (r, h0 * S - r), (w0 * S - r, h0 * S - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1)
    mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_AREA)
    rgba = np.dstack([c, mask]).astype(np.float32)
    nh = int(round(width / CARD_ASPECT))
    return cv2.resize(rgba, (int(round(width)), nh), interpolation=cv2.INTER_AREA if width < w0 else cv2.INTER_CUBIC)


CASES = [
    # name, bg, canvas, scale, rot, tilt, extras
    dict(name="01_go_thang",            bg="wood",    canvas=(1280, 960),  scale=0.70, rot=0,    tilt=0.02),
    dict(name="02_go_xoay15",           bg="wood",    canvas=(1280, 960),  scale=0.70, rot=15,   tilt=0.04),
    dict(name="03_go_xoay90",           bg="wood",    canvas=(1280, 960),  scale=0.68, rot=90,   tilt=0.05),
    dict(name="04_go_xoay180",          bg="wood",    canvas=(1280, 960),  scale=0.70, rot=180,  tilt=0.04),
    dict(name="05_go_xoay270_nghieng",  bg="wood",    canvas=(1280, 960),  scale=0.66, rot=270,  tilt=0.10),
    dict(name="06_ban_trang_nghieng",   bg="white",   canvas=(1280, 960),  scale=0.68, rot=-7,   tilt=0.07, shadow=0.35),
    dict(name="07_ban_toi_xoay-25",     bg="dark",    canvas=(1280, 960),  scale=0.66, rot=-25,  tilt=0.05),
    dict(name="08_tham_xanh_xoay40",    bg="green",   canvas=(1280, 960),  scale=0.62, rot=40,   tilt=0.08),
    dict(name="09_da_hoa_nghieng_manh", bg="marble",  canvas=(1280, 960),  scale=0.66, rot=10,   tilt=0.14),
    dict(name="10_do_vat_lon_xon",      bg="clutter", canvas=(1280, 960),  scale=0.58, rot=-12,  tilt=0.06),
    dict(name="11_nen_xanh_gan_mau_the", bg="bluesim", canvas=(1280, 960), scale=0.66, rot=8,    tilt=0.05, hard=True),
    dict(name="12_jeans_mo_nhieu",      bg="jeans",   canvas=(1280, 960),  scale=0.62, rot=30,   tilt=0.06, blur=1.6, noise=7),
    dict(name="13_the_nho",             bg="wood",    canvas=(1280, 960),  scale=0.34, rot=20,   tilt=0.05),
    dict(name="14_the_gan_kin_khung",   bg="wood",    canvas=(1280, 960),  scale=0.93, rot=3,    tilt=0.02),
    dict(name="15_loa_va_bong",         bg="white",   canvas=(1280, 960),  scale=0.66, rot=-8,   tilt=0.06, glare=True, shadow=0.45),
    dict(name="16_thieu_sang_xoay180",  bg="dark",    canvas=(1280, 960),  scale=0.64, rot=192,  tilt=0.05, gamma=1.8, noise=10, jpeg=35, light=0.35),
    dict(name="17_dien_thoai_doc_xoay90", bg="wood",  canvas=(960, 1280),  scale=0.88, rot=90,   tilt=0.07),
    dict(name="18_ban_phim_nghieng",    bg="keys",    canvas=(1280, 960),  scale=0.62, rot=-18,  tilt=0.08),
    dict(name="19_rung_tay_mo_huong",   bg="wood",    canvas=(1280, 960),  scale=0.68, rot=12,   tilt=0.05, blur=2.6, motion=9),
    dict(name="20_ngon_tay_che_goc",    bg="wood",    canvas=(1280, 960),  scale=0.68, rot=-10,  tilt=0.05, finger=True, hard=True),
]


def make_case(spec, seed):
    rng = np.random.default_rng(seed)
    W, H = spec["canvas"]
    bg = BG[spec["bg"]](W, H, rng).astype(np.float32)
    cw = spec["scale"] * min(W, H * CARD_ASPECT)
    ch = cw / CARD_ASPECT
    rect = np.array([[-cw / 2, -ch / 2], [cw / 2, -ch / 2], [cw / 2, ch / 2], [-cw / 2, ch / 2]], np.float32)
    rect += rng.uniform(-1, 1, (4, 2)).astype(np.float32) * spec["tilt"] * ch
    th = math.radians(spec["rot"])
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], np.float32)
    rect = rect @ R.T
    dx, dy = rng.uniform(-0.03, 0.03) * W, rng.uniform(-0.03, 0.03) * H
    quad = rect + np.array([W / 2 + dx, H / 2 + dy], np.float32)
    # dam bao nam tron trong khung (tru khi thuc su muon cat)
    for _ in range(40):
        if quad[:, 0].min() >= 0.012 * W and quad[:, 0].max() <= 0.988 * W and \
           quad[:, 1].min() >= 0.012 * H and quad[:, 1].max() <= 0.988 * H:
            break
        c = quad.mean(0)
        quad = c + (quad - c) * 0.97
        quad += (np.array([W / 2, H / 2], np.float32) - c) * 0.1

    card = card_rgba(max(200, np.linalg.norm(quad[1] - quad[0])))
    h0, w0 = card.shape[:2]
    src = np.array([[0, 0], [w0 - 1, 0], [w0 - 1, h0 - 1], [0, h0 - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    warped = cv2.warpPerspective(card, M, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    alpha = warped[:, :, 3:4] / 255.0
    rgb = warped[:, :, :3]
    if spec.get("glare"):
        gx, gy = quad.mean(0) + np.array([0.12 * W, -0.05 * H])
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        g = np.exp(-(((xx - gx) / (0.14 * W)) ** 2 + ((yy - gy) / (0.07 * H)) ** 2))[:, :, None]
        rgb = rgb + (255 - rgb) * np.clip(g * 0.85, 0, 1)
    sh = spec.get("shadow", 0.28)
    smask = np.zeros((H, W), np.float32)
    cv2.fillPoly(smask, [(quad + np.array([7, 11])).astype(np.int32)], 1.0)
    smask = cv2.GaussianBlur(smask, (0, 0), 10)[:, :, None]
    bg = bg * (1 - sh * smask)
    img = bg * (1 - alpha) + rgb * alpha

    if spec.get("finger"):
        c = quad[1] + (quad[1] - quad.mean(0)) * 0.0
        axes = (int(0.07 * W), int(0.045 * W))
        cv2.ellipse(img, (int(c[0] + 0.01 * W), int(c[1] - 0.01 * W)), axes, 35, 0, 360, (115, 150, 205), -1)
        img = cv2.GaussianBlur(img, (0, 0), 0.8)

    L = spec.get("light", 0.18)
    ang = rng.uniform(0, 2 * math.pi)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    grad = ((xx - W / 2) * math.cos(ang) + (yy - H / 2) * math.sin(ang)) / math.hypot(W, H)
    img = img * (1 + L * 2 * grad)[:, :, None]
    if spec.get("gamma"):
        img = 255 * np.power(np.clip(img, 0, 255) / 255, spec["gamma"])
    if spec.get("motion"):
        k = int(spec["motion"])
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0 / k
        img = cv2.filter2D(img, -1, ker)
    if spec.get("blur"):
        img = cv2.GaussianBlur(img, (0, 0), spec["blur"])
    img = img + rng.normal(0, spec.get("noise", 2.5), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(spec.get("jpeg", 82))])
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img, quad


# ------------------------------------------------------------------ do dac
def iou_quads(a, b, shape):
    H, W = shape
    ma = np.zeros((H, W), np.uint8)
    mb = np.zeros((H, W), np.uint8)
    cv2.fillPoly(ma, [np.round(a).astype(np.int32)], 1)
    cv2.fillPoly(mb, [np.round(b).astype(np.int32)], 1)
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    return float(inter / max(1, union))


def zncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def compare_to_ref(card):
    """Do giong the mau (cho phep lech toi da ~8px): tra ve (dung chieu, nguoc 180)."""
    g = cv2.cvtColor(cv2.resize(card, (449, 283), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    r = cv2.GaussianBlur(REF_GRAY, (0, 0), 1.0)
    out = []
    for ref in (r, np.ascontiguousarray(np.rot90(r, 2))):
        tmpl = ref[12:-12, 20:-20]
        res = cv2.matchTemplate(g, tmpl, cv2.TM_CCOEFF_NORMED)
        out.append(float(res.max()))
    return out[0], out[1]


def thumb(img, w, h):
    s = min(w / img.shape[1], h / img.shape[0])
    t = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), 245, np.uint8)
    y0, x0 = (h - t.shape[0]) // 2, (w - t.shape[1]) // 2
    canvas[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
    return canvas


def label(img, text, ok):
    out = img.copy()
    col = (60, 150, 60) if ok else (40, 40, 200)
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 24), col, -1)
    cv2.putText(out, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    rows = []
    panels = []
    for i, spec in enumerate(CASES):
        img, gt = make_case(spec, seed=1000 + i)
        imwrite(os.path.join(OUT, spec["name"] + "_input.jpg"), img)
        res = scan_card(img)
        H, W = img.shape[:2]
        row = dict(case=spec["name"], hard=bool(spec.get("hard")), found=res["found"], iou=0.0,
                   zncc=0.0, zncc_180=0.0, upright=False, passed=False, score=res["score"], cardlike=res.get("cardlike"))
        shown = draw_debug(img, res)
        # ve dap an dung bang mau vang (de doi chieu)
        cv2.polylines(shown, [np.round(gt).astype(np.int32).reshape(-1, 1, 2)], True, (0, 220, 255), 1)
        if res["found"]:
            row["iou"] = iou_quads(res["quad"], gt, (H, W))
            z, z180 = compare_to_ref(res["card"])
            row.update(zncc=z, zncc_180=z180, upright=bool(z > z180))
            row["passed"] = bool(row["iou"] >= 0.93 and z >= 0.75 and z > z180)
            imwrite(os.path.join(OUT, spec["name"] + "_output.jpg"), res["card"])
            out_t = thumb(res["card"], 300, 190)
        else:
            out_t = thumb(np.full((190, 300, 3), 200, np.uint8), 300, 190)
        rows.append(row)
        cap = f"{spec['name'][:2]} IoU={row['iou']:.3f} ZNCC={row['zncc']:.2f}"
        panel = np.vstack([label(thumb(shown, 300, 225), cap, row["passed"]), out_t])
        panel = cv2.copyMakeBorder(panel, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        panels.append(panel)

    # anh that cua nguoi dung (khong co dap an goc, chi so voi anh mau thu 2)
    real = imread(os.path.join(HERE, "test_cases", "real_user_photo_wood.png"))
    rr = scan_card(real)
    z, z180 = compare_to_ref(rr["card"]) if rr["found"] else (0, 0)
    rows.append(dict(case="REAL_anh_that_cua_ban", hard=False, found=rr["found"], iou=None, zncc=z, zncc_180=z180,
                     upright=bool(z > z180), passed=bool(rr["found"] and z >= 0.75 and z > z180), score=rr["score"], cardlike=rr.get("cardlike")))

    # contact sheets
    per = 10
    for si in range(0, len(panels), per):
        chunk = panels[si:si + per]
        while len(chunk) % 5:
            chunk.append(np.full_like(chunk[0], 255))
        grid = [np.hstack(chunk[j:j + 5]) for j in range(0, len(chunk), 5)]
        imwrite(os.path.join(HERE, "results", f"contact_sheet_{si // per + 1}.jpg"), np.vstack(grid), 92)

    with open(os.path.join(HERE, "results", "report.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    n_pass = sum(r["passed"] for r in rows)
    print(f"{'case':34s} found  IoU    ZNCC  dungchieu  OK")
    for r in rows:
        iou = "  -  " if r["iou"] is None else f"{r['iou']:.3f}"
        print(f"{r['case']:34s} {str(r['found']):5s}  {iou}  {r['zncc']:.2f}  {str(r['upright']):8s}  {'PASS' if r['passed'] else 'FAIL'}{'  (kho)' if r['hard'] else ''}  canh_bao={'khong' if r.get('cardlike') else 'CO'}")
    print(f"\nDat: {n_pass}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
