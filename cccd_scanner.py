#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cccd_scanner.py - Tu dong nhan dien, cat va nan thang anh Can cuoc cong dan (CCCD).

Cach dung nhanh (Windows / Linux / macOS):
    pip install opencv-python numpy

    python cccd_scanner.py anh1.jpg anh2.png          # xu ly vai file
    python cccd_scanner.py thu_muc_anh                # xu ly ca thu muc
    python cccd_scanner.py anh.jpg -o ket_qua --debug # luu them anh danh dau 4 goc

    Keo-tha file/thu muc len file cccd_scanner.py cung chay duoc.
    Khong truyen gi: tu xu ly thu muc "input" canh file nay -> ket qua o "output".

Ket qua: <ten_anh>_cccd.jpg  (the nam ngang, nan thang, cat gon nhu anh mau)

Dung trong code:
    from cccd_scanner import scan_card, imread, imwrite
    res = scan_card(imread("anh.jpg"))
    if res["found"]:
        imwrite("out.jpg", res["card"])
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import numpy as np

CARD_ASPECT = 85.6 / 53.98          # ti le the CCCD that (~1.586)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MIN_SCORE = 0.32                     # duoi nguong nay coi la khong tim thay the


# --------------------------------------------------------------------------
# Doc / ghi anh (ho tro duong dan tieng Viet tren Windows)
# --------------------------------------------------------------------------
def imread(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path, img, quality=95):
    ext = os.path.splitext(path)[1].lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError("Khong ghi duoc anh: " + path)
    buf.tofile(path)


# --------------------------------------------------------------------------
# Hinh hoc
# --------------------------------------------------------------------------
def poly_area(q):
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def order_quad(pts):
    """Sap xep 4 diem theo chieu kim dong ho, bat dau tu goc tren-trai."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    i = int(np.argmin(pts.sum(1)))
    return np.roll(pts, -i, axis=0)


def interior_angles(q):
    out = []
    for i in range(4):
        a = q[i - 1] - q[i]
        b = q[(i + 1) % 4] - q[i]
        cs = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        out.append(math.degrees(math.acos(max(-1.0, min(1.0, cs)))))
    return out


def side_lengths(q):
    return [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]


def _intersect(l1, l2):
    p1, v1 = l1
    p2, v2 = l2
    A = np.array([[v1[0], -v2[0]], [v1[1], -v2[1]]], np.float64)
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t = np.linalg.solve(A, (p2 - p1).astype(np.float64))
    return (p1 + v1 * t[0]).astype(np.float32)


# --------------------------------------------------------------------------
# Du lieu canh (de tinh chinh 4 goc cho that chinh xac)
# --------------------------------------------------------------------------
class _EdgeData:
    def __init__(self, gray):
        g = cv2.GaussianBlur(gray, (5, 5), 0)
        med = float(np.median(g))
        lo = int(max(10, 0.66 * med))
        hi = int(min(250, max(60, 1.33 * med)))
        self.edges = cv2.Canny(g, lo, hi)
        gf = g.astype(np.float32)
        gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
        ys, xs = np.nonzero(self.edges)
        self.P = np.stack([xs, ys], 1).astype(np.float32)
        G = np.stack([gx[ys, xs], gy[ys, xs]], 1)
        self.G = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-6)


def refine_quad(q, ed, diag):
    """Khop lai 4 canh bang cac diem canh that (loai nhieu), roi giao 4 duong thang."""
    q = q.astype(np.float32).copy()
    if len(ed.P) < 50:
        return q
    for band_frac in (0.045, 0.02, 0.008):
        band = band_frac * diag
        centroid = q.mean(0)
        lines = []
        for i in range(4):
            a, b = q[i], q[(i + 1) % 4]
            d = b - a
            L = float(np.linalg.norm(d))
            if L < 1:
                lines.append(None)
                continue
            u = d / L
            n = np.array([-u[1], u[0]], np.float32)
            if np.dot(n, (a + b) / 2 - centroid) < 0:
                n = -n                                 # n huong ra ngoai the
            rel = ed.P - a
            t = rel @ u / L
            dist = rel @ n                              # duong = nam ngoai canh hien tai
            m = (np.abs(dist) < band) & (t > 0.06) & (t < 0.94)
            if m.sum() < 12:
                lines.append(None)
                continue
            ok = np.abs(ed.G[m] @ n) > 0.86          # huong gradient vuong goc voi canh
            pts = ed.P[m][ok]
            dd = dist[m][ok]
            if len(pts) < 12:
                lines.append(None)
                continue
            # Lay "dinh" ngoai cung: mep that cua the, khong phai hoa van/chu ben trong
            bins = np.arange(-band, band + 1.5, 1.5)
            hist, _ = np.histogram(dd, bins)
            hs = np.convolve(hist.astype(np.float32), np.array([1, 2, 1], np.float32) / 4, "same")
            thr = max(6.0, 0.45 * float(hs.max()))
            pick = None
            for k in range(len(hs) - 1, -1, -1):
                left = hs[k - 1] if k > 0 else -1
                right = hs[k + 1] if k + 1 < len(hs) else -1
                if hs[k] >= thr and hs[k] >= left and hs[k] >= right:
                    pick = k
                    break
            if pick is not None:
                center = 0.5 * (bins[pick] + bins[pick + 1])
                sel = np.abs(dd - center) < 3.0
                if sel.sum() >= 12:
                    pts = pts[sel]
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            lines.append((np.array([x0, y0], np.float32), np.array([vx, vy], np.float32)))
        newq = q.copy()
        for i in range(4):
            l1, l2 = lines[(i - 1) % 4], lines[i]
            if l1 is None or l2 is None:
                continue
            p = _intersect(l1, l2)
            if p is not None and np.linalg.norm(p - q[i]) < 0.07 * diag:
                newq[i] = p
        q = newq
    return q


def _sample_lab(lab_f, pts):
    H, W = lab_f.shape[:2]
    ok = (pts[:, 0] >= 0) & (pts[:, 0] <= W - 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= H - 1)
    xi = np.clip(pts[:, 0], 0, W - 1).astype(np.float32).reshape(1, -1)
    yi = np.clip(pts[:, 1], 0, H - 1).astype(np.float32).reshape(1, -1)
    return cv2.remap(lab_f, xi, yi, cv2.INTER_LINEAR)[0], ok


def polish_quad(q, lab_f, diag):
    """Dich/nghieng tung canh de do tuong phan mau giua trong-ngoai canh lon nhat
    (= mep that cua the), roi giao 4 duong thang thanh 4 goc."""
    q = q.astype(np.float32).copy()
    t = np.linspace(0.1, 0.9, 60, dtype=np.float32)[:, None]
    stages = ((0.09 * diag, 3.0, 4.0, (0.0,)),
              (0.025 * diag, 1.0, 3.0, (0.0,)),
              (6.0, 0.5, 2.5, (0.0,)),
              (2.5, 0.5, 2.0, (-1.2, -0.6, 0.0, 0.6, 1.2)))
    for B, step, d, tilts in stages:
        centroid = q.mean(0)
        lines = []
        for i in range(4):
            a, b = q[i], q[(i + 1) % 4]
            e = b - a
            L = float(np.linalg.norm(e))
            if L < 1:
                lines.append(None)
                continue
            u = e / L
            n = np.array([-u[1], u[0]], np.float32)
            if np.dot(n, (a + b) / 2 - centroid) < 0:
                n = -n
            mid = (a + b) / 2
            base0 = a + e * t
            offs = np.arange(-B, B + 0.01, step)
            per_tilt = []
            gmax = 0.0
            for tilt in tilts:
                th = math.radians(tilt)
                R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], np.float32)
                base = mid + (base0 - mid) @ R.T
                nn = n @ R.T
                vals = np.full(len(offs), -1.0, np.float32)
                for k, off in enumerate(offs):
                    p = base + nn * off
                    vi, oi = _sample_lab(lab_f, p - nn * d)
                    vo, oo = _sample_lab(lab_f, p + nn * d)
                    ok = oi & oo
                    if ok.mean() < 0.5:
                        continue
                    vals[k] = float(np.percentile(np.linalg.norm(vi[ok] - vo[ok], axis=1), 25))
                per_tilt.append((tilt, vals))
                gmax = max(gmax, float(vals.max()))
            # Lay mep NGOAI CUNG du manh (>=75% mep manh nhat): tranh dinh vao hoa van/chu ben trong the
            best = (-1e9, 0.0, 0.0)
            for tilt, vals in per_tilt:
                vs = np.convolve(vals, np.array([1, 2, 1], np.float32) / 4, "same")
                pick = None
                for k in range(len(vs) - 1, -1, -1):
                    lft = vs[k - 1] if k > 0 else -1
                    rgt = vs[k + 1] if k + 1 < len(vs) else -1
                    if vs[k] >= 0.75 * gmax and vs[k] >= lft and vs[k] >= rgt:
                        pick = k
                        break
                if pick is None:
                    continue
                val = float(vs[pick]) - 0.05 * abs(tilt)
                if val > best[0]:
                    best = (val, float(offs[pick]), tilt)
            _, off, tilt = best
            th = math.radians(tilt)
            R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], np.float32)
            nn = n @ R.T
            uu = u @ R.T
            lines.append((mid + nn * off, uu))
        newq = q.copy()
        for i in range(4):
            l1, l2 = lines[(i - 1) % 4], lines[i]
            if l1 is None or l2 is None:
                continue
            p = _intersect(l1, l2)
            if p is not None and np.linalg.norm(p - q[i]) < 0.2 * diag:
                newq[i] = p
        q = newq
    return q


def _edge_support(a, b, lab_f, d=2.5):
    t = np.linspace(0.1, 0.9, 60, dtype=np.float32)[:, None]
    e = b - a
    n = np.array([-e[1], e[0]], np.float32) / (np.linalg.norm(e) + 1e-9)
    p = a + e * t
    vi, oi = _sample_lab(lab_f, p - n * d)
    vo, oo = _sample_lab(lab_f, p + n * d)
    ok = oi & oo
    if ok.mean() < 0.5:
        return 0.0
    return float(np.percentile(np.linalg.norm(vi[ok] - vo[ok], axis=1), 25))


def fix_aspect(q, lab_f, tol=0.05):
    """Neu ti le khung lech xa ti le the that (1.586), thu day MOT canh ra ngoai cho dung ti le;
    chi nhan neu o vi tri moi that su co mep (tuong phan ro)."""
    q = order_quad(q).copy()
    sl = side_lengths(q)
    A, B = (sl[0] + sl[2]) / 2, (sl[1] + sl[3]) / 2
    if A >= B:
        L, S, long_e, short_e = A, B, (0, 2), (1, 3)
    else:
        L, S, long_e, short_e = B, A, (1, 3), (0, 2)
    if abs(math.log(L / S / CARD_ASPECT)) < math.log(1 + tol):
        return q
    if L / S > CARD_ASPECT:
        need, edges = L / CARD_ASPECT - S, long_e
    else:
        need, edges = S * CARD_ASPECT - L, short_e
    centroid = q.mean(0)
    base_sup = sorted(_edge_support(q[i], q[(i + 1) % 4], lab_f) for i in range(4))
    ref = float(np.median(base_sup))
    best, best_sup = None, 0.0
    for i in edges:
        a, b = q[i], q[(i + 1) % 4]
        e = b - a
        u = e / (np.linalg.norm(e) + 1e-9)
        n = np.array([-u[1], u[0]], np.float32)
        if np.dot(n, (a + b) / 2 - centroid) < 0:
            n = -n
        line = ((a + b) / 2 + n * need, u)
        prev_l = (q[(i - 1) % 4], q[i] - q[(i - 1) % 4])
        next_l = (q[(i + 1) % 4], q[(i + 2) % 4] - q[(i + 1) % 4])
        pa, pb = _intersect(line, prev_l), _intersect(line, next_l)
        if pa is None or pb is None:
            continue
        sup = _edge_support(pa, pb, lab_f)
        if sup > best_sup:
            nq = q.copy()
            nq[i], nq[(i + 1) % 4] = pa, pb
            best, best_sup = nq, sup
    if best is not None and best_sup >= 0.6 * ref and best_sup >= 5.0:
        return best
    return q


# --------------------------------------------------------------------------
# Sinh ung vien khung the
# --------------------------------------------------------------------------
def _contour_to_quad(cnt):
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for eps in (0.01, 0.018, 0.028, 0.04, 0.06):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            return ap.reshape(4, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)


def _quads_from_mask(mask, min_area, maxn=3, mode=cv2.RETR_EXTERNAL):
    cnts, _ = cv2.findContours(mask, mode, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:maxn]
    out = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            break
        out.append(_contour_to_quad(c))
    return out


def _clean_mask(m, k):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
    return m


def _gen_candidates(small, gray, lab8, ed):
    H, W = gray.shape
    min_area = 0.05 * H * W
    k = max(5, int(0.02 * min(H, W)) | 1)
    cands = []

    # 1) Canh Canny -> contour
    for sigma in (0.25, 0.45):
        g = cv2.GaussianBlur(gray, (5, 5), 0)
        med = float(np.median(g))
        e = cv2.Canny(g, int(max(5, (1 - sigma) * med * 0.7)), int(min(255, (1 + sigma) * med + 20)))
        e = cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=1)
        e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        cands += _quads_from_mask(e, min_area, maxn=6, mode=cv2.RETR_LIST)

    # 2) Nguong Otsu tren nhieu kenh mau, ca hai chieu
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    chans = [gray, lab8[:, :, 0], lab8[:, :, 1], lab8[:, :, 2], hsv[:, :, 1], hsv[:, :, 2]]
    for ch in chans:
        ch = cv2.GaussianBlur(ch, (5, 5), 0)
        _, m = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for mm in (m, 255 - m):
            cands += _quads_from_mask(_clean_mask(mm, k), min_area, maxn=2)

    # 2b) Mau dac trung cua the CCCD (xanh-lo / xanh-lam nhat), bat ke nen
    hue = ((hsv[:, :, 0] >= 55) & (hsv[:, :, 0] <= 125) & (hsv[:, :, 1] >= 18) & (hsv[:, :, 2] >= 90))
    mh = hue.astype(np.uint8) * 255
    kk = max(9, int(0.06 * min(H, W)) | 1)
    mh = cv2.morphologyEx(mh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
    cands += _quads_from_mask(_clean_mask(mh, k), min_area, maxn=2)

    # 3) Khoang cach mau so voi nen (lay mau tu vien anh)
    fr = max(2, int(0.03 * min(H, W)))
    lab = lab8.astype(np.float32)
    border = np.concatenate([
        lab[:fr].reshape(-1, 3), lab[-fr:].reshape(-1, 3),
        lab[:, :fr].reshape(-1, 3), lab[:, -fr:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    dist = cv2.GaussianBlur(dist, (5, 5), 0)
    d8 = np.clip(dist * (255.0 / max(1.0, dist.max())), 0, 255).astype(np.uint8)
    _, m = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cands += _quads_from_mask(_clean_mask(m, k), min_area, maxn=2)

    # 4) GrabCut (anh nho) - hieu qua khi nen phuc tap
    try:
        s = 360.0 / max(H, W)
        sm = cv2.resize(small, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h2, w2 = sm.shape[:2]
        mask = np.zeros((h2, w2), np.uint8)
        rect = (int(0.04 * w2), int(0.04 * h2), int(0.92 * w2), int(0.92 * h2))
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(sm, mask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
        fg = cv2.resize(fg, (W, H), interpolation=cv2.INTER_NEAREST)
        cands += _quads_from_mask(_clean_mask(fg, k), min_area, maxn=2)
    except Exception:
        pass

    return cands


# --------------------------------------------------------------------------
# Cham diem ung vien
# --------------------------------------------------------------------------
def _edge_contrast(q, lab_f, d):
    """Do tuong phan mau trong/ngoai 4 canh: ca o khoang rong (d) lan sat mep (2.5px)."""
    c = q.mean(0)
    vals = []
    t = np.linspace(0.12, 0.88, 40, dtype=np.float32)[:, None]
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        e = b - a
        n = np.array([-e[1], e[0]], np.float32) / (np.linalg.norm(e) + 1e-9)
        if np.dot(n, c - (a + b) / 2) < 0:
            n = -n
        p = a + e * t
        parts = []
        for dist, norm_ in ((d, 22.0), (2.5, 15.0)):
            vi, oi = _sample_lab(lab_f, p + n * dist)
            vo, oo = _sample_lab(lab_f, p - n * dist)
            ok = oi & oo
            if ok.mean() < 0.4:
                parts = None
                break
            diff = np.linalg.norm(vi[ok] - vo[ok], axis=1)
            parts.append(min(1.0, float(np.median(diff)) / norm_))
        vals.append(0.12 if parts is None else 0.5 * parts[0] + 0.5 * parts[1])
    vals = np.array(vals)
    return 0.6 * vals.mean() + 0.4 * vals.min()


def score_quad(q, lab_f, shape):
    H, W = shape
    diag = math.hypot(H, W)
    ratio = poly_area(q) / (H * W)
    if not (0.05 <= ratio <= 1.3):
        return -1.0
    if not cv2.isContourConvex(q.reshape(-1, 1, 2).astype(np.float32)):
        return -1.0
    sl = side_lengths(q)
    if min(sl) < 0.1 * diag * 0.5:
        return -1.0
    # ti le canh doi
    if max(sl[0], sl[2]) / max(1e-6, min(sl[0], sl[2])) > 1.9:
        return -1.0
    if max(sl[1], sl[3]) / max(1e-6, min(sl[1], sl[3])) > 1.9:
        return -1.0
    angs = interior_angles(q)
    dev = max(abs(a - 90) for a in angs)
    if dev > 38:
        return -1.0
    longs = (sl[0] + sl[2]) / 2
    shorts = (sl[1] + sl[3]) / 2
    asp = max(longs, shorts) / max(1e-6, min(longs, shorts))
    if not (1.2 <= asp <= 2.1):
        return -1.0
    aspect_term = 1 - min(1.0, abs(math.log(asp / CARD_ASPECT)) / 0.30)
    angle_term = 1 - min(1.0, dev / 30.0)
    area_term = min(1.0, math.sqrt(min(ratio, 1.0)) / 0.7)
    d = max(4.0, 0.014 * diag)
    contrast = _edge_contrast(q, lab_f, d)
    out = 0.0
    m = 0.02 * diag
    for p in q:
        if p[0] < -m or p[0] > W + m or p[1] < -m or p[1] > H + m:
            out += 1
    return 0.52 * contrast + 0.12 * area_term + 0.14 * angle_term + 0.22 * aspect_term - 0.12 * out


# --------------------------------------------------------------------------
# Tim the
# --------------------------------------------------------------------------
def _ring_card_fraction(img, f=0.03):
    """Ti le diem anh o vien khung co mau xanh-lo/xanh-lam nhat cua the CCCD."""
    H, W = img.shape[:2]
    r = max(2, int(f * min(H, W)))
    m = np.zeros((H, W), bool)
    m[:r] = m[-r:] = True
    m[:, :r] = m[:, -r:] = True
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s = hsv[:, :, 0][m], hsv[:, :, 1][m]
    return float(((h >= 55) & (h <= 120) & (s >= 20)).mean())



def detect_card_quad(img, max_side=1000):
    """Tra ve (quad 4x2 theo toa do anh goc, diem tin cay) hoac (None, diem)."""
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
    H, W = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    lab8 = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    lab_f = cv2.GaussianBlur(cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB), (5, 5), 0)
    ed = _EdgeData(gray)
    diag = math.hypot(H, W)

    cands = _gen_candidates(small, gray, lab8, ed)
    scored, seen = [], []
    for q0 in cands:
        q0 = order_quad(q0)
        if any(np.abs(q0 - o).max() < 0.008 * diag for o in seen):
            continue
        seen.append(q0)
        best_local = None
        qr = refine_quad(q0, ed, diag)
        p0, p1 = polish_quad(q0, lab_f, diag), polish_quad(qr, lab_f, diag)
        variants = [qr, q0, p0, p1, fix_aspect(qr, lab_f), fix_aspect(p0, lab_f), fix_aspect(p1, lab_f)]
        for q in variants:
            q = order_quad(q)
            sc = score_quad(q, lab_f, (H, W))
            if best_local is None or sc > best_local[0]:
                best_local = (sc, q)
        scored.append(best_local)

    # Truong hop anh DA la the cat san (the lap day khung): them ung vien "toan khung"
    frame = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    asp_img = max(W, H) / min(W, H)
    if abs(math.log(asp_img / CARD_ASPECT)) < 0.07:
        scored.append((score_quad(frame, lab_f, (H, W)), frame))

    scored = [t for t in scored if t[0] > 0]
    if not scored:
        return None, -1.0
    scored.sort(key=lambda t: -t[0])
    # Xet them "do giong the CCCD" (co QR / quoc huy do dung vi tri) cho vai ung vien dau
    pool = scored[:4]
    if not any(np.allclose(p[1], frame) for p in pool) and abs(math.log(asp_img / CARD_ASPECT)) < 0.07:
        pool.append(scored[[np.allclose(t[1], frame) for t in scored].index(True)])
    final = []
    dd = max(4.0, 0.014 * diag)
    for sc, q in pool:
        try:
            wc = warp_quad(small, q, out_w=700, inset=0.0)
            wc = cv2.resize(wc, (1012, int(round(1012 / CARD_ASPECT))), interpolation=cv2.INTER_AREA)
            cardness = 0.5 * (abs(_qr_cue(wc)) + abs(_red_cue(wc)))
        except Exception:
            cardness = 0.0
        is_frame = bool(np.allclose(q, frame))
        final.append(dict(total=sc + 0.35 * cardness, score=sc, quad=q, cardness=cardness,
                          contrast=_edge_contrast(q, lab_f, dd), is_frame=is_frame))
    final.sort(key=lambda t: -t["total"])
    best = final[0]
    # Anh la the da cat san (khong co nen): neu ung vien tot nhat khong co vien ro rang
    # ma toan khung nhin giong the CCCD -> dung nguyen khung.
    if not best["is_frame"] and best["contrast"] < 0.85 and _ring_card_fraction(small) >= 0.4:
        fr = [t for t in final if t["is_frame"] and t["cardness"] >= 0.4]
        if fr:
            best = fr[0]
    if best["score"] < MIN_SCORE:
        return None, best["score"]
    return best["quad"] / s, min(1.0, best["total"])


# --------------------------------------------------------------------------
# Xac dinh chieu (the CCCD mat truoc: quoc huy do goc tren-trai, QR goc tren-phai)
# --------------------------------------------------------------------------
def _red_cue(card):
    hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = (((h < 10) | (h > 168)) & (s > 110) & (v > 110)).astype(np.float32)
    H, W = red.shape
    tl = red[: int(0.45 * H), : int(0.42 * W)].mean()
    br = red[int(0.55 * H):, int(0.58 * W):].mean()
    if tl + br < 0.004:
        return 0.0
    return float(np.clip((tl - br) / (tl + br), -1, 1))


def _qr_cue(card):
    try:
        det = cv2.QRCodeDetector()
        ok, pts = det.detect(card)
        if not ok or pts is None:
            return 0.0
        c = np.asarray(pts).reshape(-1, 2).mean(0)
        H, W = card.shape[:2]
        if c[0] > 0.55 * W and c[1] < 0.5 * H:
            return 1.0
        if c[0] < 0.45 * W and c[1] > 0.5 * H:
            return -1.0
    except Exception:
        pass
    return 0.0


def _photo_cue(card):
    """Khung anh chan dung nam o nua trai: vung 'da/toc' co nhieu chi tiet o ben trai."""
    g = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    # dac trung: anh chan dung toi hon va tuong phan manh hon nen the o nua trai-duoi
    left = g[int(0.30 * H): int(0.95 * H), int(0.04 * W): int(0.34 * W)]
    right = g[int(0.05 * H): int(0.70 * H), int(0.66 * W): int(0.96 * W)]
    # so sanh do lech chuan (anh chan dung >> hoa van nen)
    a, b = float(left.std()), float(right.std())
    if a + b < 1:
        return 0.0
    return float(np.clip((a - b) / (a + b) * 2.0, -1, 1))


def needs_flip180(card):
    c = cv2.resize(card, (1012, int(round(1012 / CARD_ASPECT))), interpolation=cv2.INTER_AREA)
    qr, red = _qr_cue(c), _red_cue(c)
    score = 1.6 * qr + 1.2 * red + 0.5 * _photo_cue(c)
    cardlike = bool(abs(qr) > 0 or abs(red) >= 0.3)   # thay QR / quoc huy do = dung la mat truoc CCCD
    return score < -0.3, score, cardlike


# --------------------------------------------------------------------------
# API chinh
# --------------------------------------------------------------------------
def warp_quad(img, q, out_w=None, inset=0.004):
    q = order_quad(q)
    sl = side_lengths(q)
    # dam bao canh dai nam ngang
    if (sl[1] + sl[3]) > (sl[0] + sl[2]):
        q = np.roll(q, -1, axis=0)
        sl = side_lengths(q)
    c = q.mean(0)
    q = c + (q - c) * (1 - 2 * inset)
    if out_w is None:
        out_w = int(np.clip(round(max(sl[0], sl[2])), 400, 2400))
    out_h = int(round(out_w / CARD_ASPECT))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def enhance(card):
    lab = cv2.cvtColor(card, cv2.COLOR_BGR2LAB)
    cl = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    lab[:, :, 0] = cl.apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 1.2)
    return cv2.addWeighted(out, 1.4, blur, -0.4, 0)


def scan_card(img, out_width=None, inset=0.004, orient=True, do_enhance=False):
    """
    Tra ve dict:
      found   : True/False
      card    : anh the da cat + nan thang (BGR) hoac None
      quad    : 4 goc (tl,tr,br,bl) tren anh goc
      score   : do tin cay 0..1
      flipped : da xoay 180 do de dung chieu hay chua
      cardlike: True neu thay dac trung mat truoc CCCD (QR / quoc huy do) -> ket qua dang tin
    """
    if img is None:
        return dict(found=False, card=None, quad=None, score=0.0, flipped=False)
    quad, score = detect_card_quad(img)
    if quad is None:
        return dict(found=False, card=None, quad=None, score=float(score), flipped=False)
    card = warp_quad(img, quad, out_w=out_width, inset=inset)
    flipped, _, cardlike = needs_flip180(card)
    if orient and flipped:
        card = cv2.rotate(card, cv2.ROTATE_180)
    else:
        flipped = False
    if do_enhance:
        card = enhance(card)
    return dict(found=True, card=card, quad=order_quad(quad), score=float(score), flipped=bool(flipped), cardlike=cardlike)


def draw_debug(img, res):
    out = img.copy()
    if res["quad"] is not None:
        q = res["quad"].astype(np.int32).reshape(-1, 1, 2)
        th = max(2, int(0.004 * max(out.shape[:2])))
        cv2.polylines(out, [q], True, (0, 255, 0), th)
        for i, p in enumerate(q.reshape(-1, 2)):
            cv2.circle(out, tuple(int(v) for v in p), th * 2, (0, 0, 255), -1)
    return out


# --------------------------------------------------------------------------
# Dong lenh
# --------------------------------------------------------------------------
def _collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for n in sorted(os.listdir(p)):
                if os.path.splitext(n)[1].lower() in IMG_EXT:
                    files.append(os.path.join(p, n))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print("Khong thay:", p)
    return files


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Nhan dien, cat va nan thang anh CCCD")
    ap.add_argument("inputs", nargs="*", help="file anh hoac thu muc")
    ap.add_argument("-o", "--out", default=None, help="thu muc ket qua (mac dinh: ./output)")
    ap.add_argument("--width", type=int, default=None, help="chieu rong anh ra (mac dinh: theo anh goc)")
    ap.add_argument("--inset", type=float, default=0.004, help="cat sat vao trong (ti le), mac dinh 0.004")
    ap.add_argument("--no-orient", action="store_true", help="khong tu xoay 180 do")
    ap.add_argument("--enhance", action="store_true", help="tang do net/tuong phan nhe")
    ap.add_argument("--png", action="store_true", help="luu PNG thay vi JPG")
    ap.add_argument("--debug", action="store_true", help="luu them anh danh dau 4 goc")
    a = ap.parse_args(argv)

    inputs = a.inputs
    if not inputs:
        default_in = os.path.join(here, "input")
        if os.path.isdir(default_in):
            inputs = [default_in]
        else:
            ap.print_help()
            return 1
    files = _collect(inputs)
    if not files:
        print("Khong co anh nao de xu ly.")
        return 1
    out_dir = a.out or os.path.join(here, "output")
    os.makedirs(out_dir, exist_ok=True)
    ext = ".png" if a.png else ".jpg"
    ok_n = 0
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        img = imread(f)
        if img is None:
            print(f"[LOI ] {f}: khong doc duoc anh")
            continue
        res = scan_card(img, out_width=a.width, inset=a.inset,
                        orient=not a.no_orient, do_enhance=a.enhance)
        if a.debug:
            imwrite(os.path.join(out_dir, stem + "_debug.jpg"), draw_debug(img, res))
        if res["found"]:
            imwrite(os.path.join(out_dir, stem + "_cccd" + ext), res["card"])
            ok_n += 1
            note = " (da xoay 180)" if res["flipped"] else ""
            if not res.get("cardlike", True):
                note += "  [CANH BAO: khong thay QR/quoc huy - hay mo anh kiem tra lai; neu la mat sau the thi binh thuong]"
            print(f"[OK  ] {os.path.basename(f)} -> {stem}_cccd{ext}  tin cay={res['score']:.2f}{note}")
        else:
            print(f"[KHONG TIM THAY THE] {os.path.basename(f)}  (diem={res['score']:.2f}) - hay chup ro 4 goc the")
    print(f"Xong: {ok_n}/{len(files)} anh. Ket qua o: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
