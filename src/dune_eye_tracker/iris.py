"""The iris found in the full camera frame, not in MediaPipe's face crop.

MediaPipe puts the face's landmarks from a ~256 px crop of it: at 37 cm, one degree of eye
turn moves the iris a third of a pixel there. Here each eye is cut from the full frame
(1920x1080: more pixels) along its corners (`crop`: aligned, the eye `EYE_PX` wide), and
the iris is found in it two ways:

- `limbus`: from MediaPipe's iris centre, rays out on both sides (the lids cover the top
  and the bottom: `SECTORS`) find where it turns from dark (iris) to light (sclera), to a
  fraction of a pixel; a circle is fitted to those edge points robustly, and the rays cast
  again from its centre. Its centre and radius, and how many rays agreed.
- `dark`: the centroid of how dark the iris disc is (no edges: lids and lashes pull it).

Both in image pixels of the full frame. No image is kept unless asked (`record
--keep-eyes`), only these numbers.
"""

from __future__ import annotations

import numpy as np

EYE_PX = 90.0  # the eye's corner to corner in the crop (px)
CROP = (160, 100)  # crop width, height (px)
SECTORS = ((-50.0, 50.0), (130.0, 230.0))  # degrees of the rays (0: towards the image's right), the sides
RAYS = 18  # per sector
INNER, OUTER = 0.45, 1.7  # ray from / to, in MediaPipe's iris radius
STEP = 0.5  # px along a ray
FIT_TOLS = (4.0, 2.5, 1.5, 1.0)  # px: an edge point this far off the circle does not count, tighter each pass
MIN_EDGES = 10
DARK_PCT = 25  # the darkest quarter of the disc counts (on 2 recordings 15-60 all gave 178-197 px)
IRIS = 0.21  # an iris's radius in eye widths (corner to corner), when MediaPipe's is off


def crop_affine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """The 2x3 map from crop pixels to image pixels for the eye between corners `left`
    and `right` (image px): the eye level, centred, `EYE_PX` wide."""
    d = right - left
    width = float(np.hypot(*d))
    s = width / EYE_PX  # image px per crop px
    c, sn = d / width
    centre = (left + right) / 2
    cw, ch = CROP
    rot = np.array([[c, -sn], [sn, c]]) * s
    return np.column_stack([rot, centre - rot @ np.array([cw / 2, ch / 2])])


def apply(a: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ a[:, :2].T + a[:, 2]


def invert(a: np.ndarray) -> np.ndarray:
    m = np.linalg.inv(a[:, :2])
    return np.column_stack([m, -m @ a[:, 2]])


def crops(gray: np.ndarray, landmarks: np.ndarray, eyes) -> tuple[np.ndarray, np.ndarray]:
    """The two eyes cut from the full grey frame: (2, h, w) uint8 crops and their (2, 2, 3)
    crop-to-image maps. `landmarks` normalised (478, >=2)."""
    import cv2

    h, w = gray.shape
    px = landmarks[:, :2] * np.array([w, h])
    out, maps = [], []
    for eye in eyes:
        a = crop_affine(px[eye["left"]], px[eye["right"]])
        out.append(cv2.warpAffine(gray, a, CROP, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE))
        maps.append(a)
    return np.array(out), np.array(maps)


def _sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear samples of `img` at (x, y), clamped to its edge."""
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def _smooth(img: np.ndarray) -> np.ndarray:
    k = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16
    p = np.pad(img, 2, mode="edge")
    p = sum(k[i] * p[:, i : i + img.shape[1]] for i in range(5))
    return sum(k[i] * p[i : i + img.shape[0], :] for i in range(5))


def edges(img: np.ndarray, centre: np.ndarray, radius: float) -> np.ndarray:
    """Where each ray from `centre` turns most from dark to light (crop px), subpixel;
    rays without a clear turn left out."""
    angles = np.concatenate([np.linspace(*np.radians(s), RAYS) for s in SECTORS])
    t = np.arange(INNER * radius, OUTER * radius, STEP)
    dx, dy = np.cos(angles), np.sin(angles)
    xs = centre[0] + dx[:, None] * t[None]
    ys = centre[1] + dy[:, None] * t[None]
    prof = _sample(img, xs, ys)
    grad = np.gradient(prof, STEP, axis=1)
    k = np.argmax(grad, axis=1)
    k = np.clip(k, 1, len(t) - 2)
    g0, g1, g2 = (grad[np.arange(len(k)), k + j] for j in (-1, 0, 1))
    den = g0 - 2 * g1 + g2
    off = np.where(np.abs(den) > 1e-9, 0.5 * (g0 - g2) / den, 0.0)
    r = t[k] + np.clip(off, -1, 1) * STEP
    strong = (g1 > max(2.0, 0.25 * float(np.max(g1)))) & (r > 0.6 * radius)  # a real edge, not noise or a lid
    pts = np.column_stack([centre[0] + dx * r, centre[1] + dy * r])
    return pts[strong]


def fit_circle(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Least squares circle (Kasa): centre, radius."""
    a = np.column_stack([2 * pts, np.ones(len(pts))])
    b = (pts**2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    c = sol[:2]
    return c, float(np.sqrt(max(sol[2] + c @ c, 1e-9)))


def robust_circle(pts: np.ndarray, radius: float) -> tuple[np.ndarray, float, np.ndarray] | None:
    """A circle through most of `pts`: fitted, the points further than `FIT_TOL` dropped,
    again; a radius far from `radius` (MediaPipe's) is no iris."""
    keep = np.ones(len(pts), bool)
    for tol in FIT_TOLS:
        if keep.sum() < MIN_EDGES:
            return None
        c, r = fit_circle(pts[keep])
        res = np.abs(np.linalg.norm(pts - c, axis=1) - r)
        keep = res < max(tol, 2.5 * float(np.median(res[keep]))) if tol == FIT_TOLS[0] else res < tol
    if keep.sum() < MIN_EDGES or not (0.6 * radius < r < 1.5 * radius):
        return None
    return c, r, keep


def limbus(img: np.ndarray, centre: np.ndarray, radius: float) -> np.ndarray:
    """(x, y, radius, inliers) in crop px; NaN when not found."""
    img = _smooth(img.astype(float))
    c = centre
    best = None
    for _ in range(3):  # cast again from the fitted centre
        fit = robust_circle(edges(img, c, radius), radius)
        if fit is None:
            break
        c, r, keep = fit
        best = np.array([c[0], c[1], r, keep.sum()])
    return best if best is not None else np.full(4, np.nan)


def dark(img: np.ndarray, centre: np.ndarray, radius: float) -> np.ndarray:
    """The centroid (x, y, crop px) of the darkness within the iris disc."""
    img = _smooth(img.astype(float))
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    disc = (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 <= (1.1 * radius) ** 2
    if disc.sum() < 10:
        return np.full(2, np.nan)
    v = img[disc]
    wgt = np.clip(np.percentile(v, DARK_PCT) - v, 0, None)
    if wgt.sum() <= 0:
        return np.full(2, np.nan)
    return np.array([(xx[disc] * wgt).sum() / wgt.sum(), (yy[disc] * wgt).sum() / wgt.sum()])


def locate(crop: np.ndarray, a: np.ndarray, iris_px: np.ndarray, iris_radius_px: float) -> tuple[np.ndarray, np.ndarray]:
    """One eye: (limbus x, y image px, radius image px, inliers), (dark x, y image px).
    `iris_px` and `iris_radius_px`: MediaPipe's, in image px (where to start)."""
    inv = invert(a)
    scale = float(np.sqrt(abs(np.linalg.det(a[:, :2]))))  # image px per crop px
    c = apply(inv, iris_px[None])[0]
    r = iris_radius_px / scale
    if not (0.1 * EYE_PX < r < 0.4 * EYE_PX):  # MediaPipe's rim is off: a usual iris (0.21 of the eye's width)
        r = IRIS * EYE_PX
    lb = limbus(crop, c, r)
    dk = dark(crop, c, r)
    lb_img = np.full(4, np.nan)
    if np.all(np.isfinite(lb)):
        lb_img = np.array([*apply(a, lb[None, :2])[0], lb[2] * scale, lb[3]])
    dk_img = apply(a, dk[None])[0] if np.all(np.isfinite(dk)) else np.full(2, np.nan)
    return lb_img, dk_img


def iris_radius(landmarks_px: np.ndarray, first: int) -> float:
    """MediaPipe's iris radius (px): its 4 rim points around the centre `first`."""
    rim = landmarks_px[first + 1 : first + 5]
    return float(np.mean(np.linalg.norm(rim - landmarks_px[first], axis=1)))


def both(gray: np.ndarray, landmarks: np.ndarray, eyes) -> dict:
    """Both eyes of a frame: `crops`, `maps`, `limbus` (2, 4), `dark` (2, 2)."""
    h, w = gray.shape
    px = landmarks[:, :2] * np.array([w, h])
    cs, maps = crops(gray, landmarks, eyes)
    lbs, dks = [], []
    for k, eye in enumerate(eyes):
        lb, dk = locate(cs[k], maps[k], px[eye["iris"]], iris_radius(px, eye["iris"]))
        lbs.append(lb)
        dks.append(dk)
    return {"crops": cs, "maps": maps, "limbus": np.array(lbs), "dark": np.array(dks)}
