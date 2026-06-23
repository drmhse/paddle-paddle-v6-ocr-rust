//! Small 2D geometry helpers for DB post-processing: convex hull,
//! minimum-area rectangle, the PaddleOCR point ordering, and box unclip.

pub type Pt = (f32, f32);

/// Andrew's monotone-chain convex hull. Returns CCW hull without duplicate
/// endpoint. Input may contain duplicates / collinear points.
pub fn convex_hull(points: &[Pt]) -> Vec<Pt> {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap()
            .then(a.1.partial_cmp(&b.1).unwrap())
    });
    pts.dedup();
    let n = pts.len();
    if n < 3 {
        return pts;
    }
    let cross = |o: Pt, a: Pt, b: Pt| (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0);

    let mut hull: Vec<Pt> = Vec::with_capacity(2 * n);
    // lower
    for &p in &pts {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    // upper
    let lower = hull.len() + 1;
    for &p in pts.iter().rev().skip(1) {
        while hull.len() >= lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    hull
}

/// Minimum-area enclosing rectangle via rotating-calipers over hull edges.
/// Returns the 4 corners and the rectangle's (width, height) min side.
pub fn min_area_rect(points: &[Pt]) -> ([Pt; 4], f32) {
    let hull = convex_hull(points);
    if hull.len() < 3 {
        // Degenerate: build a tiny axis-aligned box around available points.
        let (mut minx, mut miny, mut maxx, mut maxy) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for &(x, y) in points {
            minx = minx.min(x);
            miny = miny.min(y);
            maxx = maxx.max(x);
            maxy = maxy.max(y);
        }
        let r = [
            (minx, miny),
            (maxx, miny),
            (maxx, maxy),
            (minx, maxy),
        ];
        return (r, (maxx - minx).min(maxy - miny));
    }

    let n = hull.len();
    let mut best_area = f32::MAX;
    let mut best: [Pt; 4] = [(0.0, 0.0); 4];

    for i in 0..n {
        let p0 = hull[i];
        let p1 = hull[(i + 1) % n];
        let dx = p1.0 - p0.0;
        let dy = p1.1 - p0.1;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-6 {
            continue;
        }
        // Edge unit vector (ux,uy) and its normal (-uy,ux).
        let (ux, uy) = (dx / len, dy / len);
        let (mut min_u, mut max_u) = (f32::MAX, f32::MIN);
        let (mut min_v, mut max_v) = (f32::MAX, f32::MIN);
        for &(x, y) in &hull {
            let u = x * ux + y * uy;
            let v = -x * uy + y * ux;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let area = (max_u - min_u) * (max_v - min_v);
        if area < best_area {
            best_area = area;
            // Map (u,v) corners back to xy.
            let to_xy = |u: f32, v: f32| (u * ux - v * uy, u * uy + v * ux);
            best = [
                to_xy(min_u, min_v),
                to_xy(max_u, min_v),
                to_xy(max_u, max_v),
                to_xy(min_u, max_v),
            ];
        }
    }

    let w = dist(best[0], best[1]);
    let h = dist(best[1], best[2]);
    (best, w.min(h))
}

/// Reorder 4 points into PaddleOCR's convention (tl, tr, br, bl).
pub fn order_box(box4: [Pt; 4]) -> [Pt; 4] {
    let mut pts = box4;
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    // left pair = pts[0..2], right pair = pts[2..4]
    let (tl, bl) = if pts[1].1 > pts[0].1 {
        (pts[0], pts[1])
    } else {
        (pts[1], pts[0])
    };
    let (tr, br) = if pts[3].1 > pts[2].1 {
        (pts[2], pts[3])
    } else {
        (pts[3], pts[2])
    };
    [tl, tr, br, bl]
}

/// Expand a quad outward by `ratio` (DB "unclip"). For a near-rectangular box
/// this matches pyclipper's polygon offset: distance = area*ratio/perimeter,
/// each side pushed out by `distance`. Implemented by growing the min-area
/// rectangle, which is equivalent for rectangles and dependency-free.
pub fn unclip(box4: [Pt; 4], ratio: f32) -> [Pt; 4] {
    let area = poly_area(&box4).abs();
    let perim = poly_perimeter(&box4);
    if perim < 1e-6 {
        return box4;
    }
    let distance = area * ratio / perim;

    // Centroid.
    let cx = box4.iter().map(|p| p.0).sum::<f32>() / 4.0;
    let cy = box4.iter().map(|p| p.1).sum::<f32>() / 4.0;

    // Push each corner outward from the centroid along the box diagonals by a
    // step that moves the adjacent edges out by `distance`.
    let mut out = [(0.0f32, 0.0f32); 4];
    for (i, &(x, y)) in box4.iter().enumerate() {
        let (vx, vy) = (x - cx, y - cy);
        let m = (vx * vx + vy * vy).sqrt().max(1e-6);
        // sqrt(2) compensates the diagonal direction so edges move ~`distance`.
        let grow = distance * std::f32::consts::SQRT_2 / m;
        out[i] = (x + vx * grow, y + vy * grow);
    }
    out
}

pub fn dist(a: Pt, b: Pt) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

fn poly_area(p: &[Pt; 4]) -> f32 {
    let mut s = 0.0;
    for i in 0..4 {
        let j = (i + 1) % 4;
        s += p[i].0 * p[j].1 - p[j].0 * p[i].1;
    }
    s / 2.0
}

fn poly_perimeter(p: &[Pt; 4]) -> f32 {
    (0..4).map(|i| dist(p[i], p[(i + 1) % 4])).sum()
}

/// Point-in-quad test (even-odd rule) used by the box score.
pub fn point_in_quad(px: f32, py: f32, q: &[Pt; 4]) -> bool {
    let mut inside = false;
    let mut j = 3;
    for i in 0..4 {
        let (xi, yi) = q[i];
        let (xj, yj) = q[j];
        if ((yi > py) != (yj > py))
            && (px < (xj - xi) * (py - yi) / (yj - yi + f32::EPSILON) + xi)
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}
