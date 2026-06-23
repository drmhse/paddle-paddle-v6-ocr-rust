//! DBPostProcess: probability map -> oriented text boxes (PaddleOCR-compatible).

use crate::config::DetParams;
use crate::geometry::{min_area_rect, order_box, point_in_quad, unclip, Pt};
use crate::preprocess::Preprocessed;
use image::{GrayImage, Luma};
use imageproc::contours::{find_contours, BorderType};

#[derive(Debug, Clone)]
pub struct DetectedBox {
    /// Quad in original-image pixel coordinates, ordered tl, tr, br, bl.
    pub points: [Pt; 4],
    pub score: f32,
}

/// `prob` is the network output probability map of shape (H=net_h, W=net_w),
/// row-major. Returns boxes mapped back to the original image.
pub fn db_postprocess(
    prob: &[f32],
    net_w: usize,
    net_h: usize,
    pre: &Preprocessed,
    p: &DetParams,
) -> Vec<DetectedBox> {
    // 1. Binarize.
    let mut bitmap = GrayImage::new(net_w as u32, net_h as u32);
    for y in 0..net_h {
        for x in 0..net_w {
            let v = prob[y * net_w + x];
            bitmap.put_pixel(x as u32, y as u32, Luma([if v > p.thresh { 255 } else { 0 }]));
        }
    }

    // 2. Outer contours.
    let contours = find_contours::<u32>(&bitmap);

    let mut boxes = Vec::new();
    for contour in contours.into_iter() {
        if boxes.len() >= p.max_candidates {
            break;
        }
        if contour.border_type != BorderType::Outer || contour.points.len() < 4 {
            continue;
        }
        let pts: Vec<Pt> = contour
            .points
            .iter()
            .map(|pt| (pt.x as f32, pt.y as f32))
            .collect();

        // 3. Mini box + size filter.
        let (raw_box, sside) = min_area_rect(&pts);
        if sside < p.min_size {
            continue;
        }
        let raw_box = order_box(raw_box);

        // 4. Score on the probability map.
        let score = box_score(prob, net_w, net_h, &raw_box);
        if score < p.box_thresh {
            continue;
        }

        // 5. Unclip then re-fit the rectangle.
        let expanded = unclip(raw_box, p.unclip_ratio);
        let (final_box, sside2) = min_area_rect(&expanded);
        if sside2 < p.min_size + 2.0 {
            continue;
        }
        let final_box = order_box(final_box);

        // 6. Map network coords -> original image coords and clip.
        let ow = pre.orig_w as f32;
        let oh = pre.orig_h as f32;
        let mut mapped = [(0.0f32, 0.0f32); 4];
        for i in 0..4 {
            let x = (final_box[i].0 / pre.ratio_w).clamp(0.0, ow);
            let y = (final_box[i].1 / pre.ratio_h).clamp(0.0, oh);
            mapped[i] = (x.round(), y.round());
        }
        boxes.push(DetectedBox {
            points: mapped,
            score,
        });
    }
    boxes
}

/// Mean probability inside the quad (PaddleOCR `box_score_fast`): fill the quad
/// within its axis-aligned bounding box and average the covered map values.
fn box_score(prob: &[f32], w: usize, h: usize, q: &[Pt; 4]) -> f32 {
    let xs = [q[0].0, q[1].0, q[2].0, q[3].0];
    let ys = [q[0].1, q[1].1, q[2].1, q[3].1];
    let xmin = xs.iter().cloned().fold(f32::MAX, f32::min).floor().max(0.0) as usize;
    let xmax = xs
        .iter()
        .cloned()
        .fold(f32::MIN, f32::max)
        .ceil()
        .min((w - 1) as f32) as usize;
    let ymin = ys.iter().cloned().fold(f32::MAX, f32::min).floor().max(0.0) as usize;
    let ymax = ys
        .iter()
        .cloned()
        .fold(f32::MIN, f32::max)
        .ceil()
        .min((h - 1) as f32) as usize;
    if xmax < xmin || ymax < ymin {
        return 0.0;
    }

    let mut sum = 0.0f32;
    let mut count = 0u32;
    for y in ymin..=ymax {
        for x in xmin..=xmax {
            if point_in_quad(x as f32 + 0.5, y as f32 + 0.5, q) {
                sum += prob[y * w + x];
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}
