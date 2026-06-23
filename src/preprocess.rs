//! Image preprocessing: DetResizeForTest + NormalizeImage + ToCHWImage,
//! mirroring the PP-OCR detection pipeline described in `inference.yml`.

use crate::config::{DetParams, LimitType, Normalize};
use image::DynamicImage;
use tract_onnx::prelude::*;

pub struct Preprocessed {
    pub tensor: Tensor,
    /// network input size
    pub resize_w: u32,
    pub resize_h: u32,
    /// scale factors network->original: orig = net / ratio
    pub ratio_w: f32,
    pub ratio_h: f32,
    pub orig_w: u32,
    pub orig_h: u32,
}

/// Compute the resized dimensions (each a multiple of 32) per DetResizeForTest.
fn resize_dims(w: u32, h: u32, p: &DetParams) -> (u32, u32) {
    let (wf, hf) = (w as f32, h as f32);
    let limit = p.limit_side_len as f32;
    let ratio = match p.limit_type {
        LimitType::Max => {
            let m = wf.max(hf);
            if m > limit {
                limit / m
            } else {
                1.0
            }
        }
        LimitType::Min => {
            let m = wf.min(hf);
            if m < limit {
                limit / m
            } else {
                1.0
            }
        }
    };
    let round32 = |v: f32| (((v / 32.0).round() as i32).max(1) * 32) as u32;
    (round32(wf * ratio), round32(hf * ratio))
}

pub fn preprocess(img: &DynamicImage, p: &DetParams, n: &Normalize) -> Preprocessed {
    let orig_w = img.width();
    let orig_h = img.height();
    let (resize_w, resize_h) = resize_dims(orig_w, orig_h, p);

    let resized = img
        .resize_exact(resize_w, resize_h, image::imageops::FilterType::Triangle)
        .to_rgb8();

    let (rw, rh) = (resize_w as usize, resize_h as usize);
    // CHW float buffer, channel order per `n.bgr`.
    let mut data = vec![0f32; 3 * rh * rw];
    let plane = rh * rw;
    for (x, y, px) in resized.enumerate_pixels() {
        let idx = y as usize * rw + x as usize;
        let [r, g, b] = px.0;
        // RGB source -> arrange into network channel order
        let chans = if n.bgr { [b, g, r] } else { [r, g, b] };
        for c in 0..3 {
            let v = chans[c] as f32 * n.scale;
            data[c * plane + idx] = (v - n.mean[c]) / n.std[c];
        }
    }

    let tensor = Tensor::from_shape(&[1, 3, rh, rw], &data).expect("valid CHW shape");

    Preprocessed {
        tensor,
        resize_w,
        resize_h,
        ratio_w: resize_w as f32 / orig_w as f32,
        ratio_h: resize_h as f32 / orig_h as f32,
        orig_w,
        orig_h,
    }
}
