//! HTTP layer: routes, request params, JSON responses.

use crate::engine::{Engine, ParamOverrides};
use axum::{
    body::Bytes,
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

pub fn router(engine: Arc<Engine>) -> Router {
    use crate::docs;
    Router::new()
        .route("/", get(docs::demo_page))
        .route("/docs", get(docs::swagger_page))
        .route("/openapi.json", get(docs::openapi))
        .route("/swagger/:file", get(docs::swagger_asset))
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/detect", post(detect))
        .route("/v1/recognize", post(recognize))
        .route("/v1/ocr", post(ocr))
        .with_state(engine)
}

async fn health() -> &'static str {
    "ok"
}

fn err(status: StatusCode, msg: impl Into<String>) -> (StatusCode, Json<serde_json::Value>) {
    (status, Json(serde_json::json!({ "error": msg.into() })))
}

// ---------- /v1/models ----------

async fn list_models(State(engine): State<Arc<Engine>>) -> Json<serde_json::Value> {
    let det: Vec<_> = engine
        .det_ids()
        .into_iter()
        .map(|id| {
            let p = engine.default_params(&id).unwrap();
            serde_json::json!({
                "id": id, "thresh": p.thresh, "box_thresh": p.box_thresh,
                "unclip_ratio": p.unclip_ratio, "limit_side_len": p.limit_side_len
            })
        })
        .collect();
    Json(serde_json::json!({
        "detection": det,
        "recognition": engine.rec_ids(),
    }))
}

// ---------- shared helpers ----------

fn pick(model: Option<String>, ids: &[String], kind: &str) -> Result<String, (StatusCode, Json<serde_json::Value>)> {
    match model {
        Some(m) => Ok(m),
        None if ids.len() == 1 => Ok(ids[0].clone()),
        None => Err(err(
            StatusCode::BAD_REQUEST,
            format!("query param `{kind}` is required; available: {}", ids.join(", ")),
        )),
    }
}

fn decode_image(body: &Bytes) -> Result<image::DynamicImage, (StatusCode, Json<serde_json::Value>)> {
    if body.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "empty request body; POST raw image bytes"));
    }
    image::load_from_memory(body)
        .map_err(|e| err(StatusCode::BAD_REQUEST, format!("could not decode image: {e}")))
}

#[derive(Deserialize)]
struct DetectQuery {
    model: Option<String>,
    det_model: Option<String>,
    rec_model: Option<String>,
    thresh: Option<f32>,
    box_thresh: Option<f32>,
    unclip_ratio: Option<f32>,
    limit_side_len: Option<u32>,
    min_rec_score: Option<f32>,
}

impl DetectQuery {
    fn overrides(&self) -> ParamOverrides {
        ParamOverrides {
            thresh: self.thresh,
            box_thresh: self.box_thresh,
            unclip_ratio: self.unclip_ratio,
            limit_side_len: self.limit_side_len,
        }
    }
}

#[derive(Serialize)]
struct BoxOut {
    points: [[f32; 2]; 4],
    score: f32,
}

fn quad_to_points(q: &[(f32, f32); 4]) -> [[f32; 2]; 4] {
    [
        [q[0].0, q[0].1],
        [q[1].0, q[1].1],
        [q[2].0, q[2].1],
        [q[3].0, q[3].1],
    ]
}

// ---------- /v1/detect ----------

async fn detect(
    State(engine): State<Arc<Engine>>,
    Query(q): Query<DetectQuery>,
    body: Bytes,
) -> impl IntoResponse {
    let ids = engine.det_ids();
    let model = match pick(q.model.clone().or_else(|| q.det_model.clone()), &ids, "model") {
        Ok(m) => m,
        Err(e) => return e.into_response(),
    };
    if !engine.has_det(&model) {
        return err(StatusCode::NOT_FOUND, format!("unknown det model '{model}'; available: {}", ids.join(", "))).into_response();
    }
    let img = match decode_image(&body) {
        Ok(i) => i,
        Err(e) => return e.into_response(),
    };
    let (width, height) = (img.width(), img.height());
    let ov = q.overrides();

    let start = Instant::now();
    let engine2 = engine.clone();
    let m2 = model.clone();
    let res = tokio::task::spawn_blocking(move || engine2.detect(&m2, &img, ov)).await;
    let elapsed_ms = start.elapsed().as_millis();

    let (boxes, params) = match res {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return err(StatusCode::INTERNAL_SERVER_ERROR, format!("inference error: {e:#}")).into_response(),
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, format!("task join error: {e}")).into_response(),
    };

    let boxes_out: Vec<BoxOut> = boxes
        .iter()
        .map(|b| BoxOut { points: quad_to_points(&b.points), score: b.score })
        .collect();

    Json(serde_json::json!({
        "model": model,
        "image": { "width": width, "height": height },
        "params": {
            "thresh": params.thresh, "box_thresh": params.box_thresh,
            "unclip_ratio": params.unclip_ratio, "limit_side_len": params.limit_side_len
        },
        "num_boxes": boxes_out.len(),
        "elapsed_ms": elapsed_ms,
        "boxes": boxes_out,
    }))
    .into_response()
}

// ---------- /v1/recognize (single cropped line) ----------

async fn recognize(
    State(engine): State<Arc<Engine>>,
    Query(q): Query<DetectQuery>,
    body: Bytes,
) -> impl IntoResponse {
    let ids = engine.rec_ids();
    let model = match pick(q.model.clone().or_else(|| q.rec_model.clone()), &ids, "model") {
        Ok(m) => m,
        Err(e) => return e.into_response(),
    };
    if !engine.has_rec(&model) {
        return err(StatusCode::NOT_FOUND, format!("unknown rec model '{model}'; available: {}", ids.join(", "))).into_response();
    }
    let img = match decode_image(&body) {
        Ok(i) => i,
        Err(e) => return e.into_response(),
    };
    let line = img.to_rgb8();

    let start = Instant::now();
    let engine2 = engine.clone();
    let m2 = model.clone();
    let res = tokio::task::spawn_blocking(move || engine2.recognize_image(&m2, &line)).await;
    let elapsed_ms = start.elapsed().as_millis();

    match res {
        Ok(Ok((text, score))) => Json(serde_json::json!({
            "model": model, "text": text, "score": score, "elapsed_ms": elapsed_ms
        }))
        .into_response(),
        Ok(Err(e)) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("inference error: {e:#}")).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("task join error: {e}")).into_response(),
    }
}

// ---------- /v1/ocr (detect + recognize) ----------

#[derive(Serialize)]
struct LineOut {
    points: [[f32; 2]; 4],
    text: String,
    det_score: f32,
    rec_score: f32,
}

async fn ocr(
    State(engine): State<Arc<Engine>>,
    Query(q): Query<DetectQuery>,
    body: Bytes,
) -> impl IntoResponse {
    let det_ids = engine.det_ids();
    let rec_ids = engine.rec_ids();
    let det_model = match pick(q.det_model.clone().or_else(|| q.model.clone()), &det_ids, "det_model") {
        Ok(m) => m,
        Err(e) => return e.into_response(),
    };
    let rec_model = match pick(q.rec_model.clone(), &rec_ids, "rec_model") {
        Ok(m) => m,
        Err(e) => return e.into_response(),
    };
    if !engine.has_det(&det_model) {
        return err(StatusCode::NOT_FOUND, format!("unknown det model '{det_model}'")).into_response();
    }
    if !engine.has_rec(&rec_model) {
        return err(StatusCode::NOT_FOUND, format!("unknown rec model '{rec_model}'")).into_response();
    }
    let img = match decode_image(&body) {
        Ok(i) => i,
        Err(e) => return e.into_response(),
    };
    let (width, height) = (img.width(), img.height());
    let ov = q.overrides();
    let min_rec_score = q.min_rec_score.unwrap_or(0.0);

    let start = Instant::now();
    let engine2 = engine.clone();
    let (d2, r2) = (det_model.clone(), rec_model.clone());
    let res = tokio::task::spawn_blocking(move || engine2.ocr(&d2, &r2, &img, ov, min_rec_score)).await;
    let elapsed_ms = start.elapsed().as_millis();

    let (lines, params) = match res {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return err(StatusCode::INTERNAL_SERVER_ERROR, format!("inference error: {e:#}")).into_response(),
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, format!("task join error: {e}")).into_response(),
    };

    let full_text = lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join("\n");
    let lines_out: Vec<LineOut> = lines
        .iter()
        .map(|l| LineOut {
            points: quad_to_points(&l.points),
            text: l.text.clone(),
            det_score: l.det_score,
            rec_score: l.rec_score,
        })
        .collect();

    Json(serde_json::json!({
        "det_model": det_model,
        "rec_model": rec_model,
        "image": { "width": width, "height": height },
        "params": {
            "thresh": params.thresh, "box_thresh": params.box_thresh,
            "unclip_ratio": params.unclip_ratio, "limit_side_len": params.limit_side_len,
            "min_rec_score": min_rec_score
        },
        "num_lines": lines_out.len(),
        "elapsed_ms": elapsed_ms,
        "text": full_text,
        "lines": lines_out,
    }))
    .into_response()
}
