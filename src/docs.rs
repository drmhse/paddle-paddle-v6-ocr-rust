//! Static pages (demo + Swagger UI) and the OpenAPI spec, all embedded.

use crate::engine::Engine;
use axum::{
    extract::State,
    http::{header, HeaderValue, StatusCode},
    response::IntoResponse,
    Json,
};
use std::sync::Arc;

const DEMO_HTML: &str = include_str!("../assets/demo.html");
const SWAGGER_HTML: &str = include_str!("../assets/swagger.html");
const SWAGGER_CSS: &str = include_str!("../assets/swagger-ui.css");
const SWAGGER_JS: &str = include_str!("../assets/swagger-ui-bundle.js");

fn html(body: &'static str) -> impl IntoResponse {
    ([(header::CONTENT_TYPE, HeaderValue::from_static("text/html; charset=utf-8"))], body)
}

pub async fn demo_page() -> impl IntoResponse {
    html(DEMO_HTML)
}

pub async fn swagger_page() -> impl IntoResponse {
    html(SWAGGER_HTML)
}

pub async fn swagger_css() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, HeaderValue::from_static("text/css"))], SWAGGER_CSS)
}

pub async fn swagger_js() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, HeaderValue::from_static("application/javascript"))],
        SWAGGER_JS,
    )
}

pub async fn swagger_asset(axum::extract::Path(file): axum::extract::Path<String>) -> impl IntoResponse {
    match file.as_str() {
        "swagger-ui.css" => swagger_css().await.into_response(),
        "swagger-ui-bundle.js" => swagger_js().await.into_response(),
        _ => (StatusCode::NOT_FOUND, "not found").into_response(),
    }
}

/// OpenAPI 3.0 spec, with the embedded model ids filled in as enums.
pub async fn openapi(State(engine): State<Arc<Engine>>) -> Json<serde_json::Value> {
    let det_ids = engine.det_ids();
    let rec_ids = engine.rec_ids();

    let det_param = |name: &str, desc: &str| {
        serde_json::json!({
            "name": name, "in": "query", "required": false,
            "schema": {"type": "number"}, "description": desc
        })
    };

    Json(serde_json::json!({
      "openapi": "3.0.3",
      "info": {
        "title": "PP-OCRv6 OCR API",
        "version": env!("CARGO_PKG_VERSION"),
        "description": "Pure-Rust (tract) PP-OCRv6 text detection + recognition. Models are embedded in the binary."
      },
      "paths": {
        "/v1/models": {
          "get": {
            "summary": "List embedded detection & recognition models",
            "responses": {"200": {"description": "model lists with detection defaults"}}
          }
        },
        "/v1/ocr": {
          "post": {
            "summary": "Full OCR: detect text regions then recognize them",
            "requestBody": {"required": true, "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}},
            "parameters": [
              {"name": "det_model", "in": "query", "required": det_ids.len() > 1, "schema": {"type": "string", "enum": det_ids}, "description": "detection model id"},
              {"name": "rec_model", "in": "query", "required": rec_ids.len() > 1, "schema": {"type": "string", "enum": rec_ids}, "description": "recognition model id"},
              det_param("thresh", "binarization threshold (default: per-model)"),
              det_param("box_thresh", "min box score (default: per-model)"),
              det_param("unclip_ratio", "box expansion (default: per-model)"),
              {"name": "limit_side_len", "in": "query", "required": false, "schema": {"type": "integer"}, "description": "resize bound, multiple of 32 (default 960)"},
              det_param("min_rec_score", "drop lines below this recognition confidence (default 0)"),
              {"name": "mode", "in": "query", "required": false, "schema": {"type": "string", "enum": ["general", "document", "kenya_id", "kenya_logbook"], "default": "general"}, "description": "OCR post-processing mode: general=raw, document=padded crops and word splitting, kenya_id=document plus Kenyan ID cleanup, kenya_logbook=document plus Kenyan logbook cleanup"},
              {"name": "understand", "in": "query", "required": false, "schema": {"type": "boolean", "default": false}, "description": "Run the structured-understanding stage (Supra-50M, candle) on the OCR text. Requires a binary built with `--features understanding` and mode=kenya_id; the response then carries a `fields` object keyed by serial_number, id_number, full_names, date_of_birth, sex, district_of_birth, place_of_issue, date_of_issue, each as { value, confidence } where confidence is the model's lowest token probability across that field's value (0-1). Otherwise `fields` is null."}
            ],
            "responses": {"200": {"description": "extracted text + per-line boxes/scores; plus a structured `fields` object ({value, confidence} per field) when understand=true (else fields is null)"}}
          }
        },
        "/v1/detect": {
          "post": {
            "summary": "Detection only: text region boxes",
            "requestBody": {"required": true, "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}},
            "parameters": [
              {"name": "model", "in": "query", "required": det_ids.len() > 1, "schema": {"type": "string", "enum": det_ids}, "description": "detection model id (alias: det_model)"},
              det_param("thresh", "binarization threshold"),
              det_param("box_thresh", "min box score"),
              det_param("unclip_ratio", "box expansion"),
              {"name": "limit_side_len", "in": "query", "required": false, "schema": {"type": "integer"}, "description": "resize bound, multiple of 32"}
            ],
            "responses": {"200": {"description": "boxes + scores"}}
          }
        },
        "/v1/recognize": {
          "post": {
            "summary": "Recognition only on a single cropped, upright line image",
            "requestBody": {"required": true, "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}},
            "parameters": [
              {"name": "model", "in": "query", "required": rec_ids.len() > 1, "schema": {"type": "string", "enum": rec_ids}, "description": "recognition model id (alias: rec_model)"}
            ],
            "responses": {"200": {"description": "{ text, score }"}}
          }
        },
        "/health": {"get": {"summary": "Liveness", "responses": {"200": {"description": "ok"}}}}
      }
    }))
}
