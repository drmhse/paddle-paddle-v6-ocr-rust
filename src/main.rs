//! PP-OCRv6 text-detection REST API (pure-Rust ONNX inference via tract).

mod api;
mod config;
mod docs;
mod embedded;
mod engine;
mod geometry;
mod postprocess;
mod preprocess;
mod recognize;
mod remote;
#[cfg(feature = "understanding")]
mod understanding;

use anyhow::Result;
use clap::Parser;
use std::net::SocketAddr;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(name = "ppocr-server", about = "PP-OCRv6 OCR REST API (detection + recognition, embedded models)")]
struct Args {
    /// Bind address.
    #[arg(long, env = "OCR_HOST", default_value = "0.0.0.0")]
    host: String,

    /// Bind port.
    #[arg(long, env = "OCR_PORT", default_value_t = 8080)]
    port: u16,

    /// How many distinct input sizes to keep compiled per model (LRU).
    #[arg(long, env = "OCR_PLAN_CACHE", default_value_t = 16)]
    plan_cache: usize,

    /// Skip background pre-warming of recognition plans at startup.
    #[arg(long, env = "OCR_NO_PREWARM")]
    no_prewarm: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();

    let engine = Arc::new(engine::Engine::load(args.plan_cache)?);
    tracing::info!(det = ?engine.det_ids(), rec = ?engine.rec_ids(), "models ready");

    #[cfg(feature = "understanding")]
    {
        understanding::init()?;
        tracing::info!("understanding model ready (Supra-50M, candle)");
    }

    if !args.no_prewarm {
        let e = engine.clone();
        tokio::task::spawn_blocking(move || e.prewarm());
    }

    let app = api::router(engine);
    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutting down");
}
