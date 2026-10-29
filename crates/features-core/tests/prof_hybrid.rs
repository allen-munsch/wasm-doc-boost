//! One-shot: hybrid stage timing on slow (photo) vs fast (text) images.
//! Run: cargo test -p features-core --test prof_hybrid --release -- --nocapture
use std::fs;
use std::time::Instant;

fn run(name: &str, w: usize, h: usize) {
    let bytes = fs::read(format!("/tmp/{name}.rgb")).unwrap();
    assert_eq!(bytes.len(), w * h * 3);
    let full_gray = features_core::to_grayscale(&bytes, w, h);
    let (core_px, cw, ch) = features_core::downsample_rgb(&bytes, w, h, 4);
    let (flip_px, fw, fh) = features_core::downsample_rgb(&bytes, w, h, 2);
    let core_gray = features_core::to_grayscale(&core_px, cw, ch);
    let flip_gray = features_core::to_grayscale(&flip_px, fw, fh);

    macro_rules! t {
        ($label:expr, $body:expr) => {{
            let t0 = Instant::now();
            for _ in 0..10 {
                std::hint::black_box($body);
            }
            println!(
                "  {}: {:.1} ms",
                $label,
                t0.elapsed().as_secs_f64() / 10.0 * 1e3
            );
        }};
    }

    println!("{name} {w}x{h}:");
    t!(
        "downsample k4+k2",
        features_core::downsample_rgb(&bytes, w, h, 4)
    );
    t!(
        "extract_core_hybrid",
        features_core::extract_core_hybrid(&bytes, w, h)
    );
    t!(
        "to_grayscale full",
        features_core::to_grayscale(&bytes, w, h)
    );
    t!(
        "radon strided(2,1)",
        features_core::radon::radon_curve_strided(&full_gray, w, h, 2, 1)
    );
    let curve = features_core::radon::radon_curve_strided(&full_gray, w, h, 2, 1);
    let axes = features_core::radon::radon_top_axes(&curve, 4);
    let mut coarse = axes[0];
    t!("cheap_line_quality x4", {
        let mut b = (0.0f64, 0.0f64);
        for ax in &axes {
            let q = features_core::textline::cheap_line_quality(&core_gray, cw, ch, *ax);
            if q.0 > b.0 {
                b = q;
                coarse = *ax;
            }
        }
    });
    t!(
        "fine_axis",
        features_core::radon::fine_axis(&full_gray, w, h, coarse)
    );
    let refined = features_core::radon::fine_axis(&full_gray, w, h, coarse);
    t!(
        "textline_flip_at_axis k2",
        features_core::textline::textline_flip_at_axis(&flip_gray, fw, fh, refined)
    );
    t!(
        "extract_all (hybrid)",
        features_core::extract_all(&bytes, w, h)
    );
}

#[test]
fn hybrid_stage_timing() {
    run("fast", 362, 512);
    run("slow", 512, 288);
}
