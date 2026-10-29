//! One-shot profiling: extract_core vs rotation_extract internals.
//! Run: cargo test -p features-core --test prof_split --release -- --nocapture
use std::fs;
use std::time::Instant;

#[test]
fn split_core_vs_rotation() {
    let bytes = fs::read("/tmp/prof.rgb").expect("run the python dump first");
    let (w, h) = (249usize, 512usize);
    assert_eq!(bytes.len(), w * h * 3);
    let gray = features_core::to_grayscale(&bytes, w, h);

    // warmup
    let _ = features_core::extract_core(&bytes, w, h);
    let curve = features_core::radon::radon_curve(&gray, w, h);
    let axes = features_core::radon::radon_top_axes(&curve, 4);
    let _ = features_core::textline::textline_flip_best_axis(&gray, w, h, &axes);
    let coarse = features_core::textline::textline_flip_best_axis(&gray, w, h, &axes);
    let refined = features_core::radon::fine_axis(&gray, w, h, coarse.axis);
    let _ = features_core::textline::textline_flip_at_axis(&gray, w, h, refined);

    let n = 5;
    macro_rules! time {
        ($label:expr, $body:expr) => {{
            let t0 = Instant::now();
            for _ in 0..n {
                std::hint::black_box($body);
            }
            println!(
                "{}: {:.1} ms",
                $label,
                t0.elapsed().as_secs_f64() / n as f64 * 1e3
            );
        }};
    }

    time!(
        "extract_core(100)",
        features_core::extract_core(&bytes, w, h)
    );
    time!(
        "radon_curve",
        features_core::radon::radon_curve(&gray, w, h)
    );
    let curve = features_core::radon::radon_curve(&gray, w, h);
    time!(
        "radon_top_axes",
        features_core::radon::radon_top_axes(&curve, 4)
    );
    let axes = features_core::radon::radon_top_axes(&curve, 4);
    time!(
        "textline_flip_best_axis",
        features_core::textline::textline_flip_best_axis(&gray, w, h, &axes)
    );
    let coarse = features_core::textline::textline_flip_best_axis(&gray, w, h, &axes);
    time!(
        "fine_axis",
        features_core::radon::fine_axis(&gray, w, h, coarse.axis)
    );
    let refined = features_core::radon::fine_axis(&gray, w, h, coarse.axis);
    time!(
        "textline_flip_at_axis",
        features_core::textline::textline_flip_at_axis(&gray, w, h, refined)
    );
    time!("extract_all", features_core::extract_all(&bytes, w, h));
}
