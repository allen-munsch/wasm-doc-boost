//! stride timing: (1,1) vs (2,1) vs (2,2) same image
use std::fs;
use std::time::Instant;

#[test]
fn stride_timing() {
    let bytes = fs::read("/tmp/fast.rgb").unwrap();
    let (w, h) = (362usize, 512usize);
    let gray = features_core::to_grayscale(&bytes, w, h);
    macro_rules! t {
        ($label:expr, $body:expr) => {{
            let t0 = Instant::now();
            for _ in 0..20 {
                std::hint::black_box($body);
            }
            println!(
                "{}: {:.1} ms",
                $label,
                t0.elapsed().as_secs_f64() / 20.0 * 1e3
            );
        }};
    }
    t!(
        "radon (1,1)",
        features_core::radon::radon_curve_strided(&gray, w, h, 1, 1)
    );
    t!(
        "radon (2,1)",
        features_core::radon::radon_curve_strided(&gray, w, h, 2, 1)
    );
    t!(
        "radon (1,2)",
        features_core::radon::radon_curve_strided(&gray, w, h, 1, 2)
    );
    t!(
        "radon (2,2)",
        features_core::radon::radon_curve_strided(&gray, w, h, 2, 2)
    );
    t!(
        "radon (3,1)",
        features_core::radon::radon_curve_strided(&gray, w, h, 3, 1)
    );
    t!(
        "radon (4,1)",
        features_core::radon::radon_curve_strided(&gray, w, h, 4, 1)
    );
}
