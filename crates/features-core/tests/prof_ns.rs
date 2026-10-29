//! noise/shadow cost + how much k=2/k=4 downsampling perturbs the 9 features
use std::fs;
use std::time::Instant;

fn gray_of(rgb: &[u8], w: usize, h: usize) -> Vec<f64> {
    features_core::to_grayscale(rgb, w, h)
}

#[test]
fn noise_shadow_cost() {
    for (name, w, h) in [("fast", 362usize, 512usize), ("slow", 512usize, 288usize)] {
        let bytes = fs::read(format!("/tmp/{name}.rgb")).unwrap();
        let full = gray_of(&bytes, w, h);
        let (k2, w2, h2) = features_core::downsample_rgb(&bytes, w, h, 2);
        let (k4, w4, h4) = features_core::downsample_rgb(&bytes, w, h, 4);
        let g2 = gray_of(&k2, w2, h2);
        let g4 = gray_of(&k4, w4, h4);
        macro_rules! t {
            ($label:expr, $body:expr) => {{
                let t0 = Instant::now();
                for _ in 0..20 {
                    std::hint::black_box($body);
                }
                println!(
                    "  {}: {:.1} ms",
                    $label,
                    t0.elapsed().as_secs_f64() / 20.0 * 1e3
                );
            }};
        }
        println!("{name} {w}x{h}:");
        t!(
            "ns full-res",
            features_core::extract_noise_shadow(&full, w, h)
        );
        t!("ns k=2", features_core::extract_noise_shadow(&g2, w2, h2));
        t!("ns k=4", features_core::extract_noise_shadow(&g4, w4, h4));
        let f1 = features_core::extract_noise_shadow(&full, w, h);
        let f2 = features_core::extract_noise_shadow(&g2, w2, h2);
        let f4 = features_core::extract_noise_shadow(&g4, w4, h4);
        println!(
            "  full: {:?}",
            f1.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>()
        );
        println!(
            "  k2  : {:?}",
            f2.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>()
        );
        println!(
            "  k4  : {:?}",
            f4.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>()
        );
    }
}
