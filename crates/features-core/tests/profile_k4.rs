//! k4-resolution stage profile of extract_core (find the 16ms hotspots).
//! Run: cargo test --release -p features-core --test profile_k4 -- --ignored --nocapture

use features_core::{self as fc};
use std::fs;
use std::time::Instant;

fn parse_ppm(bytes: &[u8]) -> (Vec<u8>, usize, usize) {
    let mut it = bytes
        .split(|&b| b == b' ' || b == b'\n' || b == b'\r' || b == b'\t')
        .filter(|s| !s.is_empty());
    assert_eq!(it.next().unwrap(), b"P6");
    let mut w = 0usize; let mut h = 0usize; let mut got = 0usize;
    for tok in it {
        if tok[0] == b'#' { continue; }
        let v: usize = std::str::from_utf8(tok).unwrap().parse().unwrap();
        match got { 0 => w = v, 1 => h = v, 2 => { assert_eq!(v, 255); break; }, _ => unreachable!() }
        got += 1;
    }
    let mut i = 0usize; let mut fields = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'#' { while i < bytes.len() && bytes[i] != b'\n' { i += 1; } continue; }
        if bytes[i].is_ascii_whitespace() { i += 1; continue; }
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() { i += 1; }
        fields += 1;
        if fields == 4 { let data = &bytes[i..]; let n = w * h * 3; assert!(data.len() >= n); return (data[..n].to_vec(), w, h); }
    }
    panic!("no pixel data");
}

#[test]
#[ignore]
fn profile_extract_core_k4() {
    let bytes = fs::read("/tmp/profile_512.ppm").unwrap();
    let (pixels, w, h) = parse_ppm(&bytes);
    let (core_px, cw, ch) = fc::downsample_rgb(&pixels, w, h, 4);
    let core_gray = fc::to_grayscale(&core_px, cw, ch);
    println!("\n=== extract_core stage profile @ {cw}x{ch} (k4) ===\n");
    let iters = 50usize;
    let mut t = Instant::now();
    for _ in 0..iters { let _ = fc::to_grayscale(&core_px, cw, ch); }
    println!("{:>34} {:8.3} ms", "to_grayscale", t.elapsed().as_secs_f64()*1e3/iters as f64);

    t = Instant::now();
    for _ in 0..iters { let _ = fc::color::per_channel_stats(&core_px, cw, ch); let _ = fc::color::grayscale_stats(&core_px, cw, ch); let _ = fc::color::colorfulness(&core_px, cw, ch); let _ = fc::color::saturation_stats(&core_px, cw, ch); }
    println!("{:>34} {:8.3} ms", "color (all)", t.elapsed().as_secs_f64()*1e3/iters as f64);

    t = Instant::now();
    for _ in 0..iters { let _ = fc::edges::sobel(&core_gray, cw, ch); }
    println!("{:>34} {:8.3} ms", "sobel", t.elapsed().as_secs_f64()*1e3/iters as f64);

    let sobel = fc::edges::sobel(&core_gray, cw, ch);
    t = Instant::now();
    for _ in 0..iters { let _ = fc::edges::laplacian_variance(&core_gray, cw, ch); let _ = fc::edges::sobel_stats(&sobel); let _ = fc::edges::edge_density(&sobel); let _ = fc::edges::edge_direction_histogram(&sobel); let _ = fc::edges::hv_edge_ratio(&sobel); let _ = fc::edges::canny_edge_density(&sobel); let _ = fc::edges::structure_tensor_features(&sobel); let _ = fc::edges::sobel_circular_stats(&sobel); }
    println!("{:>34} {:8.3} ms", "edges (derived)", t.elapsed().as_secs_f64()*1e3/iters as f64);

    t = Instant::now();
    for _ in 0..iters { let _ = fc::texture::dct_low_freq_ratio(&core_gray, cw, ch); }
    println!("{:>34} {:8.3} ms", "dct_low_freq_ratio", t.elapsed().as_secs_f64()*1e3/iters as f64);

    t = Instant::now();
    for _ in 0..iters { let _ = fc::texture::lbp_histogram(&core_gray, cw, ch); }
    println!("{:>34} {:8.3} ms", "lbp_histogram", t.elapsed().as_secs_f64()*1e3/iters as f64);

    t = Instant::now();
    for _ in 0..iters { let _ = fc::texture::glcm_features(&core_gray, cw, ch); }
    println!("{:>34} {:8.3} ms", "glcm_features", t.elapsed().as_secs_f64()*1e3/iters as f64);

    t = Instant::now();
    for _ in 0..iters { let _ = fc::texture::fractal_dimension(&core_gray, cw, ch); }
    println!("{:>34} {:8.3} ms", "fractal_dimension", t.elapsed().as_secs_f64()*1e3/iters as f64);

    t = Instant::now();
    for _ in 0..iters { let r = fc::noise::high_pass_residual_variance(&core_gray, cw, ch); let _ = fc::noise::jpeg_blockiness(&core_gray, cw, ch); let _ = fc::noise::gradient_snr_given_residual(&core_gray, cw, ch, r); }
    println!("{:>34} {:8.3} ms", "noise (all)", t.elapsed().as_secs_f64()*1e3/iters as f64);

    t = Instant::now();
    for _ in 0..iters { let _ = fc::shadow::shadow_features(&core_gray, cw, ch); }
    println!("{:>34} {:8.3} ms", "shadow_features", t.elapsed().as_secs_f64()*1e3/iters as f64);

    t = Instant::now();
    for _ in 0..iters { let _ = fc::crumple::lbp_variance(&core_gray, cw, ch); let _ = fc::crumple::edge_density_std(&core_gray, cw, ch); let _ = fc::crumple::texture_anisotropy(&core_gray, cw, ch); let _ = fc::crumple::peak_local_entropy(&core_gray, cw, ch); }
    println!("{:>34} {:8.3} ms", "crumple (all)", t.elapsed().as_secs_f64()*1e3/iters as f64);

    t = Instant::now();
    for _ in 0..iters { let _ = fc::document::document_features_gray(&core_gray, &core_px, cw, ch); }
    println!("{:>34} {:8.3} ms", "document_features", t.elapsed().as_secs_f64()*1e3/iters as f64);

    t = Instant::now();
    for _ in 0..iters { let _ = fc::projection::projection_features(&core_gray, cw, ch); }
    println!("{:>34} {:8.3} ms", "projection_features", t.elapsed().as_secs_f64()*1e3/iters as f64);

    t = Instant::now();
    for _ in 0..iters { let _ = fc::ink::ink_features(&core_gray, cw, ch); }
    println!("{:>34} {:8.3} ms", "ink_features", t.elapsed().as_secs_f64()*1e3/iters as f64);
}
