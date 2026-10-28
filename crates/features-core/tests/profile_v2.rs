//! Stage-by-stage timing of extract_all_v2 on a real 512-long-edge fixture.
//! Mirrors features-core::extract_core + rotation_features exactly.
//! Run with:
//!   cargo test -p features-core --test profile_v2 -- --ignored --nocapture
//! The test binary is std (integration tests link std), so file I/O + timing
//! are fine even though the library is no_std.

use features_core::{self as fc};
use std::fs;
use std::time::Instant;

fn parse_ppm(bytes: &[u8]) -> (Vec<u8>, usize, usize) {
    let mut it = bytes
        .split(|&b| b == b' ' || b == b'\n' || b == b'\r' || b == b'\t')
        .filter(|s| !s.is_empty());
    let magic = it.next().unwrap();
    assert_eq!(magic, b"P6");
    let mut w = 0usize;
    let mut h = 0usize;
    let mut got = 0usize;
    for tok in it {
        if tok[0] == b'#' {
            continue;
        }
        let s = std::str::from_utf8(tok).unwrap();
        let v: usize = s.parse().unwrap();
        match got {
            0 => w = v,
            1 => h = v,
            2 => {
                assert_eq!(v, 255);
                break;
            }
            _ => unreachable!(),
        }
        got += 1;
    }
    // pixel data: after the maxval ws byte
    let mut i = 0usize;
    let mut fields = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'#' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        fields += 1;
        if fields == 4 {
            let _ = start;
            let data = &bytes[i..];
            let n = w * h * 3;
            assert!(data.len() >= n);
            return (data[..n].to_vec(), w, h);
        }
    }
    panic!("no pixel data");
}

#[test]
#[ignore]
fn profile_v2_pipeline() {
    let path = "/tmp/profile_512.ppm";
    let bytes = fs::read(path).expect("run the fixture generator first");
    let (pixels, w, h) = parse_ppm(&bytes);
    println!("\n=== extract_all_v2 stage timing @ {w}x{h} ===\n");

    let mut total = 0.0f64;
    let mut t = Instant::now();
    let gray = fc::to_grayscale(&pixels, w, h);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    total += ms;
    println!("{:>42} {:8.2} ms", "to_grayscale", ms);

    // ---- CORE (100 feats) ----
    t = Instant::now();
    let c1 = fc::color::per_channel_stats(&pixels, w, h);
    let c2 = fc::color::grayscale_stats(&pixels, w, h);
    let c3 = fc::color::colorfulness(&pixels, w, h);
    let c4 = fc::color::saturation_stats(&pixels, w, h);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    total += ms;
    println!("{:>42} {:8.2} ms", "color (all)", ms);

    t = Instant::now();
    let sobel = fc::edges::sobel(&gray, w, h);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    total += ms;
    println!("{:>42} {:8.2} ms", "edges::sobel", ms);

    t = Instant::now();
    let e1 = fc::edges::laplacian_variance(&gray, w, h);
    let e2 = fc::edges::sobel_stats(&sobel);
    let e3 = fc::edges::edge_density(&sobel);
    let e4 = fc::edges::edge_direction_histogram(&sobel);
    let e5 = fc::edges::hv_edge_ratio(&sobel);
    let e6 = fc::edges::canny_edge_density(&sobel);
    let e7 = fc::edges::structure_tensor_features(&sobel);
    let e8 = fc::edges::sobel_circular_stats(&sobel);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    total += ms;
    println!("{:>42} {:8.2} ms", "edges (derived from sobel)", ms);

    t = Instant::now();
    let t1 = fc::texture::dct_low_freq_ratio(&gray, w, h);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    total += ms;
    println!("{:>42} {:8.2} ms", "texture::dct_low_freq_ratio", ms);

    t = Instant::now();
    let t2 = fc::texture::lbp_histogram(&gray, w, h);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    total += ms;
    println!("{:>42} {:8.2} ms", "texture::lbp_histogram", ms);

    t = Instant::now();
    let t3 = fc::texture::glcm_features(&gray, w, h);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    total += ms;
    println!("{:>42} {:8.2} ms", "texture::glcm_features", ms);

    t = Instant::now();
    let t4 = fc::texture::fractal_dimension(&gray, w, h);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    total += ms;
    println!("{:>42} {:8.2} ms", "texture::fractal_dimension", ms);

    t = Instant::now();
    let n1 = fc::noise::high_pass_residual_variance(&gray, w, h);
    let n2 = fc::noise::jpeg_blockiness(&gray, w, h);
    let n3 = fc::noise::gradient_snr_given_residual(&gray, w, h, n1);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    total += ms;
    println!("{:>42} {:8.2} ms", "noise (all)", ms);

    t = Instant::now();
    let s1 = fc::shadow::shadow_features(&gray, w, h);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    total += ms;
    println!("{:>42} {:8.2} ms", "shadow::shadow_features", ms);

    t = Instant::now();
    let cr1 = fc::crumple::lbp_variance(&gray, w, h);
    let cr2 = fc::crumple::edge_density_std(&gray, w, h);
    let cr3 = fc::crumple::texture_anisotropy(&gray, w, h);
    let cr4 = fc::crumple::peak_local_entropy(&gray, w, h);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    total += ms;
    println!("{:>42} {:8.2} ms", "crumple (all)", ms);

    t = Instant::now();
    let d1 = fc::document::document_features_gray(&gray, &pixels, w, h);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    total += ms;
    println!("{:>42} {:8.2} ms", "document::document_features", ms);

    t = Instant::now();
    let p1 = fc::projection::projection_features(&gray, w, h);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    total += ms;
    println!("{:>42} {:8.2} ms", "projection::projection_features", ms);

    t = Instant::now();
    let i1 = fc::ink::ink_features(&gray, w, h);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    total += ms;
    println!("{:>42} {:8.2} ms", "ink::ink_features", ms);

    println!("{:>42} {:8.2} ms", "--- CORE SUBTOTAL ---", total);

    // ---- ROTATION (24 feats) ----
    t = Instant::now();
    let curve = fc::radon::radon_curve(&gray, w, h);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    total += ms;
    println!("{:>42} {:8.2} ms", "radon::radon_curve", ms);

    t = Instant::now();
    let axes = fc::radon::radon_top_axes(&curve, 4);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    total += ms;
    println!("{:>42} {:8.2} ms", "radon::radon_top_axes", ms);

    t = Instant::now();
    let best = fc::textline::textline_flip_best_axis(&gray, w, h, &axes);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    total += ms;
    println!("{:>42} {:8.2} ms", "textline::flip_best_axis (4 deskews)", ms);

    t = Instant::now();
    let (pk, pm, conc) = fc::radon::radon_summary(&curve);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    total += ms;
    println!("{:>42} {:8.2} ms", "radon::radon_summary", ms);

    t = Instant::now();
    let b1 = fc::boundary::boundary_features(&pixels, w, h);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    total += ms;
    println!("{:>42} {:8.2} ms", "boundary::boundary_features", ms);

    t = Instant::now();
    let da = fc::ink::docstrum_axis(&gray, w, h);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    total += ms;
    println!("{:>42} {:8.2} ms", "ink::docstrum_axis", ms);

    println!("{:>42} {:8.2} ms", "=== TOTAL ===", total);

    // sanity: feature counts sum to 124
    let core_n = c1.len() + c2.len() + 1 + c4.len() + 1 + e2.len() + 1 + e4.len()
        + 1 + 1 + e7.len() + e8.len() + 1 + t2.len() + t3.len() + 1 + 1 + n2.len()
        + 1 + s1.len() + cr1.len() + 1 + 1 + 1 + d1.len() + p1.len() + i1.len();
    let rot_n = 1 + 1 + 1 + 1 + 6 + 4 + 2 + 4 + 4;
    let _ = (c3, e1, e3, e5, e6, t1, t4, n1, n3, cr2, cr3, cr4, pk, pm, conc, b1, da);
    let _ = best.features.len();
    println!("core feature count (sanity, expect 100): {core_n}");
    println!("rotation feature count (sanity, expect 24): {rot_n}");
}

#[test]
#[ignore]
fn profile_hybrid() {
    let path = "/tmp/profile_512.ppm";
    let bytes = fs::read(path).expect("run the fixture generator first");
    let (pixels, w, h) = parse_ppm(&bytes);
    println!("\n=== extract_all_v2_hybrid @ {w}x{h} ===\n");

    // warm-up
    let _ = fc::extract_all_v2_hybrid(&pixels, w, h);
    let iters = 20;
    let t = Instant::now();
    let mut acc_logodds = 0.0f64;
    let mut acc_count = 0.0f64;
    for _ in 0..iters {
        let (f, lo, c) = fc::extract_all_v2_hybrid(&pixels, w, h);
        assert_eq!(f.len(), 124);
        acc_logodds += lo;
        acc_count += c;
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    println!(
        "hybrid: {:8.2} ms/image | glyph_count={:.0} | glyph_logodds={:.2}\n",
        ms,
        acc_count / iters as f64,
        acc_logodds / iters as f64
    );
}

#[test]
#[ignore]
fn profile_hybrid_stages() {
    let path = "/tmp/profile_512.ppm";
    let bytes = fs::read(path).expect("run the fixture generator first");
    let (pixels, w, h) = parse_ppm(&bytes);
    println!("\n=== hybrid stage timing @ {w}x{h} ===\n");

    // warm-up
    let _ = fc::extract_all_v2_hybrid(&pixels, w, h);
    let iters = 50;

    let mut t = Instant::now();
    for _ in 0..iters { let _ = fc::downsample_rgb(&pixels, w, h, 4); }
    println!("{:>42} {:8.3} ms", "downsample_rgb k=4", t.elapsed().as_secs_f64() * 1e3 / iters as f64);

    t = Instant::now();
    for _ in 0..iters { let _ = fc::downsample_rgb(&pixels, w, h, 2); }
    println!("{:>42} {:8.3} ms", "downsample_rgb k=2", t.elapsed().as_secs_f64() * 1e3 / iters as f64);

    let (core_px, cw, ch) = fc::downsample_rgb(&pixels, w, h, 4);
    let (flip_px, fw, fh) = fc::downsample_rgb(&pixels, w, h, 2);

    t = Instant::now();
    for _ in 0..iters { let _ = fc::to_grayscale(&core_px, cw, ch); }
    println!("{:>42} {:8.3} ms", "to_grayscale core (k4)", t.elapsed().as_secs_f64() * 1e3 / iters as f64);

    t = Instant::now();
    for _ in 0..iters { let _ = fc::to_grayscale(&flip_px, fw, fh); }
    println!("{:>42} {:8.3} ms", "to_grayscale flip (k2)", t.elapsed().as_secs_f64() * 1e3 / iters as f64);

    t = Instant::now();
    for _ in 0..iters { let _ = fc::extract_core(&core_px, cw, ch); }
    println!("{:>42} {:8.3} ms", "extract_core (k4, 100 feats)", t.elapsed().as_secs_f64() * 1e3 / iters as f64);

    let core_gray = fc::to_grayscale(&core_px, cw, ch);
    let flip_gray = fc::to_grayscale(&flip_px, fw, fh);

    t = Instant::now();
    for _ in 0..iters { let _ = fc::radon::radon_curve(&core_gray, cw, ch); }
    println!("{:>42} {:8.3} ms", "radon_curve (k4)", t.elapsed().as_secs_f64() * 1e3 / iters as f64);

    let curve = fc::radon::radon_curve(&core_gray, cw, ch);
    let axes = fc::radon::radon_top_axes(&curve, 4);

    t = Instant::now();
    for _ in 0..iters { let _ = fc::textline::textline_flip_best_axis_cheap(&core_gray, cw, ch, &flip_gray, fw, fh, &axes); }
    println!("{:>42} {:8.3} ms", "textline flip cheap (k4 score + k2 deskew)", t.elapsed().as_secs_f64() * 1e3 / iters as f64);

    t = Instant::now();
    for _ in 0..iters { let _ = fc::radon::radon_summary(&curve); }
    println!("{:>42} {:8.3} ms", "radon_summary", t.elapsed().as_secs_f64() * 1e3 / iters as f64);

    t = Instant::now();
    for _ in 0..iters { let _ = fc::boundary::boundary_features(&core_px, cw, ch); }
    println!("{:>42} {:8.3} ms", "boundary_features (k4)", t.elapsed().as_secs_f64() * 1e3 / iters as f64);

    t = Instant::now();
    for _ in 0..iters { let _ = fc::ink::docstrum_axis(&core_gray, cw, ch); }
    println!("{:>42} {:8.3} ms", "docstrum_axis (k4)", t.elapsed().as_secs_f64() * 1e3 / iters as f64);

    let best = fc::textline::textline_flip_best_axis_cheap(&core_gray, cw, ch, &flip_gray, fw, fh, &axes);
    t = Instant::now();
    for _ in 0..iters { let _ = fc::glyph::glyph_logodds_from_crops(&best.glyph_crops); }
    println!("{:>42} {:8.3} ms", "glyph_logodds_from_crops", t.elapsed().as_secs_f64() * 1e3 / iters as f64);

    t = Instant::now();
    for _ in 0..iters { let _ = fc::extract_all_v2_hybrid(&pixels, w, h); }
    println!("{:>42} {:8.3} ms", "=== HYBRID TOTAL ===", t.elapsed().as_secs_f64() * 1e3 / iters as f64);
}
