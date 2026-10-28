//! Resolution-scaling validation for the sub-10ms downsample path.
//!
//! The 0-vs-180 flip decision is made by the per-glyph classifier, not the
//! flip_score feature: `glyph_sum_logodds` positive = flipped, negative =
//! upright. This harness checks whether that decision keeps its sign at
//! k=2/k=4 versus full res, and reports the per-image time at each scale.
//!
//! Run:
//!   cargo test --release -p features-core --test scale_validate -- --ignored --nocapture

use features_core::{self as fc};
use std::fs;
use std::time::Instant;

fn parse_netpbm(bytes: &[u8]) -> (Vec<u8>, usize, usize) {
    let mut it = bytes
        .split(|&b| b == b' ' || b == b'\n' || b == b'\r' || b == b'\t')
        .filter(|s| !s.is_empty());
    let magic = it.next().unwrap();
    assert_eq!(magic, b"P5");
    let mut w = 0usize;
    let mut h = 0usize;
    let mut got = 0usize;
    for tok in it {
        if tok[0] == b'#' {
            continue;
        }
        let v: usize = std::str::from_utf8(tok).unwrap().parse().unwrap();
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
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        fields += 1;
        if fields == 4 {
            let data = &bytes[i..];
            let n = w * h;
            assert!(data.len() >= n);
            return (data[..n].to_vec(), w, h);
        }
    }
    panic!("no pixel data");
}

fn flip180_gray(gray: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = gray[(h - 1 - y) * w + (w - 1 - x)];
        }
    }
    out
}

fn gray_to_rgb(gray: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(gray.len() * 3);
    for &v in gray {
        rgb.push(v);
        rgb.push(v);
        rgb.push(v);
    }
    rgb
}

fn sign(f: f64) -> i8 {
    if f > 0.0 {
        1
    } else if f < 0.0 {
        -1
    } else {
        0
    }
}

#[test]
#[ignore]
fn glyph_flip_and_time_vs_scale() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/bench_pgms");
    let docs = [0usize, 1, 2];
    let angles: Vec<usize> = (0..180).step_by(10).collect();
    let ks = [1usize, 2, 4];
    const HYBRID: usize = 3;

    let mut time = [0.0f64; 4];
    let mut n_images = 0usize;
    // separation_ok: upright glyph logodds < 0 AND flipped > 0 (correct flip)
    let mut separation_ok = [0usize; 4];
    // sign agreement vs k=1
    let mut agree_up = [0usize; 4];
    let mut agree_dn = [0usize; 4];

    for &doc in &docs {
        for &a in &angles {
            let pgm = fs::read(format!("{base}/doc{doc}_r{a:03}.pgm")).unwrap();
            let (gray, w, h) = parse_netpbm(&pgm);
            let rgb = gray_to_rgb(&gray);
            let flipped = gray_to_rgb(&flip180_gray(&gray, w, h));

            let mut up = [0.0f64; 4];
            let mut dn = [0.0f64; 4];
            for (ki, &k) in ks.iter().enumerate() {
                let (d_up, uw, uh) = fc::downsample_rgb(&rgb, w, h, k);
                let (d_dn, dw, dh) = fc::downsample_rgb(&flipped, w, h, k);
                let t = Instant::now();
                let (_, u_logodds, _) = fc::extract_all_v2_with_glyph(&d_up, uw, uh);
                let (_, d_logodds, _) = fc::extract_all_v2_with_glyph(&d_dn, dw, dh);
                time[ki] += t.elapsed().as_secs_f64() * 1e3;
                up[ki] = u_logodds;
                dn[ki] = d_logodds;
            }
            {
                let t = Instant::now();
                let (_, u_logodds, _) = fc::extract_all_v2_hybrid(&rgb, w, h);
                let (_, d_logodds, _) = fc::extract_all_v2_hybrid(&flipped, w, h);
                time[HYBRID] += t.elapsed().as_secs_f64() * 1e3;
                up[HYBRID] = u_logodds;
                dn[HYBRID] = d_logodds;
            }

            for ki in 0..4 {
                if up[ki] < 0.0 && dn[ki] > 0.0 {
                    separation_ok[ki] += 1;
                }
                if ki > 0 && sign(up[ki]) == sign(up[0]) {
                    agree_up[ki] += 1;
                }
                if ki > 0 && sign(dn[ki]) == sign(dn[0]) {
                    agree_dn[ki] += 1;
                }
            }
            n_images += 1;
        }
    }

    println!(
        "\n=== glyph flip decision + time vs scale ({n_images} doc/angle pairs) ==="
    );
    for (ki, &k) in ks.iter().enumerate() {
        let ms = time[ki] / (2.0 * n_images as f64);
        println!(
            "k={} : {:7.2} ms/image | upright<0 & flipped>0: {}/{} ({:.0}%) | sign agree up: {}/{} | sign agree dn: {}/{}",
            k,
            ms,
            separation_ok[ki],
            n_images,
            100.0 * separation_ok[ki] as f64 / n_images as f64,
            agree_up[ki],
            n_images,
            agree_dn[ki],
            n_images
        );
    }
    let ms = time[HYBRID] / (2.0 * n_images as f64);
    println!(
        "hybrid : {:7.2} ms/image | upright<0 & flipped>0: {}/{} ({:.0}%) | sign agree up: {}/{} | sign agree dn: {}/{}",
        ms,
        separation_ok[HYBRID],
        n_images,
        100.0 * separation_ok[HYBRID] as f64 / n_images as f64,
        agree_up[HYBRID],
        n_images,
        agree_dn[HYBRID],
        n_images
    );
}
