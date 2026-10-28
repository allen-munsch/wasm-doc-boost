//! Does the hybrid's cheap candidate scoring pick the same winner axis as the
//! full-resolution, full-scoring path? Decomposes the hybrid winner error into
//! (a) k=4 radon axis shift and (b) cheap-vs-full scoring.
//!
//! Run:
//!   cargo test --release -p features-core --test winner_agree -- --ignored --nocapture

use features_core::{self as fc};
use std::fs;

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

fn gray_to_rgb(gray: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(gray.len() * 3);
    for &v in gray {
        rgb.push(v);
        rgb.push(v);
        rgb.push(v);
    }
    rgb
}

fn circular_err_deg(a: f64, b: f64) -> f64 {
    let d = (a - b).abs();
    d.min(180.0 - d)
}

#[test]
#[ignore]
fn hybrid_winner_agreement() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/bench_pgms");
    let docs = [0usize, 1, 2];
    let angles: Vec<usize> = (0..180).step_by(10).collect();

    let mut worst_full = 0.0f64;
    let mut sum_full = 0.0f64;
    let mut agree_full = 0usize; // within 1.5 deg
    let mut n = 0usize;
    for &doc in &docs {
        for &a in &angles {
            let pgm = fs::read(format!("{base}/doc{doc}_r{a:03}.pgm")).unwrap();
            let (gray, w, h) = parse_netpbm(&pgm);
            let rgb = gray_to_rgb(&gray);

            // full-res winner
            let full_gray = fc::to_grayscale(&rgb, w, h);
            let full_axes = fc::radon::radon_top_axes(&fc::radon::radon_curve(&full_gray, w, h), 4);
            let full_best = fc::textline::textline_flip_best_axis(&full_gray, w, h, &full_axes);
            let full_deg = full_best.axis.to_degrees().rem_euclid(180.0);

            // hybrid winner (cheap scoring at k4, flip at k2)
            let (core_px, cw, ch) = fc::downsample_rgb(&rgb, w, h, 4);
            let (flip_px, fw, fh) = fc::downsample_rgb(&rgb, w, h, 2);
            let core_gray = fc::to_grayscale(&core_px, cw, ch);
            let flip_gray = fc::to_grayscale(&flip_px, fw, fh);
            let axes4 = fc::radon::radon_top_axes(&fc::radon::radon_curve(&core_gray, cw, ch), 4);
            let hy_best = fc::textline::textline_flip_best_axis_cheap(
                &core_gray, cw, ch, &flip_gray, fw, fh, &axes4,
            );
            let hy_deg = hy_best.axis.to_degrees().rem_euclid(180.0);

            // also: full scoring at k2 with k4 axes (isolate cheap scoring)
            let k2_best = fc::textline::textline_flip_best_axis(&flip_gray, fw, fh, &axes4);
            let k2_deg = k2_best.axis.to_degrees().rem_euclid(180.0);

            let err = circular_err_deg(full_deg, hy_deg);
            sum_full += err;
            if err > worst_full {
                worst_full = err;
            }
            if err <= 1.5 {
                agree_full += 1;
            }
            let err_cheap = circular_err_deg(k2_deg, hy_deg);
            if err_cheap > 1.5 {
                println!(
                    "doc{doc}_r{a:03}: full={full_deg:.1} k2={k2_deg:.1} hy={hy_deg:.1} (cheap-vs-k2 {err_cheap:.1} deg)"
                );
            }
            n += 1;
        }
    }
    println!(
        "\nhybrid winner vs full: mean {:.3} deg, worst {:.3} deg, agree<=1.5deg {}/{} ({:.0}%)\n",
        sum_full / n as f64,
        worst_full,
        agree_full,
        n,
        100.0 * agree_full as f64 / n as f64
    );
}

#[test]
#[ignore]
fn stride_winner_agreement() {
    // Does stride-sampled radon change the FINAL hybrid winner (after cheap
    // textline scoring), against the full-res winner? This is the metric that
    // decides whether stride sampling is safe, not the raw radon top-1.
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/bench_pgms");
    let docs = [0usize, 1, 2];
    let angles: Vec<usize> = (0..180).step_by(10).collect();

    for &s in &[2usize, 3] {
        let mut worst_full = 0.0f64;
        let mut sum_full = 0.0f64;
        let mut agree_full = 0usize;
        let mut n = 0usize;
        for &doc in &docs {
            for &a in &angles {
                let pgm = fs::read(format!("{base}/doc{doc}_r{a:03}.pgm")).unwrap();
                let (gray, w, h) = parse_netpbm(&pgm);
                let rgb = gray_to_rgb(&gray);

                let full_gray = fc::to_grayscale(&rgb, w, h);
                let full_axes = fc::radon::radon_top_axes(&fc::radon::radon_curve(&full_gray, w, h), 4);
                let full_best = fc::textline::textline_flip_best_axis(&full_gray, w, h, &full_axes);
                let full_deg = full_best.axis.to_degrees().rem_euclid(180.0);

                let (core_px, cw, ch) = fc::downsample_rgb(&rgb, w, h, 4);
                let (flip_px, fw, fh) = fc::downsample_rgb(&rgb, w, h, 2);
                let core_gray = fc::to_grayscale(&core_px, cw, ch);
                let flip_gray = fc::to_grayscale(&flip_px, fw, fh);
                let axes_s = fc::radon::radon_top_axes(&fc::radon::radon_curve_strided(&core_gray, cw, ch, s, 1), 4);
                let hy_best = fc::textline::textline_flip_best_axis_cheap(&core_gray, cw, ch, &flip_gray, fw, fh, &axes_s);
                let hy_deg = hy_best.axis.to_degrees().rem_euclid(180.0);

                let err = circular_err_deg(full_deg, hy_deg);
                sum_full += err;
                if err > worst_full {
                    worst_full = err;
                }
                if err <= 1.5 {
                    agree_full += 1;
                }
                n += 1;
            }
        }
        println!(
            "stride={} hybrid winner vs full: mean {:.3} deg, worst {:.3} deg, agree<=1.5deg {}/{} ({:.0}%)",
            s,
            sum_full / n as f64,
            worst_full,
            agree_full,
            n,
            100.0 * agree_full as f64 / n as f64
        );
    }
}
