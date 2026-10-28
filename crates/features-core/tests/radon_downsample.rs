//! Does the radon axis survive downsampling? Compare full-res vs box-downsampled
//! gray on real rotated invoice fixtures (data/bench_pgms/doc{r}_r{a}.pgm).
//! Run: cargo test --release -p features-core --test radon_downsample -- --ignored --nocapture

use features_core::radon::{radon_curve, radon_curve_strided, radon_top_axes};
use std::fs;
use std::time::Instant;

fn parse_pgm(bytes: &[u8]) -> (Vec<f64>, usize, usize) {
    let (raw, w, h) = parse_netpbm(bytes, 1);
    (raw.iter().map(|&b| b as f64).collect(), w, h)
}

fn parse_netpbm(bytes: &[u8], channels: usize) -> (Vec<u8>, usize, usize) {
    let mut it = bytes
        .split(|&b| b == b' ' || b == b'\n' || b == b'\r' || b == b'\t')
        .filter(|s| !s.is_empty());
    let magic = it.next().unwrap();
    let expected = if channels == 1 { b"P5" } else { b"P6" };
    assert_eq!(magic, expected);
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
            let n = w * h * channels;
            assert!(data.len() >= n);
            return (data[..n].to_vec(), w, h);
        }
    }
    panic!("no pixel data");
}

/// Box downsample by integer factor `k` (average).
fn box_downsample(gray: &[f64], w: usize, h: usize, k: usize) -> (Vec<f64>, usize, usize) {
    let nw = w / k;
    let nh = h / k;
    let mut out = vec![0.0f64; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let mut acc = 0.0f64;
            for dy in 0..k {
                for dx in 0..k {
                    acc += gray[(y * k + dy) * w + (x * k + dx)];
                }
            }
            out[y * nw + x] = acc / (k * k) as f64;
        }
    }
    (out, nw, nh)
}

fn circular_err_deg(a: f64, b: f64) -> f64 {
    let d = (a - b).abs();
    d.min(180.0 - d)
}

#[test]
#[ignore]
fn radon_axis_vs_downsample() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/bench_pgms");
    let docs = [0usize, 1, 2];
    let angles: Vec<usize> = (0..180).step_by(10).collect();
    let ks = [2usize, 4, 8];

    for &k in &ks {
        let mut n = 0usize;
        let mut t_full = 0.0f64;
        let mut t_down = 0.0f64;
        let mut sum_min_delta = 0.0f64; // best circular match, full axis -> nearest down axis
        let mut top1_match = 0usize; // top-1 (by curve value) matches within 1.5 deg
        let mut worst_min_delta = 0.0f64;
        for doc in docs {
            for &a in &angles {
                let pgm = fs::read(format!("{base}/doc{doc}_r{a:03}.pgm")).unwrap();
                let (gray, w, h) = parse_pgm(&pgm);

                let t = Instant::now();
                let full_axes = radon_top_axes(&radon_curve(&gray, w, h), 4);
                t_full += t.elapsed().as_secs_f64() * 1e3;

                let (dgray, dw, dh) = box_downsample(&gray, w, h, k);
                let t = Instant::now();
                let down_axes = radon_top_axes(&radon_curve(&dgray, dw, dh), 4);
                t_down += t.elapsed().as_secs_f64() * 1e3;

                let fdeg: Vec<f64> = full_axes.iter().map(|r| r.to_degrees().rem_euclid(180.0)).collect();
                let ddeg: Vec<f64> = down_axes.iter().map(|r| r.to_degrees().rem_euclid(180.0)).collect();

                // for each full axis (rank order), min circular distance to any down axis
                let mut min_delta = f64::INFINITY;
                for &f in &fdeg {
                    let mut m = f64::INFINITY;
                    for &d in &ddeg {
                        m = m.min(circular_err_deg(f, d));
                    }
                    min_delta = min_delta.min(m);
                }
                sum_min_delta += min_delta;
                if min_delta > worst_min_delta {
                    worst_min_delta = min_delta;
                }
                if circular_err_deg(fdeg[0], ddeg[0]) <= 1.5 {
                    top1_match += 1;
                }
                n += 1;
            }
        }
        println!(
            "k={} : mean best-match delta {:.3} deg, worst {:.3} deg, top-1 agree {}/{} ({:.0}%), radon full {:.1} ms, radon down {:.1} ms",
            k,
            sum_min_delta / n as f64,
            worst_min_delta,
            top1_match,
            n,
            100.0 * top1_match as f64 / n as f64,
            t_full / n as f64,
            t_down / n as f64
        );
    }
}

#[test]
#[ignore]
fn radon_axis_vs_stride() {
    // Does stride sampling inside radon_curve preserve the axis? Compare
    // stride-1 vs stride-{2,3,4} on the k=4 core image (the exact hybrid
    // radon input) over real rotated invoice fixtures.
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/bench_pgms");
    let docs = [0usize, 1, 2];
    let angles: Vec<usize> = (0..180).step_by(10).collect();
    let strides = [2usize, 3, 4];

    for &s in &strides {
        let mut n = 0usize;
        let mut sum_min_delta = 0.0f64;
        let mut top1_match = 0usize;
        let mut worst_min_delta = 0.0f64;
        for doc in docs {
            for &a in &angles {
                let pgm = fs::read(format!("{base}/doc{doc}_r{a:03}.pgm")).unwrap();
                let (gray, w, h) = parse_pgm(&pgm);
                let (core, cw, ch) = box_downsample(&gray, w, h, 4);
                let full_axes = radon_top_axes(&radon_curve(&core, cw, ch), 4);
                let str_axes = radon_top_axes(&radon_curve_strided(&core, cw, ch, s, 1), 4);
                let fdeg: Vec<f64> = full_axes.iter().map(|r| r.to_degrees().rem_euclid(180.0)).collect();
                let sdeg: Vec<f64> = str_axes.iter().map(|r| r.to_degrees().rem_euclid(180.0)).collect();
                let mut min_delta = f64::INFINITY;
                for &f in &fdeg {
                    let mut m = f64::INFINITY;
                    for &d in &sdeg {
                        m = m.min(circular_err_deg(f, d));
                    }
                    min_delta = min_delta.min(m);
                }
                sum_min_delta += min_delta;
                if min_delta > worst_min_delta {
                    worst_min_delta = min_delta;
                }
                if circular_err_deg(fdeg[0], sdeg[0]) <= 1.5 {
                    top1_match += 1;
                }
                n += 1;
            }
        }
        println!(
            "stride={} : mean best-match delta {:.3} deg, worst {:.3} deg, top-1 agree {}/{} ({:.0}%)",
            s,
            sum_min_delta / n as f64,
            worst_min_delta,
            top1_match,
            n,
            100.0 * top1_match as f64 / n as f64
        );
    }
}
