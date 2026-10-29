//! Winner+fine axis error vs COARSE radon resolution (full-res fine kept).
use std::fs;

fn parse_ppm(bytes: &[u8]) -> (Vec<u8>, usize, usize) {
    let mut it = bytes
        .split(|&b| b == b' ' || b == b'\n' || b == b'\r' || b == b'\t')
        .filter(|s| !s.is_empty());
    let _magic = it.next().unwrap();
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
            let n = w * h * 3;
            assert!(data.len() >= n);
            return (data[..n].to_vec(), w, h);
        }
    }
    panic!("no data");
}

fn cerr(a: f64, b: f64) -> f64 {
    let d = (a - b).abs();
    d.min(180.0 - d)
}

fn winner_axis(pixels: &[u8], w: usize, h: usize, ck: usize) -> f64 {
    let full_gray = features_core::to_grayscale(pixels, w, h);
    let (core_px, cw, ch) = features_core::downsample_rgb(pixels, w, h, ck);
    let core_gray = features_core::to_grayscale(&core_px, cw, ch);
    let curve = features_core::radon::radon_curve(&core_gray, cw, ch);
    let axes = features_core::radon::radon_top_axes(&curve, 4);
    let mut coarse = axes[0];
    let mut bq = (0.0f64, 0.0f64);
    for ax in &axes {
        let q = features_core::textline::cheap_line_quality(&core_gray, cw, ch, *ax);
        if q.0 > bq.0 || (q.0 == bq.0 && q.1 > bq.1) {
            coarse = *ax;
            bq = q;
        }
    }
    features_core::radon::fine_axis(&full_gray, w, h, coarse)
}

#[test]
fn coarse_res_error() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/bench_pgms");
    let angles: Vec<usize> = (0..180).step_by(10).collect();
    for &ck in &[1usize, 2, 4] {
        let mut sum = 0.0;
        let mut worst = 0.0;
        let mut n = 0;
        for doc in 0..3usize {
            for &a in &angles {
                let ppm = fs::read(format!("{base}/doc{doc}_r{a:03}.ppm")).unwrap();
                let (rgb, w, h) = parse_ppm(&ppm);
                let ax = winner_axis(&rgb, w, h, ck).to_degrees().rem_euclid(180.0);
                let gt = ((180 - a) % 180) as f64;
                let e = cerr(ax, gt);
                sum += e;
                if e > worst {
                    worst = e;
                }
                n += 1;
            }
        }
        println!(
            "coarse k={ck}: winner+fine axis MAE {:.3} deg, worst {:.3} deg (n={n})",
            sum / n as f64,
            worst
        );
    }
}
