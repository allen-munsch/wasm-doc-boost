//! Coarse radon axis error vs stride on real invoice fixtures (full-res).
use std::fs;

fn parse_pgm(bytes: &[u8]) -> (Vec<f64>, usize, usize) {
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
            let n = w * h;
            assert!(data.len() >= n);
            return (data[..n].iter().map(|&b| b as f64).collect(), w, h);
        }
    }
    panic!("no data");
}

fn cerr(a: f64, b: f64) -> f64 {
    let d = (a - b).abs();
    d.min(180.0 - d)
}

#[test]
#[ignore]
fn stride_axis_error() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/bench_pgms");
    let angles: Vec<usize> = (0..180).step_by(10).collect();
    let strides = [(1usize, 1usize), (2, 1), (3, 1), (4, 1), (2, 2), (1, 2)];
    for &(sx, sy) in &strides {
        let mut sum = 0.0;
        let mut worst = 0.0;
        let mut n = 0;
        for doc in 0..3usize {
            for &a in &angles {
                let pgm = fs::read(format!("{base}/doc{doc}_r{a:03}.pgm")).unwrap();
                let (gray, w, h) = parse_pgm(&pgm);
                let curve = features_core::radon::radon_curve_strided(&gray, w, h, sx, sy);
                let axes = features_core::radon::radon_top_axes(&curve, 4);
                let top = axes[0].to_degrees().rem_euclid(180.0);
                let gt = ((180 - a) % 180) as f64;
                let e = cerr(top, gt);
                sum += e;
                if e > worst {
                    worst = e;
                }
                n += 1;
            }
        }
        println!(
            "stride ({sx},{sy}): top-1 coarse axis MAE {:.3} deg, worst {:.3} deg (n={n})",
            sum / n as f64,
            worst
        );
    }
}
