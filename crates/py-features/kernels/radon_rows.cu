// radon_rows.cu — CUDA kernel for the projection-variance row sums.
//
// One thread per (angle, row). It reproduces features-core's
// projection_variance_at sampling arithmetic bit-for-bit so the CPU and GPU
// backends produce identical feature vectors (parity), then the HOST runs the
// same mean/variance reduction over the returned row sums.
//
// Per angle the host passes 8 doubles (cos, sin, cx, cy, last_x, last_y,
// step_x, step_y) and 3 ints (nrows, ncols, sy). cos/sin are computed on the
// host with libm so the rotation geometry is bit-identical to the CPU path;
// only the nearest-neighbour sampling + in-order f64 row accumulation run here.

extern "C" __global__ void radon_rows(
    const double* __restrict__ gray,    // size*size padded, row-major
    int size,
    const double* __restrict__ d_angle, // n_angles * 8
    const int*    __restrict__ i_angle, // n_angles * 3
    double*       __restrict__ row_sums,
    const int*    __restrict__ row_off) // n_angles + 1 (cumulative nrows)
{
    const int a = blockIdx.y;
    const double* g = d_angle + (long)a * 8;
    const int*    gi = i_angle + (long)a * 3;
    const int ry = blockIdx.x * blockDim.x + threadIdx.x;

    const int nrows = gi[0];
    if (ry >= nrows) return;
    const int ncols = gi[1];
    const int sy = gi[2];

    const double cos_t = g[0], sin_t = g[1], cx = g[2], cy = g[3];
    const double last_x = g[4], last_y = g[5];
    const double step_x = g[6], step_y = g[7];

    // Matches the CPU loop exactly: y = ry * sy is an INTEGER multiply.
    const int y = ry * sy;
    const double yc = (double)y - cy;
    double xs = cx - cx * cos_t - yc * sin_t;
    double ys = cy - cx * sin_t + yc * cos_t;

    double sum = 0.0;
    for (int c = 0; c < ncols; c++) {
        // (xs + 0.5).clamp(0.0, last_x) as usize, same for y.
        double s = xs + 0.5;
        double cl = s < 0.0 ? 0.0 : (s > last_x ? last_x : s);
        int xi = (int)cl;
        double s2 = ys + 0.5;
        double cl2 = s2 < 0.0 ? 0.0 : (s2 > last_y ? last_y : s2);
        int yi = (int)cl2;
        sum += gray[(long)yi * size + xi];
        xs += step_x;
        ys += step_y;
    }
    row_sums[row_off[a] + ry] = sum;
}
