__kernel void get_convolution(__global float* a, __global float* b, __global float* c, int n, int m) {
    int row = get_global_id(0);
    int column = get_global_id(1);
    if (row >= n || column >= n)
        return;
    int hm = (m - 1) / 2;
    c[row * n + column] = 0.0;

    for (int k = -hm; k <= hm; k++) {
        if (row + k < 0 || row + k >= n)
            continue;
        for (int l = -hm; l <= hm; l++) {
            if (column + l < 0 || column + l >= n)
                continue;
            c[row * n + column] += a[(row + k) * n + column + l] * b[(k + hm) * m + l + hm];
        }
    }
}