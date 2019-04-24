__kernel void get_convolution(__global float* a, __global float* b, __global float* c, int n, int m) {
    int id = get_global_id(0);
    if (id >= n * n)
        return;
    int row = id / n;
    int column = id % n;
    int hm = (m - 1) / 2;
    c[id] = 0.0;

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