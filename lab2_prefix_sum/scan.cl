__kernel void scan_blelloch(__global float * a, __global float * r, __local float * b, __global float * sm, int max_gid)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);
    uint gr_id = get_group_id(0);

    if (gid >= max_gid)
        return;
    uint dp = 1;

    b[lid] = a[gid];

    for(uint s = block_size>>1; s > 0; s >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lid < s)
        {
            uint i = dp*(2*lid+1)-1;
            uint j = dp*(2*lid+2)-1;
            b[j] += b[i];
        }

        dp <<= 1;
    }

    if(lid == 0) b[block_size - 1] = 0;

    for(uint s = 1; s < block_size; s <<= 1)
    {
        dp >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if(lid < s)
        {
            uint i = dp*(2*lid+1)-1;
            uint j = dp*(2*lid+2)-1;

            float t = b[j];
            b[j] += b[i];
            b[i] = t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    r[gid] = b[lid];

    if (lid == block_size - 1)
        sm[gr_id] = a[gid] + b[lid];
}

__kernel void sum_with_inter(__global float * output, __global float * s, int max_gid)
{
    int gid = get_global_id(0);
    int group_id = get_group_id(0);
    if (gid >= max_gid)
        return;
    output[gid] += s[group_id];
}

