#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NN_IMPLEMENTATION
#include "nn.h"

int get_bit(int x, int pos) {
    return (x >> pos) & 1;
}

float eo_data[16 * 5];

int main() {
    for (int i = 0; i < 16; ++i) {
        eo_data[i*5 + 0] = get_bit(i, 0);
        eo_data[i*5 + 1] = get_bit(i, 1);
        eo_data[i*5 + 2] = get_bit(i, 2);
        eo_data[i*5 + 3] = get_bit(i, 3);
        eo_data[i*5 + 4] = i % 2;
    }

    srand((unsigned)time(0));
    size_t stride = 5;
    size_t n = 16;

    Mat ti = {.rows = n, .cols = 4, .stride = stride, .es = eo_data};
    Mat to = {.rows = n, .cols = 1, .stride = stride, .es = eo_data + 4};

    size_t arch[] = {4, 8, 1};
    NN nn = nn_alloc(arch, sizeof(arch)/sizeof(arch[0]));
    NN g = nn_alloc(arch, sizeof(arch)/sizeof(arch[0]));
    nn_rand(nn, -1, 1);

    float eps = 1e-4;
    float rate = 1e-2;

    for(size_t i = 0; i < 20000; ++i) {
        nn_finite_diff(nn, g, eps, ti, to);
        nn_learn(nn, g, rate);
        if(i % 2000 == 0) {
            printf("%zu: Cost = %f\n", i, nn_cost(nn, ti, to));
        }
    }

    printf("\nPredictions (0-15):\n");
    for(int i = 0; i <= 15; ++i) {
        MAT_AT(NN_INPUT(nn), 0, 0) = get_bit(i, 0);
        MAT_AT(NN_INPUT(nn), 0, 1) = get_bit(i, 1);
        MAT_AT(NN_INPUT(nn), 0, 2) = get_bit(i, 2);
        MAT_AT(NN_INPUT(nn), 0, 3) = get_bit(i, 3);
        nn_forward(nn);
        float out = MAT_AT(NN_OUTPUT(nn), 0, 0);
        printf("%2d (%d%d%d%d) -> %s (%.3f)\n", i, get_bit(i,3), get_bit(i,2), get_bit(i,1), get_bit(i,0), out > 0.5 ? "odd" : "even", out);
    }
    return 0;
}
