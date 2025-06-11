#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NN_IMPLEMENTATION
#include "nn.h"

float eo_data[] = {
    0.0/30.0, 0,
    1.0/30.0, 1,
    2.0/30.0, 0,
    3.0/30.0, 1,
    4.0/30.0, 0,
    5.0/30.0, 1,
    6.0/30.0, 0,
    7.0/30.0, 1,
    8.0/30.0, 0,
    9.0/30.0, 1,
    10.0/30.0, 0,
    11.0/30.0, 1,
    12.0/30.0, 0,
    13.0/30.0, 1,
    14.0/30.0, 0,
    15.0/30.0, 1,
    16.0/30.0, 0,
    17.0/30.0, 1,
    18.0/30.0, 0,
    19.0/30.0, 1,
    20.0/30.0, 0,
    21.0/30.0, 1,
    22.0/30.0, 0,
    23.0/30.0, 1,
    24.0/30.0, 0,
    25.0/30.0, 1,
    26.0/30.0, 0,
    27.0/30.0, 1,
    28.0/30.0, 0,
    29.0/30.0, 1,
    30.0/30.0, 0,
};

int main(){
    srand(time(0));   
    
    size_t stride = 3;
    // float *eo_data = eo_data_or; // comment this line; by default training data is set to XOR
    size_t n = sizeof(eo_data)/sizeof(eo_data[0])/stride;

    Mat ti = {
        .rows = n, 
        .cols = 1, 
        .stride = stride, 
        .es = eo_data};
        
    Mat to = {
        .rows = n, 
        .cols = 1, 
        .stride = stride, 
        .es = eo_data + 1};
    
    size_t arch[] = {1, 4, 4, 1}; // NN_architecture 2 input, 4 hidden, 4 hidden, 2 hidden, 1 output
    NN nn = nn_alloc(arch, sizeof(arch)/sizeof(arch[0]));
    NN g = nn_alloc(arch, sizeof(arch)/sizeof(arch[0]));
    nn_rand(nn, 0, 1);

    float eps = 5e-3;
    float rate = 1e-2;

    for(size_t i = 0; i < 1000*50; ++i){
        nn_finite_diff(nn, g, eps, ti, to);
        nn_learn(nn, g, rate);
        if(i % 1000 == 0) {
            printf("%zu: Cost = %f\n", i, nn_cost(nn, ti, to));
        }
    }

    for(size_t i = 0; i <= 10; ++i){
    MAT_AT(NN_INPUT(nn), 0, 0) = i/30.0f;
    nn_forward(nn);
    printf("%zu -> %s (%.3f)\n", i, MAT_AT(NN_OUTPUT(nn), 0, 0) > 0.5 ? "odd" : "even", MAT_AT(NN_OUTPUT(nn), 0, 0));
}
}