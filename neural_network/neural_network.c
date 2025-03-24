#include <time.h>

#define NN_IMPLEMENTATION
#include "nn.h"

float td_and [] = {
    0, 0, 0,
    0, 1, 0,
    1, 0, 0,
    1, 1, 1,
};

float td_or [] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 1,
};


float td [] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

float td_sum []  = {
    0, 0,   0, 0,   0, 0,
    0, 0,   0, 1,   0, 1,
    0, 1,   0, 1,   1, 0,
    0, 1,   1, 0,   1, 1,
};


int main(){
    // srand(time(0));   
    
    size_t stride = 3;
    float *td = td_or; // comment this line; by default training data is set to XOR
    size_t n = 4; //sizeof(td)/sizeof(td[0])/stride;


    Mat ti = {
        .rows = n, 
        .cols = 2, 
        .stride = stride, 
        .es = td};
        
    Mat to = {
        .rows = n, 
        .cols = 1, 
        .stride = stride, 
        .es = td + 2};
    
    size_t arch[] = {2, 4, 4, 2, 1}; // NN_architecture 2 input, 4 hidden, 4 hidden, 2 hidden, 1 output
    NN nn = nn_alloc(arch, sizeof(arch)/sizeof(arch[0]));
    NN g = nn_alloc(arch, sizeof(arch)/sizeof(arch[0]));
    nn_rand(nn, 0, 1);

    float eps = 1e-1;
    float rate = 9e-1;

    for(size_t i = 0; i < 1000*50; ++i){
        nn_finite_diff(nn, g, eps, ti, to);
        nn_learn(nn, g, rate);
        printf("%zu: Cost = %f\n", i, nn_cost(nn, ti, to));
    }

    for(size_t i = 0; i < 2; ++i){
        for(size_t j = 0; j < 2; ++j){
            MAT_AT(NN_INPUT(nn), 0, 0) = i;
            MAT_AT(NN_INPUT(nn), 0, 1) = j;
            nn_forward(nn);
            printf("%zu ^ %zu -> %f\n", i, j, MAT_AT(NN_OUTPUT(nn), 0, 0));
        }
    }

    return 0;
}