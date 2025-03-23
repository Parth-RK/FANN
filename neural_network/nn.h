#ifndef NN_H_
#define NN_H_


#include<stddef.h>
#include<stdio.h>
#include<math.h>

#ifndef NN_MALLOC
#include<stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include<assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT


// float d[]  ={
    //      0, 0, 0, 
    //      0, 1, 1,
    //      1, 0, 1, 
    //      1, 1, 1,
    // };
    // Mat di = {.rows = 4, .cols = 2, .stride = 3 .es = &d[0]};
    // Mat do = {.rows = 4, .cols = 1, .stride = 3 .es = &d[2]};
    
#define ARRAY_LEN(xs) (sizeof(xs)/sizeof(xs[0]))

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

float rand_float(void);
float sigmoidf(float x);

Mat mat_alloc(size_t rows, size_t cols);
void mat_fill(Mat m, float x);
void mat_rand(Mat m, float low, float high);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_add(Mat dst, Mat a);
void mat_sig(Mat m);
void mat_print(Mat m, const char *name);
#define MAT_PRINT(m) mat_print(m, #m)

typedef struct {
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as; // The amount of activations is count+1
} NN;
NN nn_alloc(size_t *arch, size_t arch_count);
void nn_rand(NN nn, float low, float high);
#define NN_PRINT(nn) nn_print(nn, #nn)

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

float rand_float(void){
    return (float)rand()/(float)RAND_MAX;
}

float sigmoidf(float x){
    return 1.f / (1.f + expf(-x));
}

  Mat mat_alloc(size_t rows, size_t cols){
      Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = NN_MALLOC(sizeof(*m.es)*rows*cols);
    NN_ASSERT(m.es != NULL);
    return m;
}


void mat_fill(Mat m, float x){
    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            MAT_AT(m, i, j) = x;
        }
    }
}


void mat_rand(Mat m, float low, float high){
    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            MAT_AT(m, i, j) = rand_float()*(high - low) + low;
        }
    }
}



Mat mat_row(Mat m, size_t row){
    return (Mat){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m, row, 0)
    };
}
void mat_copy(Mat dst, Mat src){
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);

    for(size_t i = 0; i < dst.rows; i++){
        for(size_t j = 0; j < dst.cols; j++){
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}


void mat_dot(Mat dst, Mat a, Mat b){
    NN_ASSERT(dst.rows == a.rows);
    size_t rows = a.rows;
    NN_ASSERT(dst.cols == b.cols);
    size_t cols = b.cols;
    NN_ASSERT(a.cols == b.rows);

    for(size_t i = 0; i < a.rows; i++){
        for(size_t j = 0; j < b.cols; j++){
            MAT_AT(dst, i, j) = 0;
            for(size_t k = 0; k < a.cols; k++){
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}


void mat_add(Mat dst, Mat a){

    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.cols);
    for(size_t i = 0; i < a.rows; i++){
        for(size_t j = 0; j < a.cols; j++){
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}


void mat_sig(Mat m){
    for(size_t i = 0; i < m.rows; ++i){
        for(size_t j = 0; j < m.cols; ++j){
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}


void mat_print(Mat m, const char *name){
    printf("%s = [\n", name);
    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            printf("    %f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("        ]\n");
}



NN nn_alloc(size_t *arch, size_t arch_count){
    NN nn;
    nn.count = arch_count - 1;
    nn.ws = NN_MALLOC(sizeof(*nn.ws)*nn.count);
    NN_ASSERT(nn.ws != NULL);
    nn.bs = NN_MALLOC(sizeof(*nn.bs)*nn.count);
    NN_ASSERT(nn.bs != NULL);
    nn.as = NN_MALLOC(sizeof(*nn.as)*(nn.count+1));
    NN_ASSERT(nn.as != NULL);

    nn.as[0] = mat_alloc(1, arch[0]);
    for(size_t i = 1; i < arch_count; ++i){
        nn.ws[i-1] = mat_alloc(nn.as[i-1].cols, arch[i]);
        nn.bs[i-1] = mat_alloc(1, arch[i]);
        nn.as[i]   = mat_alloc(1, arch[i]);
    }
    return nn;
}


void nn_rand(NN nn, float low, float high){
    for(size_t i = 0; i < nn.count; ++i){
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}


void nn_print(NN nn, const char *name){
    printf("%s = [\n", name);
    for(size_t i = 0; i < nn.count; ++i){
        printf("    W%ld = [\n", i);
        mat_print(nn.ws[i], "        W");
        printf("    ]\n");
        printf("    B%ld = [\n", i);
        mat_print(nn.bs[i], "        B");
        printf("    ]\n");
    }
    printf("]\n");
}

#endif // NN_IMPLEMENTATION