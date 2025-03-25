// using basic functions of nn.h
#define NN_IMPLEMENTATION
#include "nn.h"
#include <time.h>

int main(){
    srand(time(0));
    Mat a = mat_alloc(1, 2);
    mat_rand(a, 5, 10);


    float id_data[4] = {
        1, 0,
        0, 1,
    };
    Mat b = {.rows = 2, .cols = 2, .es = id_data};
    // mat_rand(b, 5, 10);
    Mat dst = mat_alloc(1, 2);

    mat_print(a, "A");
    mat_print(b, "B");
    printf("----------------------------------\n");
    mat_dot(dst, a, b);
    mat_print(dst, "C = A * B");

    mat_add(dst, a);
    mat_print(dst, "A + C");
    
    return 0;
}