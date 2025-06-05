#include<stdio.h>
#include<stdlib.h>
#include<time.h>

float train[][2] = {
  {0, 0},
  {1, 1},
  {2, 4},
  {3, 6},
  {4, 8},
  {5, 10},
};

#define train_count (sizeof(train) / sizeof(train[0]))

float rand_float(void){
  return (float) rand() / (float) RAND_MAX;
}

float cost(float w){

  float result = 0.0f;
  
  for(size_t i = 0; i < train_count; ++i){
    float x = train [i][0];
    float y = x*w;

    float d = y - train[i][1];
    result += d*d;
  }
  result /= train_count;
  return result;
}

float dcost(float w){

    float result = 0.0f;
    size_t n = train_count;
    for(size_t i=0; i<n; ++i){
        float x = train [i][0];
        float y = train [i][1];
        result += 2*(x*w - y)*x;
        result /= n;
    }
    return result;
}

int main(){
  
    // srand(time(0));
    srand(69);

    float w = rand_float()*10.0f;
    float rate = 5e-2;
    int iter = 50;

    printf("cost = %f, w = %f\n", cost(w), w);
    for(size_t i = 0; i < iter; ++i){
#if 0
        float eps = 1e-3;
        float c = cost(w);
        float dw = (cost(w + eps) - c)/eps;  // uses finite diff instead of actual derivative
#else        
        float dw = dcost(w);
#endif    
        w -= rate*dw;
        printf("cost = %f, w = %f\n", cost(w), w);
    }
    printf("-----------------------------\n");
    printf("%f\n", w);
    return 0;
    }
