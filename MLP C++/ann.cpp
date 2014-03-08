#include "ann.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
   
   ann::ann(int *LS, int n, float bs, float bta, float Nu){
      LayerSizes = LS;  // Number of nodes per layer
      N = n;            // Size of LayerSizes array
      bias = bs;        // Bias term
      beta = bta;       // Beta term used in the sigmoid function
      nu = Nu;          // Learning rate
      
      int j,NN;
      
      srand (time(NULL));
      
      // Build the weights
      NN = 0;
      for (j = 0; j < N-1; j++)
         NN += LayerSizes[j]*LayerSizes[j+1];
      
      w = (float*) calloc(NN, sizeof(float));
      
      for (j = 0; j < NN; j++)
         w[j] = ((float) rand()) / ((float) RAND_MAX) * 2.0f - 1.0f;

      // Build the bias weights
      NN = 0;
      for (j = 1; j < N; j++)
         NN += LayerSizes[j];
      
      v = (float*) calloc(NN, sizeof(float));
      
      for (j = 0; j < NN; j++)
         v[j] = ((float) rand()) / ((float) RAND_MAX) * 2.0f - 1.0f;
      
      // Build the layers
      NN = 0;
      for (j = 0; j < N; j++)
         NN += LayerSizes[j];
      
      x = (float*) calloc(NN, sizeof(float));
      
      // Build the back propagation error arrays
      NN = 0;
      for (j = 0; j < N; j++)
         NN += LayerSizes[j];
      
      d = (float*) calloc(NN, sizeof(float));

   }
   
   ann::~ann(){}

  /*****************************************************
   * Sigmoid function
   *
   *                 1
   * g(z) =  --------------------
   *          1 + exp(-beta * z)
   *
   *****************************************************/
   
   float ann::g(float z, float beta){
      return 1.0f / (1.0f + exp(-beta * z));
   }
   
   /**************************************************
    * Apply an MxN matrix to the N dimensional vector
    * b
    *
    * y = A*b
    **************************************************/
   
   void ann::mult(float *y, float *A, float *b, int M, int N){
   
      int i,j,ind;
      
      for (i = 0, ind = 0; i < M; i++){
         y[i] = 0;
         for (j = 0; j < N; j++, ind++)
            y[i] += A[ind] * b[j];
      }
   }
   
  /****************************************************
   * Apply the transpose of an MxN matrix to the
   * M dimensional vector b
   *
   * y = A'*b
   ****************************************************/
   
   void ann::Tmult(float *y, float *A, float *b, int M, int N){
   
      float At[N][M];
      int i,j,ind;
      
      // Transpose A
      for (i = 0, ind = 0; i < M; i++)
         for (j = 0; j < N; j++, ind++)
            At[j][i] = A[ind];
      
      mult(y, &At[0][0], b, N, M);
   }
   
  /**************************************************
   * Apply the outer product of vector x of length M
   * and vector y of length N and put the result into
   * vector z.
   **************************************************/
   
   void ann::outer(float *z, float *x, float *y, int M, int N){
   
      int i,j,ind;
      
      for (i = 0, ind = 0; i < M; i++)
         for (j = 0; j < N; j++, ind++)
            z[ind] = x[i]*y[j];
   
   }
