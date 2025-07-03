#include "blas_symbs.h"
#include "util.h"
namespace CTF_BLAS {
  template <typename dtype>
  void gemm(const char *,
            const char *,
            const int *,
            const int *,
            const int *,
            const dtype *,
            const dtype *,
            const int *,
            const dtype *,
            const int *,
            const dtype *,
            dtype *,
            const int *){
    printf("CTF ERROR GEMM not available for this type.\n");
    ASSERT(0);
    assert(0);
  }
#define INST_GEMM(dtype,s)                     \
  template <>                                  \
  void gemm<dtype>(const char * a,             \
            const char * b,                    \
            const int * c,                     \
            const int * d,                     \
            const int * e,                     \
            const dtype * f,                   \
            const dtype * g,                   \
            const int * h,                     \
            const dtype * i,                   \
            const int * j,                     \
            const dtype * k,                   \
            dtype * l,                         \
            const int * m){                    \
    s ## GEMM(a,b,c,d,e,f,g,h,i,j,k,l,m); \
  }
  INST_GEMM(float,S)
  INST_GEMM(double,D)
  INST_GEMM(std::complex<float>,C)
  INST_GEMM(std::complex<double>,Z)
#undef INST_GEMM

  template <typename dtype>
  void gemv(const char *    TRANS,
            const int *     M,
            const int *     N,
            const dtype *   ALPHA,
            dtype *         A,
            const int *     LDA,
            dtype *         X,
            const int *     INCX,
            const dtype *   BETA,
            dtype *         Y,
            const int *     INCY){
    printf("CTF ERROR GEMV not available for this type.\n");
    ASSERT(0);
    assert(0);
  }
#define INST_GEMV(dtype,s)                     \
  template <>                                  \
  void gemv<dtype>(const char * a,             \
            const int * b,                     \
            const int * c,                     \
            const dtype * d,                   \
            dtype * e,                         \
            const int * f,                     \
            dtype * g,                         \
            const int * h,                     \
            const dtype * i,                   \
            dtype * j,                         \
            const int * k){                    \
    s ## GEMV(a,b,c,d,e,f,g,h,i,j,k); \
  }
  INST_GEMV(float,S)
  INST_GEMV(double,D)
  INST_GEMV(std::complex<float>,C)
  INST_GEMV(std::complex<double>,Z)
#undef INST_GEMV

  template <typename dtype>
  void ger(const int *m,
          const int *n,
          const dtype *alpha,
          const dtype *x,
          const int *incx,
          const dtype *y,
          const int *incy,
          dtype *A,
          const int *lda) {
      printf("CTF ERROR: GER not available for this type.\n");
      assert(0);
  }

  #define INST_GER(dtype, prefix)                                      \
  template <>                                                          \
  void ger<dtype>(const int *a,                                        \
                  const int *b,                                        \
                  const dtype *c,                                      \
                  const dtype *d,                                      \
                  const int *e,                                        \
                  const dtype *f,                                      \
                  const int *g,                                        \
                  dtype *h,                                            \
                  const int *i) {                                      \
      prefix##GER(a, b, c, d, e, f, g, h, i);                          \
  }
  INST_GER(float, S)
  INST_GER(double, D)
  #undef INST_GER

  template <typename dtype>
  void syr(const char *       UPLO ,
            const int *        N , 
            const dtype *     ALPHA, 
            const dtype *     X , 
            const int *        INCX , 
            dtype *           A , 
            const int *        LDA ){
     printf("CTF ERROR POSV not available for this type.\n");
    ASSERT(0);
    assert(0);
  }

#define INST_SYR(dtype,s)                     \
  template <>                                  \
  void syr<dtype>(const char * a, \
            const int *    b, \
            const dtype *        c, \
            const dtype *        d, \
            const int *    e, \
            dtype *        f, \
            const int *    g){ \
    s ## SYR(a,b,c,d,e,f,g); \
  }
  INST_SYR(float,S)
  INST_SYR(double,D)
  INST_SYR(std::complex<float>,C)
  INST_SYR(std::complex<double>,Z)
#undef INST_GEMM

  template <typename dtype>
  void syrk(const char *      UPLO ,
            const char *      TRANS,
            const int *        N ,
            const int *        K, 
            const dtype *     ALPHA, 
            const dtype *     A , 
            const int *       LDA ,
            const dtype *    BETA, 
            dtype *           C, 
            const int *        LDC){
     printf("CTF ERROR SYRK not available for this type.\n");
    ASSERT(0);
    assert(0);
  }

#define INST_SYRK(dtype,s)                     \
  template <>                                  \
  void syrk<dtype>(const char *      a ,    \
            const char *      b,           \
            const int *        c ,              \
            const int *        d,               \
            const dtype *     e,            \
            const dtype *     f ,             \
            const int *       g ,           \
            const dtype *    h,            \
            dtype *           i,              \
            const int *       j){ \
    s ## SYRK(a,b,c,d,e,f,g,h,i,j); \
  }
  INST_SYRK(float,S)
  INST_SYRK(double,D)
  INST_SYRK(std::complex<float>,C)
  INST_SYRK(std::complex<double>,Z)
#undef INST_SYRK
      
  template <typename dtype>
  void posv(char const *        UPLO ,
            const int *         N, 
            const int *         NRHS,
            dtype *            A, 
            const int *         LDA, 
            dtype *            B, 
            const int *         LDB, 
            int *               INFO){
     printf("CTF ERROR POSV not available for this type.\n");
    ASSERT(0);
    assert(0);
  }



#define INST_POSV(dtype,s)                     \
  template <>                                  \
  void posv<dtype>(char const * a, \
            const int *    b, \
            const int *    c, \
            dtype *        d, \
            const int *    e, \
            dtype *        f, \
            const int *    g, \
            int *          h){ \
    s ## POSV(a,b,c,d,e,f,g,h); \
  }
  INST_POSV(float,S)
  INST_POSV(double,D)
  INST_POSV(std::complex<float>,C)
  INST_POSV(std::complex<double>,Z)
#undef INST_GEMM



#ifdef USE_BATCH_GEMM
  template <typename dtype>
  void gemm_batch(const char *,
            const char *,
            const int *,
            const int *,
            const int *,
            const dtype *,
            dtype **,
            const int *,
            dtype **,
            const int *,
            const dtype *,
            dtype **,
            const int *,
            const int *,
            const int *){
    printf("CTF ERROR gemm_batch not available for this type.\n");
    ASSERT(0);
    assert(0);
  }

#define INST_GEMM_BATCH(dtype,s)                         \
  template <>                                            \
  void gemm_batch<dtype>(const char * a,                 \
            const char * b,                              \
            const int * c,                               \
            const int * d,                               \
            const int * e,                               \
            const dtype * f,                             \
            dtype ** g,                                  \
            const int * h,                               \
            dtype ** i,                                  \
            const int * j,                               \
            const dtype * k,                             \
            dtype ** l,                                  \
            const int * m,                               \
            const int * n,                               \
            const int * o){                              \
    s ## GEMM_BATCH(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o); \
  }
  INST_GEMM_BATCH(float,S)
  INST_GEMM_BATCH(double,D)
  INST_GEMM_BATCH(std::complex<float>,C)
  INST_GEMM_BATCH(std::complex<double>,Z)
#endif
}
#undef INST_GEMM_BATCH
