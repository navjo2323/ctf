/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <ctf.hpp>

void trace(int const  n,
           CTF_World  *dw){
  int rank, i, num_pes;
  int64_t np;
  double * pairs;
  double tr1, tr2, tr3, tr4;
  int64_t * indices;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  if (rank == 0)
    printf("Starting matrix cyclic trace test\n");

  CTF_Matrix A(n, n, NS, dw);
  CTF_Matrix B(n, n, NS, dw);
  CTF_Matrix C(n, n, NS, dw);
  CTF_Matrix D(n, n, NS, dw);
  CTF_Matrix C1(n, n, NS, dw);
  CTF_Matrix C2(n, n, NS, dw);
  CTF_Matrix C3(n, n, NS, dw);
  CTF_Matrix C4(n, n, NS, dw);
  CTF_Vector DIAG(n, dw);

  srand48(13*rank);

  A.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48();;
  A.write_remote_data(np, indices, pairs);
  free(pairs);
  free(indices);
  B.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48();;
  B.write_remote_data(np, indices, pairs);
  free(pairs);
  free(indices);
  C.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48();;
  C.write_remote_data(np, indices, pairs);
  free(pairs);
  free(indices);
  D.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48();;
  D.write_remote_data(np, indices, pairs);
  free(pairs);
  free(indices);
  

  C1["ij"] = A["ia"]*B["ab"]*C["bc"]*D["cj"];
  C2["ij"] = D["ia"]*A["ab"]*B["bc"]*C["cj"];
  C3["ij"] = C["ia"]*D["ab"]*A["bc"]*B["cj"];
  C4["ij"] = B["ia"]*C["ab"]*D["bc"]*A["cj"];

  DIAG["i"] = C1["ii"];
  tr1 = DIAG.reduce(CTF_OP_SUM);
  DIAG["i"] = C2["ii"];
  tr2 = DIAG.reduce(CTF_OP_SUM);
  DIAG["i"] = C3["ii"];
  tr3 = DIAG.reduce(CTF_OP_SUM);
  DIAG["i"] = C4["ii"];
  tr4 = DIAG.reduce(CTF_OP_SUM);
  
  if (rank == 0){
    printf("tr(ABCD)=%lf, tr(DABC)=%lf, tr(CDAB)=%lf, tr(BCDA)=%lf\n",
            tr1, tr2, tr3, tr4);
    if (fabs(tr1-tr2)/tr1>1.E-6 || fabs(tr2-tr3)/tr2>1.E-6 || fabs(tr3-tr4)/tr3>1.E-6){
      printf("FAILED MATRIX TEST: trace not preserved in cyclic permutation\n");
    } else {
      printf("PASSED MATRIX TEST: trace preserved during cyclic permutation\n");
    }
  }
  
} 


char* getCmdOption(char ** begin,
                 char ** end,
                 const   std::string & option){
char ** itr = std::find(begin, end, option);
if (itr != end && ++itr != end){
  return *itr;
}
return 0;
}


int main(int argc, char ** argv){
  int rank, np, n;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 7;
  } else n = 7;


  CTF_World * dw = new CTF_World();

  if (rank == 0){
    printf("Checking trace calculation n = %d, p = %d:\n",n,np);
  }
  trace(n, dw);

  MPI_Finalize();
  return 0;
 }
