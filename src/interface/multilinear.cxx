#include "vector.h"
#include "timer.h"
#include "../mapping/mapping.h"
#include "../shared/blas_symbs.h"

namespace CTF {
  template<typename dtype>
  void TTTP(Tensor<dtype> * T, int num_ops, int const * modes, Tensor<dtype> ** mat_list, bool aux_mode_first){
    Timer t_tttp("TTTP");
    t_tttp.start();
    int k = -1;
    bool is_vec = mat_list[0]->order == 1;
    if (!is_vec)
      k = mat_list[0]->lens[1-aux_mode_first];
    dtype ** arrs = (dtype**)malloc(sizeof(dtype*)*num_ops);
    int64_t * ldas = (int64_t*)malloc(num_ops*sizeof(int64_t));
    int * op_lens = (int*)malloc(num_ops*sizeof(int));
    int * phys_phase = (int*)malloc(T->order*sizeof(int));
    int * mat_strides = NULL;
    if (!is_vec)
      mat_strides = (int*)malloc(2*num_ops*sizeof(int));
    for (int i=0; i<T->order; i++){
      phys_phase[i] = T->edge_map[i].calc_phys_phase();
    }

    int64_t npair;
    Pair<dtype> * pairs;
    if (T->is_sparse){
      pairs = (Pair<dtype>*)T->data;
      npair = T->nnz_loc;
    } else
      T->get_local_pairs(&npair, &pairs, true, false);

    for (int i=0; i<num_ops; i++){
      //printf("i=%d/%d %d %d %d\n",i,num_ops,modes[i],mat_list[i]->lens[aux_mode_first], T->lens[modes[i]]);
      if (i>0) IASSERT(modes[i] > modes[i-1] && modes[i]<T->order);
      if (is_vec){
        IASSERT(mat_list[i]->order == 1);
      } else {
        IASSERT(mat_list[i]->order == 2);
        IASSERT(mat_list[i]->lens[1-aux_mode_first] == k);
        IASSERT(mat_list[i]->lens[aux_mode_first] == T->lens[modes[i]]);
      }
      int last_mode = 0;
      if (i>0) last_mode = modes[i-1];
      op_lens[i] = T->lens[modes[i]];///phys_phase[modes[i]];
      ldas[i] = 1;//phys_phase[modes[i]];
      for (int j=last_mode; j<modes[i]; j++){
        ldas[i] *= T->lens[j];
      }
/*      if (i>0){
        ldas[i] = ldas[i] / phys_phase[modes[i-1]];
      }*/
    }

    int64_t max_memuse = CTF_int::proc_bytes_available();
    int64_t tot_size = 0;
    int div = 1;
    if (is_vec){
      for (int i=0; i<num_ops; i++){
        tot_size += mat_list[i]->lens[0]/phys_phase[modes[i]];
      }
      if (tot_size*(int64_t)sizeof(dtype) > max_memuse){
        printf("CTF ERROR: insufficeint memory for TTTP");
      }
    } else {
      //div = 2;
      do {
        tot_size = 0;
        int kd = (k+div-1)/div;
        for (int i=0; i<num_ops; i++){
          tot_size += 2*mat_list[i]->lens[aux_mode_first]*kd/phys_phase[modes[i]];
        }
        if (div > 1)
          tot_size += npair;
        //if (T->wrld->rank == 0)
        //  printf("tot_size = %ld max_memuse = %ld\n", tot_size*(int64_t)sizeof(dtype), max_memuse);
        if (tot_size*(int64_t)sizeof(dtype) > max_memuse){
          if (div == k){
            printf("CTF ERROR: insufficeint memory for TTTP");
            IASSERT(0);
            assert(0);
          } else
            div = std::min(div*2, k);
        } else
          break;
      } while(true);
    }
    MPI_Allreduce(MPI_IN_PLACE, &div, 1, MPI_INT, MPI_MAX, T->wrld->comm);
    //if (T->wrld->rank == 0)
    //  printf("In TTTP, chosen div is %d\n",div);
    dtype * acc_arr = NULL;
    if (!is_vec && div>1){
      acc_arr = (dtype*)T->sr->alloc(npair);
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (int64_t i=0; i<npair; i++){
        acc_arr[i] = 0.;
      }
    } 
    Tensor<dtype> ** redist_mats = (Tensor<dtype>**)malloc(sizeof(Tensor<dtype>*)*num_ops);
    Partition par(T->topo->order, T->topo->lens);
    char * par_idx = (char*)malloc(sizeof(char)*T->topo->order);
    for (int i=0; i<T->topo->order; i++){
      par_idx[i] = 'a'+i+1;
    }
    char mat_idx[2];
    int slice_st[2];
    int slice_end[2];
    int k_start = 0;
    int kd = 0;
    for (int d=0; d<div; d++){
      k_start += kd;
      kd = k/div + (d < k%div);
      int k_end = k_start + kd;

      for (int i=0; i<num_ops; i++){
        Tensor<dtype> mmat;
        Tensor<dtype> * mat = mat_list[i];
        if (div>1){
          if (aux_mode_first){
            slice_st[0] = k_start;
            slice_st[1] = 0;
            slice_end[0] = k_end;
            slice_end[1] = T->lens[modes[i]];
            mat_strides[2*i+0] = kd;
            mat_strides[2*i+1] = 1;
          } else {
            slice_st[1] = k_start;
            slice_st[0] = 0;
            slice_end[1] = k_end;
            slice_end[0] = T->lens[modes[i]];
            mat_strides[2*i+0] = 1;
            mat_strides[2*i+1] = T->lens[modes[i]];
          }
          mmat = mat_list[i]->slice(slice_st, slice_end);
          mat = &mmat;
        } else if(!is_vec) {
          if (aux_mode_first){
            mat_strides[2*i+0] = k;
            mat_strides[2*i+1] = 1;
          } else {
            mat_strides[2*i+0] = 1;
            mat_strides[2*i+1] = T->lens[modes[i]];
          }
        }

        if (phys_phase[modes[i]] == 1){
          if (is_vec)
            arrs[i] = (dtype*)T->sr->alloc(T->lens[modes[i]]);
          else
            arrs[i] = (dtype*)T->sr->alloc(T->lens[modes[i]]*kd);
          mat->read_all(arrs[i], true);
          redist_mats[i] = NULL;
        } else {
          int nrow, ncol;
          int topo_dim = T->edge_map[modes[i]].cdt;
          IASSERT(T->edge_map[modes[i]].type == CTF_int::PHYSICAL_MAP);
          IASSERT(!T->edge_map[modes[i]].has_child || T->edge_map[modes[i]].child->type != CTF_int::PHYSICAL_MAP);
          int comm_lda = 1;
          for (int l=0; l<topo_dim; l++){
            comm_lda *= T->topo->dim_comm[l].np;
          }
          CTF_int::CommData cmdt(T->wrld->rank-comm_lda*T->topo->dim_comm[topo_dim].rank,T->topo->dim_comm[topo_dim].rank,T->wrld->cdt);
          if (is_vec){
            Vector<dtype> * v = new Vector<dtype>(mat_list[i]->lens[0], par_idx[topo_dim], par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            v->operator[]("i") += mat_list[i]->operator[]("i");
            redist_mats[i] = v;
            arrs[i] = (dtype*)v->data;
            cmdt.bcast(v->data,v->size,T->sr->mdtype(),0);
          } else {
            if (aux_mode_first){
              nrow = kd;
              ncol = T->lens[modes[i]];
              mat_idx[0] = 'a';
              mat_idx[1] = par_idx[topo_dim];
            } else {
              nrow = T->lens[modes[i]];
              ncol = kd;
              mat_idx[0] = par_idx[topo_dim];
              mat_idx[1] = 'a';
            }
            Matrix<dtype> * m = new Matrix<dtype>(nrow, ncol, mat_idx, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            m->operator[]("ij") += mat->operator[]("ij");
            redist_mats[i] = m;
            arrs[i] = (dtype*)m->data;

            cmdt.bcast(m->data,m->size,T->sr->mdtype(),0);
            if (aux_mode_first){
              mat_strides[2*i+0] = kd;
              mat_strides[2*i+1] = 1;
            } else {
              mat_strides[2*i+0] = 1;
              mat_strides[2*i+1] = m->pad_edge_len[0]/phys_phase[modes[i]];
            }
          }
        }
        
      }
      //if (T->wrld->rank == 0)
      //  printf("Completed redistribution in TTTP\n");
  #ifdef _OPENMP
      #pragma omp parallel
  #endif
      {
        if (is_vec){
  #ifdef _OPENMP
          #pragma omp for
  #endif
          for (int64_t i=0; i<npair; i++){
            int64_t key = pairs[i].k;
            for (int j=0; j<num_ops; j++){
              //printf("i=%ld, j=%d\n",i,j);
              key = key/ldas[j];
              //FIXME: handle general semiring
              pairs[i].d *= arrs[j][(key%op_lens[j])/phys_phase[modes[j]]];
            }
          }
        } else {
          int * inds = (int*)malloc(num_ops*sizeof(int));
  #ifdef _OPENMP
          #pragma omp for
  #endif
          for (int64_t i=0; i<npair; i++){
            int64_t key = pairs[i].k;
            for (int j=0; j<num_ops; j++){
              key = key/ldas[j];
              inds[j] = (key%op_lens[j])/phys_phase[j];
            }
            dtype acc = 0;
            for (int kk=0; kk<kd; kk++){
              dtype a = arrs[0][inds[0]*mat_strides[0]+kk*mat_strides[1]];
              for (int j=1; j<num_ops; j++){
                a *= arrs[j][inds[j]*mat_strides[2*j]+kk*mat_strides[2*j+1]];
              }
              acc += a;
            }
            if (acc_arr == NULL)
              pairs[i].d *= acc;
            else
              acc_arr[i] += acc;
          }
          free(inds);
        }
      }
      for (int j=0; j<num_ops; j++){
        if (redist_mats[j] != NULL){
          if (redist_mats[j]->data != (char*)arrs[j])
            T->sr->dealloc((char*)arrs[j]);
          delete redist_mats[j];
        } else
          T->sr->dealloc((char*)arrs[j]);
      }
    }
    if (acc_arr != NULL){
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (int64_t i=0; i<npair; i++){
        pairs[i].d *= acc_arr[i];
      }
      T->sr->dealloc((char*)acc_arr);
    }

    if (!T->is_sparse){
      T->write(npair, pairs);
      T->sr->pair_dealloc((char*)pairs);
    }
    //if (T->wrld->rank == 0)
    //  printf("Completed TTTP\n");
    free(redist_mats);
    if (mat_strides != NULL) free(mat_strides);
    free(par_idx);
    free(phys_phase);
    free(ldas);
    free(op_lens);
    free(arrs);
    t_tttp.stop();
    
  }


  template<typename dtype>
  void svd(Tensor<dtype> & dA, char const * idx_A, Idx_Tensor const & U, Idx_Tensor const & S, Idx_Tensor const & VT, int rank, double threshold, bool use_rand_svd, int num_iter, int oversamp){
    bool need_transpose_A  = false;
    bool need_transpose_U  = false;
    bool need_transpose_VT = false;
    IASSERT(strlen(S.idx_map) == 1);
    int ndim_U = strlen(U.idx_map);
    int ndim_VT = strlen(VT.idx_map);
    IASSERT(ndim_U+ndim_VT-2 == dA.order);
    int nrow_U = 1;
    int ncol_VT = 1;
    char aux_idx = S.idx_map[0];
    if (U.idx_map[ndim_U-1] != aux_idx)
      need_transpose_U = true;
    if (VT.idx_map[0] != aux_idx)
      need_transpose_VT = true;
    char * unf_idx_A = (char*)malloc(sizeof(char)*(dA.order));
    int iA = 0;
    int idx_aux_U;
    int idx_aux_VT;
    for (int i=0; i<ndim_U; i++){
      if (U.idx_map[i] != aux_idx){
        unf_idx_A[iA] = U.idx_map[i];
        iA++;
      } else idx_aux_U = i;
    }
    for (int i=0; i<ndim_VT; i++){
      if (VT.idx_map[i] != aux_idx){
        unf_idx_A[iA] = VT.idx_map[i];
        iA++;
      } else idx_aux_VT = i;
    }
    int * unf_lens_A = (int*)malloc(sizeof(int)*(dA.order));
    int * unf_lens_U = (int*)malloc(sizeof(int)*(ndim_U));
    int * unf_lens_VT = (int*)malloc(sizeof(int)*(ndim_VT));
    int * lens_U = (int*)malloc(sizeof(int)*(ndim_U));
    int * lens_VT = (int*)malloc(sizeof(int)*(ndim_VT));
    for (int i=0; i<dA.order; i++){
      if (idx_A[i] != unf_idx_A[i]){
        need_transpose_A = true;
      }
      int match = 0;
      for (int j=0; j<dA.order; j++){
        if (idx_A[j] == unf_idx_A[i]){
          match++;
          unf_lens_A[i] = dA.lens[j];
          if (i<ndim_U-1){
            unf_lens_U[i] = unf_lens_A[i];
            nrow_U *= unf_lens_A[i];
          } else {
            unf_lens_VT[i-ndim_U+2] = unf_lens_A[i];
            ncol_VT *= unf_lens_A[i];
          }
        }
      }
      IASSERT(match==1);
      
    }
    Matrix<dtype> A(nrow_U, ncol_VT, SP*dA.is_sparse, *dA.wrld, *dA.sr);
    if (need_transpose_A){
      Tensor<dtype> T(dA.order, dA.is_sparse, unf_lens_A, *dA.wrld, *dA.sr);
      T[unf_idx_A] += dA.operator[](idx_A);
      A.reshape(T);
    } else {
      A.reshape(dA);
    }
    Matrix<dtype> tU, tVT;
    Vector<dtype> tS;
    if (use_rand_svd){
      A.svd_rand(tU, tS, tVT, rank, num_iter, oversamp);
    } else {
      A.svd(tU, tS, tVT, rank, threshold);
    }
    (*(Tensor<dtype>*)S.parent) = tS;
    int fin_rank = tS.lens[0];
    unf_lens_U[ndim_U-1] = fin_rank;
    unf_lens_VT[0] = fin_rank;
    char * unf_idx_U = (char*)malloc(sizeof(char)*(ndim_U));
    char * unf_idx_VT = (char*)malloc(sizeof(char)*(ndim_VT));
    unf_idx_U[ndim_U-1] = aux_idx;
    unf_idx_VT[0] = aux_idx;
    lens_U[idx_aux_U] = fin_rank;
    lens_VT[idx_aux_VT] = fin_rank;
    for (int i=0; i<ndim_U; i++){
      if (i<idx_aux_U){
        lens_U[i] = unf_lens_U[i];
        unf_idx_U[i] = U.idx_map[i];
      }
      if (i>idx_aux_U){
        lens_U[i] = unf_lens_U[i-1];
        unf_idx_U[i-1] = U.idx_map[i];
      }
    }
    for (int i=0; i<ndim_VT; i++){
      if (i<idx_aux_VT){
        lens_VT[i] = unf_lens_VT[i+1];
        unf_idx_VT[i+1] = VT.idx_map[i];
      }
      if (i>idx_aux_VT){
        lens_VT[i] = unf_lens_VT[i];
        unf_idx_VT[i] = VT.idx_map[i];
      }
    }
    if (need_transpose_U){
      Tensor<dtype> TU(ndim_U, unf_lens_U, *dA.wrld, *dA.sr);
      TU.reshape(tU);
      (*(Tensor<dtype>*)U.parent) = Tensor<dtype>(ndim_U, lens_U, *dA.wrld, *dA.sr);
      U.parent->operator[](U.idx_map) += U.parent->operator[](U.idx_map);
      U.parent->operator[](U.idx_map) += TU[unf_idx_U];
    } else {
      (*(Tensor<dtype>*)U.parent) = Tensor<dtype>(ndim_U, unf_lens_U, *dA.wrld, *dA.sr);
      ((Tensor<dtype>*)U.parent)->reshape(tU);
    }
    if (need_transpose_VT){
      Tensor<dtype> TVT(ndim_VT, unf_lens_VT, *dA.wrld, *dA.sr);
      TVT.reshape(tVT);
      (*(Tensor<dtype>*)VT.parent) = Tensor<dtype>(ndim_VT, lens_VT, *dA.wrld, *dA.sr);
      VT.parent->operator[](VT.idx_map) += TVT[unf_idx_VT];
    } else {
      (*(Tensor<dtype>*)VT.parent) = Tensor<dtype>(ndim_VT, unf_lens_VT, *dA.wrld, *dA.sr);
      ((Tensor<dtype>*)VT.parent)->reshape(tVT);
    }
    free(unf_lens_A);
    free(unf_lens_U);
    free(unf_lens_VT);
    free(unf_idx_A);
    free(unf_idx_U);
    free(unf_idx_VT);
    free(lens_U);
    free(lens_VT);
  }

  template<typename dtype>
  void MTTKRP(Tensor<dtype> * T, Tensor<dtype> ** mat_list, int mode, bool aux_mode_first){
    Timer t_mttkrp("MTTKRP");
    t_mttkrp.start();
    int k = -1;
    bool is_vec = mat_list[0]->order == 1;
    if (!is_vec)
      k = mat_list[0]->lens[1-aux_mode_first];
    IASSERT(mode >= 0 && mode < T->order);
    for (int i=0; i<T->order; i++){
      IASSERT(is_vec || T->lens[i] == mat_list[i]->lens[aux_mode_first]);
      IASSERT(!mat_list[i]->is_sparse);
    }
    dtype ** arrs = (dtype**)malloc(sizeof(dtype*)*T->order);
    int64_t * ldas = (int64_t*)malloc(T->order*sizeof(int64_t));
    int * phys_phase = (int*)malloc(T->order*sizeof(int));
    int * mat_strides = NULL;
    if (!is_vec)
      mat_strides = (int*)malloc(2*T->order*sizeof(int));
    for (int i=0; i<T->order; i++){
      phys_phase[i] = T->edge_map[i].calc_phys_phase();
    }

    int64_t npair;
    Pair<dtype> * pairs;
    if (T->is_sparse){
      pairs = (Pair<dtype>*)T->data;
      npair = T->nnz_loc;
    } else
      T->get_local_pairs(&npair, &pairs, true, false);

    ldas[0] = 1;
    for (int i=1; i<T->order; i++){
      ldas[i] = ldas[i-1] * T->lens[i-1];
    }

    Tensor<dtype> ** redist_mats = (Tensor<dtype>**)malloc(sizeof(Tensor<dtype>*)*T->order);
    Partition par(T->topo->order, T->topo->lens);
    char * par_idx = (char*)malloc(sizeof(char)*T->topo->order);
    for (int i=0; i<T->topo->order; i++){
      par_idx[i] = 'a'+i+1;
    }
    char mat_idx[2];
    int slice_st[2];
    int slice_end[2];
    int k_start = 0;
    int kd = 0;
    int div = 1;
    for (int d=0; d<div; d++){
      k_start += kd;
      kd = k/div + (d < k%div);
      int k_end = k_start + kd;

      Timer t_mttkrp_remap("MTTKRP_remap_mats");
      t_mttkrp_remap.start();
      for (int i=0; i<T->order; i++){
        Tensor<dtype> mmat;
        Tensor<dtype> * mat = mat_list[i];

        int64_t tot_sz;
        if (is_vec)
          tot_sz = T->lens[i];
        else
          tot_sz = T->lens[i]*kd;
        if (div>1){
          if (aux_mode_first){
            slice_st[0] = k_start;
            slice_st[1] = 0;
            slice_end[0] = k_end;
            slice_end[1] = T->lens[i];
            mat_strides[2*i+0] = kd;
            mat_strides[2*i+1] = 1;
          } else {
            slice_st[1] = k_start;
            slice_st[0] = 0;
            slice_end[1] = k_end;
            slice_end[0] = T->lens[i];
            mat_strides[2*i+0] = 1;
            mat_strides[2*i+1] = T->lens[i];
          }
          mmat = mat_list[i]->slice(slice_st, slice_end);
          mat = &mmat;
        } else if (!is_vec) {
          if (aux_mode_first){
            mat_strides[2*i+0] = k;
            mat_strides[2*i+1] = 1;
          } else {
            mat_strides[2*i+0] = 1;
            mat_strides[2*i+1] = T->lens[i];
          }
        }
        int nrow, ncol;
        if (aux_mode_first){
          nrow = kd;
          ncol = T->lens[i];
        } else {
          nrow = T->lens[i];
          ncol = kd;
        }
        if (phys_phase[i] == 1){
          redist_mats[i] = NULL;
          if (T->wrld->np == 1){
            IASSERT(div == 1);
            arrs[i] = (dtype*)mat_list[i]->data;
            if (i == mode)
              std::fill(arrs[i], arrs[i]+mat_list[i]->size, *((dtype*)T->sr->addid()));
          } else if (i != mode){
            arrs[i] = (dtype*)T->sr->alloc(tot_sz);
            mat->read_all(arrs[i], true);
          } else {
            if (is_vec)
              redist_mats[i] = new Vector<dtype>(mat_list[i]->lens[0], 'a'-1, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            else {
              char nonastr[2];
              nonastr[0] = 'a'-1;
              nonastr[1] = 'a'-2;
              redist_mats[i] = new Matrix<dtype>(nrow, ncol, nonastr, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            }
            arrs[i] = (dtype*)redist_mats[i]->data;
          }
        } else {
          int topo_dim = T->edge_map[i].cdt;
          IASSERT(T->edge_map[i].type == CTF_int::PHYSICAL_MAP);
          IASSERT(!T->edge_map[i].has_child || T->edge_map[i].child->type != CTF_int::PHYSICAL_MAP);
          if (aux_mode_first){
            mat_idx[0] = 'a';
            mat_idx[1] = par_idx[topo_dim];
          } else {
            mat_idx[0] = par_idx[topo_dim];
            mat_idx[1] = 'a';
          }

          int comm_lda = 1;
          for (int l=0; l<topo_dim; l++){
            comm_lda *= T->topo->dim_comm[l].np;
          }
          CTF_int::CommData cmdt(T->wrld->rank-comm_lda*T->topo->dim_comm[topo_dim].rank,T->topo->dim_comm[topo_dim].rank,T->wrld->cdt);
          if (is_vec){
            Vector<dtype> * v = new Vector<dtype>(mat_list[i]->lens[0], par_idx[topo_dim], par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            if (i != mode)
              v->operator[]("i") += mat_list[i]->operator[]("i");
            redist_mats[i] = v;
            arrs[i] = (dtype*)v->data;
            if (i != mode)
              cmdt.bcast(v->data,v->size,T->sr->mdtype(),0);
          } else {
            Matrix<dtype> * m = new Matrix<dtype>(nrow, ncol, mat_idx, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            if (i != mode)
              m->operator[]("ij") += mat->operator[]("ij");
            redist_mats[i] = m;
            arrs[i] = (dtype*)m->data;

            if (i != mode)
              cmdt.bcast(m->data,m->size,T->sr->mdtype(),0);
            if (aux_mode_first){
              mat_strides[2*i+0] = kd;
              mat_strides[2*i+1] = 1;
            } else {
              mat_strides[2*i+0] = 1;
              mat_strides[2*i+1] = m->pad_edge_len[0]/phys_phase[i];
            }
          }
        }
        
      }
      t_mttkrp_remap.stop();

      Timer t_mttkrp_work("MTTKRP_work");
      t_mttkrp_work.start();
      {
        if (!is_vec){
          ((Semiring<dtype>*)T->sr)->MTTKRP(T->order, T->lens, phys_phase, kd, npair, mode, aux_mode_first, pairs, arrs, arrs[mode]);
        } else {
          ((Semiring<dtype>*)T->sr)->MTTKRP(T->order, T->lens, phys_phase, npair, mode, pairs, arrs, arrs[mode]);
          //if (is_vec){
          //  for (int64_t i=0; i<npair; i++){
          //    int64_t key = pairs[i].k;
          //    dtype d = pairs[i].d;
          //    for (int j=0; j<T->order; j++){
          //      if (j != mode){
          //        int64_t ke = key/ldas[j];
          //        d *= arrs[j][(ke%T->lens[j])/phys_phase[j]];
          //      }
          //    }
          //    int64_t ke = key/ldas[mode];
          //    arrs[mode][(ke%T->lens[mode])/phys_phase[mode]] += d;
          //  }
          //} else {
          //  int * inds = (int*)malloc(T->order*sizeof(int));
          //  for (int64_t i=0; i<npair; i++){
          //    int64_t key = pairs[i].k;
          //    for (int j=0; j<T->order; j++){
          //      int64_t ke = key/ldas[j];
          //      inds[j] = (ke%T->lens[j])/phys_phase[j];
          //    }
          //    for (int kk=0; kk<kd; kk++){
          //      dtype d = pairs[i].d;
          //      //dtype a = arrs[0][inds[0]*mat_strides[0]+kk*mat_strides[1]];
          //      for (int j=0; j<T->order; j++){
          //        if (j != mode)
          //          d *= arrs[j][inds[j]*mat_strides[2*j]+kk*mat_strides[2*j+1]];
          //      }
          //      arrs[mode][inds[mode]*mat_strides[2*mode]+kk*mat_strides[2*mode+1]] += d;
          //    }
          //  }
          //  free(inds);
          //}
        }
      }
      t_mttkrp_work.stop();
      for (int j=0; j<T->order; j++){
        if (j == mode){
          int red_len = T->wrld->np/phys_phase[j];
          if (red_len > 1){
            int64_t sz;
            if (redist_mats[j] == NULL){
              if (is_vec)
                sz = T->lens[j];
              else
                sz = T->lens[j]*kd;
            } else {
              sz = redist_mats[j]->size;
            }
            int jr = T->edge_map[j].calc_phys_rank(T->topo);
            MPI_Comm cm;
            MPI_Comm_split(T->wrld->comm, jr, T->wrld->rank, &cm);
            int cmr;
            MPI_Comm_rank(cm, &cmr);

            Timer t_mttkrp_red("MTTKRP_Reduce");
            t_mttkrp_red.start();
            if (cmr == 0)
              MPI_Reduce(MPI_IN_PLACE, arrs[j], sz, T->sr->mdtype(), T->sr->addmop(), 0, cm);
            else {
              MPI_Reduce(arrs[j], NULL, sz, T->sr->mdtype(), T->sr->addmop(), 0, cm);
              std::fill(arrs[j], arrs[j]+sz, *((dtype*)T->sr->addid()));
            }
            t_mttkrp_red.stop();
            MPI_Comm_free(&cm);
          }
          if (redist_mats[j] != NULL){
            mat_list[j]->set_zero();
            mat_list[j]->operator[]("ij") += redist_mats[j]->operator[]("ij");
            delete redist_mats[j];
          } else {
            IASSERT((dtype*)mat_list[j]->data == arrs[j]);
          }
        } else {
          if (redist_mats[j] != NULL){
            if (redist_mats[j]->data != (char*)arrs[j])
              T->sr->dealloc((char*)arrs[j]);
            delete redist_mats[j];
          } else {
            if (arrs[j] != (dtype*)mat_list[j]->data)
              T->sr->dealloc((char*)arrs[j]);
          }
        }
      }
    }
    free(redist_mats);
    if (mat_strides != NULL) free(mat_strides);
    free(par_idx);
    free(phys_phase);
    free(ldas);
    free(arrs);
    if (!T->is_sparse)
      T->sr->pair_dealloc((char*)pairs);
    t_mttkrp.stop();
  }
  

  template<typename dtype>
  void Solve_Factor(Tensor<dtype> * T, Tensor<dtype> ** mat_list, Tensor<dtype> * RHS, int mode, Tensor <dtype>* regu, double epsilon, double barrier, bool proj, bool add_ones, bool aux_mode_first){
    // Mode defines what factor index we're computing

    // Following code to check the distributions are the same for nnz 
    IASSERT(T->order == RHS->order) ;
    int64_t npair ;
    int64_t npair_RHS;
    Pair<dtype> * pairs ;
    Pair<dtype> * pairs_RHS;
    if (T->is_sparse){
      IASSERT(RHS->is_sparse);
      pairs = (Pair<dtype>*)T->data;
      npair = T->nnz_loc;
      pairs_RHS = (Pair<dtype>*)RHS->data;
      npair_RHS = RHS->nnz_loc;
    } else{
      T->get_local_pairs(&npair, &pairs, true, false);
      RHS->get_local_pairs(&npair_RHS, &pairs_RHS, true, false);
    }
    IASSERT(npair == npair_RHS);
    for (int i=0; i<T->order; i++){
      IASSERT(T->edge_map[i].calc_phys_phase() == RHS->edge_map[i].calc_phys_phase());
    }
    Timer t_solve_factor("Solve_Factor");
    t_solve_factor.start();
    int k = -1;
    bool is_vec = mat_list[0]->order == 1;
    if (!is_vec)
      k = mat_list[0]->lens[1-aux_mode_first];
    IASSERT(mode >= 0 && mode < T->order);
    for (int i=0; i<T->order; i++){
      IASSERT(is_vec || T->lens[i] == mat_list[i]->lens[aux_mode_first]);
      IASSERT(!mat_list[i]->is_sparse);
    }
    dtype ** arrs = (dtype**)malloc(sizeof(dtype*)*T->order);
    dtype * regu_arr;
    int64_t * ldas = (int64_t*)malloc(T->order*sizeof(int64_t));
    int * phys_phase = (int*)malloc(T->order*sizeof(int));
    int * mat_strides = NULL;
    if (!is_vec)
      mat_strides = (int*)malloc(2*T->order*sizeof(int));
    for (int i=0; i<T->order; i++){
      phys_phase[i] = T->edge_map[i].calc_phys_phase();
    }

    ldas[0] = 1;
    for (int i=1; i<T->order; i++){
      ldas[i] = ldas[i-1] * T->lens[i-1];
    }

    Tensor<dtype> ** redist_mats = (Tensor<dtype>**)malloc(sizeof(Tensor<dtype>*)*T->order);
    Tensor<dtype> * redist_regu = NULL;

    Partition par(T->topo->order, T->topo->lens);
    char * par_idx = (char*)malloc(sizeof(char)*T->topo->order);
    for (int i=0; i<T->topo->order; i++){
      par_idx[i] = 'a'+i+1;
    }
    char mat_idx[2];
    int slice_st[2];
    int slice_end[2];
    int k_start = 0;
    int kd = 0;
    int div = 1;
    for (int d=0; d<div; d++){
      k_start += kd;
      kd = k/div + (d < k%div);
      int k_end = k_start + kd;

      Timer t_solve_remap("Solve_remap_mats");
      t_solve_remap.start();
      for (int i=0; i<T->order; i++){
        Tensor<dtype> mmat;
        Tensor<dtype> * mat ;
        mat = mat_list[i];
        int64_t tot_sz;
        if (is_vec)
          tot_sz = T->lens[i];
        else
          tot_sz = T->lens[i]*kd;
        if (div>1){
          if (aux_mode_first){
            slice_st[0] = k_start;
            slice_st[1] = 0;
            slice_end[0] = k_end;
            slice_end[1] = T->lens[i];
            mat_strides[2*i+0] = kd;
            mat_strides[2*i+1] = 1;
          } else {
            slice_st[1] = k_start;
            slice_st[0] = 0;
            slice_end[1] = k_end;
            slice_end[0] = T->lens[i];
            mat_strides[2*i+0] = 1;
            mat_strides[2*i+1] = T->lens[i];
          }
          mmat = mat_list[i]->slice(slice_st, slice_end);
          mat = &mmat;
        } else if (!is_vec) {
          if (aux_mode_first){
            mat_strides[2*i+0] = k;
            mat_strides[2*i+1] = 1;
          } else {
            mat_strides[2*i+0] = 1;
            mat_strides[2*i+1] = T->lens[i];
          }
        }
        int nrow, ncol;
        if (aux_mode_first){
          nrow = kd;
          ncol = T->lens[i];
        } else {
          nrow = T->lens[i];
          ncol = kd;
        }
        if (phys_phase[i] == 1){
          redist_mats[i] = NULL;
          if (T->wrld->np == 1){
            IASSERT(div == 1);
            arrs[i] = (dtype*)mat_list[i]->data;
            if (i==mode){
                regu_arr = (dtype*)regu->data;
            }
          }
          else if (i!=mode) {
            arrs[i] = (dtype*)T->sr->alloc(tot_sz);
            mat->read_all(arrs[i], true);
          } else{
            if (is_vec){
              Vector<dtype> * v = new Vector<dtype>(mat_list[i]->lens[0], 'a'-1, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
              v->operator[]("i") += mat_list[i]->operator[]("i");
              redist_mats[i] = v;
              arrs[i] = (dtype*)v->data;
            }
            else {
              char nonastr[2];
              nonastr[0] = 'a'-1;
              nonastr[1] = 'a'-2;
              Matrix<dtype> * m = new Matrix<dtype>(nrow, ncol, nonastr, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
                m->operator[]("ij") += mat->operator[]("ij");
                redist_mats[i] = m;
                arrs[i] = (dtype*)m->data;
            }
            arrs[i] = (dtype*)redist_mats[i]->data;
            mat->read_all(arrs[i], true);
            Vector<dtype> * v_reg = new Vector<dtype>(regu->lens[0], 'a'-1, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            v_reg->operator[]("i") += regu->operator[]("i");
            redist_regu = v_reg;
            regu_arr = (dtype*)v_reg->data;
          }
        } 
        else{
          int topo_dim = T->edge_map[i].cdt;
          IASSERT(T->edge_map[i].type == CTF_int::PHYSICAL_MAP);
          IASSERT(!T->edge_map[i].has_child || T->edge_map[i].child->type != CTF_int::PHYSICAL_MAP);
          if (aux_mode_first){
            mat_idx[0] = 'a';
            mat_idx[1] = par_idx[topo_dim];
          } else {
            mat_idx[0] = par_idx[topo_dim];
            mat_idx[1] = 'a';
          }
          int comm_lda = 1;
          for (int l=0; l<topo_dim; l++){
            comm_lda *= T->topo->dim_comm[l].np;
          }
          CTF_int::CommData cmdt(T->wrld->rank-comm_lda*T->topo->dim_comm[topo_dim].rank,T->topo->dim_comm[topo_dim].rank,T->wrld->cdt);
          if (is_vec){
            Vector<dtype> * v = new Vector<dtype>(mat_list[i]->lens[0], par_idx[topo_dim], par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            v->operator[]("i") += mat_list[i]->operator[]("i");
            redist_mats[i] = v;
            arrs[i] = (dtype*)v->data;
            if (i != mode)
              cmdt.bcast(v->data,v->size,T->sr->mdtype(),0);
           } else {
            Matrix<dtype> * m = new Matrix<dtype>(nrow, ncol, mat_idx, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            m->operator[]("ij") += mat->operator[]("ij");
            redist_mats[i] = m;
            arrs[i] = (dtype*)m->data;
            if (i != mode)
              cmdt.bcast(m->data,m->size,T->sr->mdtype(),0);
            if (aux_mode_first){
              mat_strides[2*i+0] = kd;
              mat_strides[2*i+1] = 1;
            } else {
              mat_strides[2*i+0] = 1;
              mat_strides[2*i+1] = m->pad_edge_len[0]/phys_phase[i];
            }
          }
          if (i== mode){
            Vector<dtype> * v_reg = new Vector<dtype>(regu->lens[0], par_idx[topo_dim], par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            v_reg->operator[]("i") += regu->operator[]("i");
            redist_regu = v_reg;
            regu_arr = (dtype*)v_reg->data;
          }
        }
      }

      t_solve_remap.stop();
      int jr = T->edge_map[mode].calc_phys_rank(T->topo);
      MPI_Comm slice_comm;
      MPI_Comm_split(T->wrld->comm, jr, T->wrld->rank, &slice_comm);
      int cm_rank,cm_size;
      MPI_Comm_rank(slice_comm, &cm_rank);
      MPI_Comm_size(slice_comm,&cm_size);
      
      int64_t I = T->pad_edge_len[mode]/T->edge_map[mode].np ;
      int R = mat_list[0]->lens[1-aux_mode_first];
      
      
      Timer t_trav("Sort_nnz");
      Timer t_LHS_work("LHS_work");
      Timer t_solve_lhs("LHS_solves");
      Timer t_scatter("Scatter");
      Timer t_gather("Gather");
      t_trav.start() ; 

      /*
      std::sort(pairs,pairs+ npair,[T,phys_phase,mode,ldas](Pair<dtype> i1, Pair<dtype> i2){
        return (((i1.k/ldas[mode])%T->lens[mode])/phys_phase[mode] < ((i2.k/ldas[mode])%T->lens[mode])/phys_phase[mode] ) ; 
      }) ;
      */
      

      Pair<dtype> * pairs_copy = (Pair<dtype>*)malloc(npair*2*sizeof(int64_t)) ;
      Pair<dtype> * pairs_copy_RHS = (Pair<dtype>*)malloc(npair*2*sizeof(int64_t)) ;
      int * indices = (int *)malloc(npair*sizeof(int64_t)) ;
      int64_t * c = (int64_t *) calloc(I+1,sizeof(int64_t));
      int64_t * count = (int64_t *) calloc(I,sizeof(int64_t));

      std::copy(pairs_RHS, pairs_RHS + npair, pairs_copy_RHS) ;

      
      for (int64_t i=0; i<npair ; i++){
        int64_t key = pairs[i].k/ldas[mode] ; 
        indices[i] = (key%T->lens[mode])/phys_phase[mode];
        ++c[indices[i]];
        ++count[indices[i]] ; 
      }

      for(int64_t i=1;i<=I;i++){
        c[i]+=c[i-1];            
      }

      for(int64_t i=npair-1;i>=0;i--){
        pairs_copy[c[indices[i]]-1] = pairs[i];
        pairs_RHS[c[indices[i]]-1] = pairs_copy_RHS[i];
        --c[indices[i]] ;       
      } 
      std::copy(pairs_copy, pairs_copy + npair,pairs)  ;
      //std::copy(pairs_copy_RHS, pairs_copy_RHS + npair,pairs_RHS)  ;

      free(c);
      free(pairs_copy);
      free(pairs_copy_RHS);
      free(indices);

      t_trav.stop();

      //int64_t I_s = std::ceil(float(I)/cm_size) ;
      int batches = 1 ;
      int64_t batched_I = I ; 
      int64_t max_memuse = CTF_int::proc_bytes_available() ;
      int64_t I_s ;
      int64_t rows ;
      int buffer = 2048*5; 

      while (true){
        if (max_memuse > ((std::ceil(float(I/batches)/cm_size)*cm_size*(R+3)*R)*2 + buffer*(R+1) + 10*R + 2*I + 100 )*(int64_t)sizeof(dtype) *(int64_t)sizeof(dtype) ) {
          break ;
        }
        else{
          batches +=1 ;  
        }
      }
      MPI_Allreduce(MPI_IN_PLACE, &batches, 1, MPI_INT, MPI_MAX, T->wrld->comm);
      batched_I = I/batches ; 

      int64_t total= 0;
      for (int b =0 ; b<batches ; b++){
        if (b != batches-1){
          rows = batched_I; 
        } 
        else{
          rows = batched_I + I%batches ;
        }

        I_s = std::ceil(float(rows)/cm_size) ;

        dtype * LHS_list = (dtype *) calloc(I_s*cm_size*R*R,sizeof(dtype) );
        dtype * RHS_buf = (dtype *) calloc(I_s*cm_size*R,sizeof(dtype) );
          
          if (LHS_list == 0){
            printf("Memory full LHS for proc [%d] \n",T->wrld->rank);
          }
        
        //define how the symmetric arrays are referenced, keep this consistent throughout
        char* uplo = "L" ;
        char* trans = "N" ; //if want to incorporate column major then change this 
        int scale = 1 ;
        double alpha = 1.0 ; 
        int info =0 ; 

        
        t_LHS_work.start();
        //int * inds = (int*)malloc(T->order*sizeof(int));
        
        int sweeps ; 
          
        //double * row = (double *) malloc(R* sizeof(double) );
        dtype * H = (dtype *) calloc(buffer*R,sizeof(dtype) ) ;
        dtype * entries = (dtype *) calloc(buffer,sizeof(dtype) ) ;
        
        for (int64_t j =0  ; j< rows ; j++){ 
          sweeps = count[j+ b*batched_I]/buffer ;

          if (sweeps >0){
            for (int s = 0 ; s< sweeps ; s++){
#ifdef _OPENMP
              #pragma omp parallel for 
#endif
              for(int q = 0 ; q<buffer ; q++){
                int64_t key = pairs[total + q].k ;
                std::fill(
                H + q*R,
                H + (q+1)*R ,
                std ::sqrt(pairs[total + q].d)) ;
                entries[q] = pairs_RHS[total + q].d ;
                for(int i = 0 ; i < T->order ; i++){
                  if (i!=mode){
                    int64_t ke= key/ldas[i];
                    int index = (ke%T->lens[i])/phys_phase[i];
                    CTF_int::default_vec_mul(&arrs[i][index*R], H+q*R, H+q*R, R) ;
                  }
                }
              }
              CTF_BLAS::syrk<dtype>(uplo,trans,&R,&buffer,&alpha,H,&R,&alpha,&LHS_list[j*R*R],&R) ;
                
              CTF_BLAS::gemv<dtype>(trans,  // No transpose
                  &R,            // m: rows = R
                  &buffer,       // n: cols = buffer
                  &alpha,
                  H,             // H is buffer x R row-major
                  &R,            // ld = R (row stride in row-major layout)
                  entries,       // vector of length sweeps
                  &scale,
                  &alpha,
                  RHS_buf + j*R, // output vector of length R
                  &scale);
              
              std::fill(
                  H,
                  H+ buffer*R,
                  0.);
              std::fill(
                  entries,
                  entries + buffer,
                  0.);
              total+=buffer ; 
            }
          }
          sweeps = count[j+ b*batched_I]%buffer ;
          if (sweeps>0){
#ifdef _OPENMP
            #pragma omp parallel for
#endif
              
            for(int q = 0 ; q<sweeps ; q++){
              int64_t key = pairs[total + q].k ;
              std::fill(
              H + q*R,
              H + (q+1)*R ,
              std ::sqrt(pairs[total + q].d)) ;
              entries[q] = pairs_RHS[total + q].d ;
              for(int i = 0 ; i < T->order ; i++){
                if (i!=mode){
                  int64_t ke= key/ldas[i];
                  int index = (ke%T->lens[i])/phys_phase[i];
                  CTF_int::default_vec_mul(&arrs[i][index*R], H+q*R, H+q*R, R) ;
                }
              }
            }
            CTF_BLAS::syrk<dtype>(uplo,trans,&R,&sweeps,&alpha,H,&R,&alpha,&LHS_list[j*R*R],&R) ;
              
            
            CTF_BLAS::gemv<dtype>(trans,  // No transpose
                      &R,            // m: rows = R
                      &sweeps,       // n: cols = sweeps
                      &alpha,
                      H,             // H is buffer x R row-major
                      &R,            // ld = R (row stride in row-major layout)
                      entries,       // vector of length sweeps
                      &scale,
                      &alpha,
                      RHS_buf + j*R, // output vector of length R
                      &scale);
              
            std::fill(
                H,
                H+ sweeps*R,
                0.);
            std::fill(
                  entries,
                  entries + sweeps,
                  0.);
            total+=sweeps ; 
          } 
        }

        free(H) ;
        free(entries);
        //free(inds) ;
        t_LHS_work.stop();


        
        t_scatter.start() ;

        int R_sym = R * (R + 1) / 2;

        dtype* combined_LHS_RHS = (dtype*) malloc(I_s * cm_size * (R_sym + R) * sizeof(dtype));
          
        for (int j = 0; j < I_s * cm_size; j++) {
            dtype* lhs_src = LHS_list + j * R * R;                    // full matrix block
            dtype* rhs_src = RHS_buf + j * R;                         // full RHS block
            dtype* out     = combined_LHS_RHS + j * (R_sym + R);      // packed block
        
            // Copy upper triangle row-major into packed format
            int idx = 0;
            for (int row = 0; row < R; row++) {
                for (int col = row; col < R; col++) {  // upper triangle
                    out[idx++] = lhs_src[row * R + col];
                }
            }
        
            // Copy RHS
            std::copy(rhs_src, rhs_src + R, out + R_sym);
        }


        //scatter reduce left hand sides and scatter right hand sides in a buffer
        int* Recv_count = (int*) malloc(sizeof(int)*cm_size) ; 
        std::fill(Recv_count, Recv_count + cm_size, I_s * (R_sym + R));

        MPI_Reduce_scatter( MPI_IN_PLACE, combined_LHS_RHS, Recv_count , MPI_DOUBLE, MPI_SUM, slice_comm );

        free(LHS_list) ;
        free(RHS_buf) ;

        free(Recv_count);
        int barrier_block_len;
        dtype* barrier_mat = nullptr;
        
        if (barrier == -1.0 && !proj) {
            // Only regularization is needed
            barrier_block_len = 1;
            barrier_mat = (dtype*) calloc(I_s * cm_size * barrier_block_len, sizeof(dtype));
            
            if (cm_rank == 0) {
                std::copy(regu_arr + b * batched_I,
                          regu_arr + b * batched_I + rows,
                          barrier_mat);
            }
            
        } else {
            // Need arrs + regu
            barrier_block_len = R + 1;
            barrier_mat = (dtype*) calloc(I_s * cm_size * barrier_block_len, sizeof(dtype));
        
            if (cm_rank == 0) {
                int64_t start_idx = b * batched_I;
            
                for (int i = 0; i < rows; i++) {
                    int64_t global_cm_idx = start_idx + i;
            
                    // Copy R entries from arrs[mode]
                    std::copy(arrs[mode] + global_cm_idx * R,
                              arrs[mode] + (global_cm_idx + 1) * R,
                              barrier_mat + i * barrier_block_len);
                    
                    // Copy regularization value
                    barrier_mat[i * barrier_block_len + R] = regu_arr[global_cm_idx];
                }
            }
        }
        
        // Scatter based on actual block length
        MPI_Scatter(barrier_mat,
                    I_s * barrier_block_len, MPI_DOUBLE,
                    (cm_rank == 0 ? MPI_IN_PLACE : barrier_mat),
                    I_s * barrier_block_len, MPI_DOUBLE,
                    0, slice_comm);


        std::vector<dtype> thresholds(I_s, 0.0);
        dtype* row_ptr ;
        dtype* mat_block ;
        dtype regular ;
        for (int i = 0; i < I_s; i++) {
            dtype* block_base = combined_LHS_RHS + i * (R_sym + R);
            dtype* lhs_block = block_base;         // packed upper triangle
            dtype* rhs_block = block_base + R_sym; // full RHS vector

            if (barrier != -1.0 || proj){
                row_ptr = barrier_mat + i * (R + 1);
                mat_block = row_ptr;
                regular = row_ptr[R];
            }
            

        
            if (barrier != -1.0) {
                // Barrier term added to RHS and LHS diagonal
                
                for (int r = 0; r < R; r++) {
                    rhs_block[r] += barrier / (mat_block[r] + 1e-10);
                }
        
                // Update diagonal entries of LHS in packed format
                int idx = 0;
                for (int d = 0; d < R; d++) {
                    dtype mat_val = mat_block[d];
                    dtype diag_update = barrier / (mat_val * mat_val + 1e-10) + regular;
                    lhs_block[idx] += diag_update;
                    idx += R - d;
                }
            }

            else if (proj){
                // Update diagonal using the regularization pointer above
                int idx = 0;
                for (int d = 0; d < R; d++) {
                    lhs_block[idx] += regular;
                    idx += R - d;
                }
            }
            else {
                // No barrier, only regularization
                regular  = barrier_mat[i] ;
                int idx = 0;
                for (int d = 0; d < R; d++) {
                    lhs_block[idx] += regular;
                    idx += R - d;
                }
            }
        
            // Optional constant addition
            if (add_ones) {
                for (int r = 0; r < R; r++) {
                    rhs_block[r] -= 1.0;
                }
            }
        
            if (proj) {
                double sq_sum = 0.0;
                for (int r = 0; r < R; r++) {
                    dtype val = mat_block[r] + rhs_block[r];
                    dtype clipped_val = val > 0.0 ? val : 0.0;
                    dtype diff = mat_block[r] - clipped_val;
                    sq_sum += diff * diff;
                }
                dtype nrm = std::sqrt(sq_sum);
                thresholds[i] = std::min(nrm, epsilon);
            }
        }
          
        
              
        
        t_scatter.stop() ; 
        // Predeclare vectors once, outside all conditionals
        // Declare once outside the loop
        std::vector<int> selected_indices;
        std::vector<int> unselected_indices;
        std::vector<dtype> lhs_selected(R * R, 0.0);
        std::vector<dtype> rhs_selected;

        // Preallocate with maximum possible sizes (R and R x R)
        // Optionally preallocate to max size if you expect reuse
        if (proj) {
            rhs_selected.resize(R);
        }
        // Call local spd solve on I/cm_size different systems locally
        t_solve_lhs.start();
          
        for (int i = 0; i < I_s; i++) {
            if (i + cm_rank * I_s + b * batched_I < I - (T->lens[mode] % T->edge_map[mode].np > 0) + (jr < T->lens[mode] % T->edge_map[mode].np)) {
                dtype* block_base = combined_LHS_RHS + i * (R_sym + R);
                dtype* lhs_block = block_base;         // packed upper triangle
                dtype* rhs_block = block_base + R_sym; // RHS vector
                
                dtype* row_ptr = barrier_mat + i * (R + 1);
                dtype* mat_block = row_ptr;            // arrs[mode][i * R]
                
                
                if (proj) {
        
                    selected_indices.clear();
                    unselected_indices.clear();
                    // negative gradient pointing to make things negative
                    // not selected if current iterate already 0 or <= threshold
                    // Classify entries once
                    for (int r = 0; r < R; r++) {
                        bool is_rhs_nonpos = rhs_block[r] <= 0.0;
                        bool is_mat_zero = mat_block[r] == 0.0;
                        bool is_mat_small = mat_block[r] > 0.0 && mat_block[r] <= thresholds[i];
                    
                        if (is_rhs_nonpos && (is_mat_zero || is_mat_small)) {
                            unselected_indices.push_back(r);  // Collect unselected
                        } else {
                            selected_indices.push_back(r);    // Collect selected
                        }
                    }
        
                    int R_selected = selected_indices.size();

        
                    if (R_selected > 0) {
                        lhs_selected.resize(R_selected * R_selected);
                        rhs_selected.resize(R_selected);
        
        
                        for (int ii = 0; ii < R_selected; ii++) {
                            int idx_i = selected_indices[ii];
                            rhs_selected[ii] = rhs_block[idx_i];
                        
                            for (int jj = ii; jj < R_selected; jj++) {
                                int idx_j = selected_indices[jj];
                        
                                // Access (idx_i, idx_j) from global packed matrix
                                int row = std::min(idx_i, idx_j); // ensure row ≤ col
                                int col = std::max(idx_i, idx_j);
                                int packed_idx = row * R - (row * (row - 1)) / 2 + (col - row);
                        
                                lhs_selected[ii * R_selected + jj] = lhs_block[packed_idx];
                            }
                        }
        
                        // printf("=== [Rank %d, Local i = %d] Projected Solve ===\n", cm_rank, i);
                        // printf("RHS (rhs_selected):\n");
                        // for (int ii = 0; ii < R_selected; ii++) {
                        //     printf("%.6e ", rhs_selected[ii]);
                        // }
                        // printf("\nLHS (lhs_selected, upper triangle only):\n");
                        // for (int ii = 0; ii < R_selected; ii++) {
                        //     for (int jj = 0; jj < R_selected; jj++) {
                        //         if (jj >= ii)
                        //             printf("%.6e ", lhs_selected[ii * R_selected + jj]);
                        //         else
                        //             printf("    .     ");  // visually denote lower triangle
                        //     }
                        //     printf("\n");
                        // }
                        // Solve system
                        CTF_BLAS::posv<dtype>(uplo, &R_selected, &scale,
                                              lhs_selected.data(), &R_selected,
                                              rhs_selected.data(), &R_selected, &info);
        
                        // Write solved entries back in correct order
                        for (int ii = 0; ii < R_selected; ii++) {
                            mat_block[selected_indices[ii]] = rhs_selected[ii];
                        }
                    }
        
                    // Fallback handling for unselected entries
                    for (int idx : unselected_indices) {
                        mat_block[idx] = (mat_block[idx] == 0.0) ? 0.0 : rhs_block[idx];
                    }

                    double sum = 0.0;
                    // We are computing 1/2 d^T H d - rhs^T d
                    for (int ii = 0; ii < R_selected; ++ii) {
                        double di = rhs_selected[ii];  // Packed solution vector
                        int idx_i = selected_indices[ii];
                    
                        // Diagonal term
                        dtype regs = barrier_mat[i * (R + 1) + R];
                        sum += 0.5 * (lhs_selected[ii * R_selected + ii] - regs) * di * di;
                    
                        // Off-diagonal symmetric terms
                        for (int jj = ii + 1; jj < R_selected; ++jj) {
                            double dj = rhs_selected[jj];
                            sum += lhs_selected[ii * R_selected + jj] * di * dj;
                        }
                        // Dot product with original RHS (rhs_orig is full-sized and negated)
                        sum -= di * rhs_block[idx_i];
                    }
                    barrier_mat[i * (R + 1) + R] = (dtype) sum;
        
                } else {
                    // No projection, full system solve
                    // Fill upper triangle from packed `lhs_block`
                    // Posv wants full R xR working memory
                    
                    int packed_idx = 0;
                    for (int ii = 0; ii < R; ii++) {
                        int row_offset = ii * R;
                        for (int jj = ii; jj < R; jj++) {
                            lhs_selected[row_offset + jj] = lhs_block[packed_idx++];
                        }
                    }

                    CTF_BLAS::posv<dtype>(uplo, &R, &scale, lhs_selected.data(), &R, rhs_block, &R, &info);

                }
                
            }
        }
        t_solve_lhs.stop();

        t_gather.start();
        
        if (proj){
            free(combined_LHS_RHS);
            if (cm_rank==0){
              MPI_Gather(MPI_IN_PLACE, I_s*(R+1), MPI_DOUBLE, 
                         barrier_mat, I_s*(R+1), MPI_DOUBLE, 
                         0, slice_comm);
            }
            else{
              MPI_Gather(barrier_mat, I_s*(R+1), MPI_DOUBLE, 
                         NULL, I_s*(R+1), MPI_DOUBLE,
                         0, slice_comm);
            }
              
              
            if (cm_rank == 0) {
                int64_t start_idx = b * batched_I;
            
                for (int i = 0; i < rows; i++) {
                    int64_t global_idx = start_idx + i;
            
                    std::copy(
                        barrier_mat + i * (R + 1),
                        barrier_mat + i * (R + 1) + R,
                        arrs[mode] + global_idx * R
                    );
            
                    regu_arr[global_idx] = barrier_mat[i * (R + 1) + R];
                }
            }
            free(barrier_mat);
        }

        else{
            free(barrier_mat);
            dtype* right_sides = (dtype*) calloc(I_s * cm_size * R, sizeof(dtype));
            
            for (int i = 0; i < rows; i++) {
                dtype* block_base = combined_LHS_RHS + i * (R_sym + R);
                dtype* rhs_block = block_base + R_sym;
                std::copy(rhs_block, rhs_block + R, right_sides + i * R);
            }
            free(combined_LHS_RHS);
            
            if (cm_rank == 0) {
                MPI_Gather(MPI_IN_PLACE, I_s * R, MPI_DOUBLE,
                           right_sides,  I_s * R, MPI_DOUBLE,
                           0, slice_comm);
            } else {
                MPI_Gather(right_sides, I_s * R, MPI_DOUBLE,
                           NULL, I_s * R, MPI_DOUBLE,
                           0, slice_comm);
            }

            if (cm_rank == 0) {
                std::copy(right_sides, right_sides + rows * R,
                          arrs[mode] + b * batched_I * R);
            }
            free(right_sides);
        }
          
        
      }
      
      free(count) ;

      
      MPI_Comm_free(&slice_comm);
      t_gather.stop();


      for (int j=0 ; j< T->order ; j++){
        if (j==mode){
          if (redist_mats[j] != NULL){
            mat_list[j]->set_zero();
            if (is_vec){
                mat_list[j]->operator[]("i") += redist_mats[j]->operator[]("i");
            }
            else{
              mat_list[j]->operator[]("ij") += redist_mats[j]->operator[]("ij");
            }
            delete redist_mats[j];
          }
          else {
            IASSERT((dtype*)mat_list[j]->data == arrs[j]);
          } 
        }
        else {
          if (redist_mats[j] != NULL){
            if (redist_mats[j]->data != (char*)arrs[j])
              T->sr->dealloc((char*)arrs[j]);
            delete redist_mats[j];
          } else {
            if (arrs[j] != (dtype*)mat_list[j]->data)
              T->sr->dealloc((char*)arrs[j]);
          }
        }
      }
      if (redist_regu != NULL){
        regu->set_zero();
        regu->operator[]("i") += redist_regu->operator[]("i");
        delete redist_regu;
      }
      else{
        IASSERT((dtype*)regu->data == regu_arr);
      }
    }
    free(redist_mats);

    if (mat_strides != NULL) free(mat_strides);
    free(par_idx);
    free(phys_phase);
    free(ldas);
    free(arrs);
    if (!T->is_sparse){
      T->sr->pair_dealloc((char*)pairs);
      RHS->sr->pair_dealloc((char*)pairs_RHS);
    }
      
    t_solve_factor.stop();
  }

template<typename dtype>
  void Solve_Factor_with_RHS(Tensor<dtype> * T, Tensor<dtype> ** mat_list, Tensor<dtype> * RHS, int mode, double regu, double barrier, bool aux_mode_first){
    // Mode defines what factor index we're computing

    int64_t npair ;
    Pair<dtype> * pairs ;
    if (T->is_sparse){
      pairs = (Pair<dtype>*)T->data;
      npair = T->nnz_loc;
    } else{
      T->get_local_pairs(&npair, &pairs, true, false);
    }
    Timer t_solve_factor("Solve_Factor");
    t_solve_factor.start();
    int k = -1;
    bool is_vec = mat_list[0]->order == 1;
    if (!is_vec)
      k = mat_list[0]->lens[1-aux_mode_first];
    IASSERT(mode >= 0 && mode < T->order);
    for (int i=0; i<T->order; i++){
      IASSERT(is_vec || T->lens[i] == mat_list[i]->lens[aux_mode_first]);
      IASSERT(!mat_list[i]->is_sparse);
    }
    dtype ** arrs = (dtype**)malloc(sizeof(dtype*)*T->order);
    dtype * rhs_arr = NULL ;
    int64_t * ldas = (int64_t*)malloc(T->order*sizeof(int64_t));
    int * phys_phase = (int*)malloc(T->order*sizeof(int));
    int * mat_strides = NULL;
    if (!is_vec)
      mat_strides = (int*)malloc(2*T->order*sizeof(int));
    for (int i=0; i<T->order; i++){
      phys_phase[i] = T->edge_map[i].calc_phys_phase();
    }

    ldas[0] = 1;
    for (int i=1; i<T->order; i++){
      ldas[i] = ldas[i-1] * T->lens[i-1];
    }

    Tensor<dtype> ** redist_mats = (Tensor<dtype>**)malloc(sizeof(Tensor<dtype>*)*T->order);
    Tensor<dtype> * redist_rhs = NULL;

    Partition par(T->topo->order, T->topo->lens);
    char * par_idx = (char*)malloc(sizeof(char)*T->topo->order);
    for (int i=0; i<T->topo->order; i++){
      par_idx[i] = 'a'+i+1;
    }
    char mat_idx[2];
    int slice_st[2];
    int slice_end[2];
    int k_start = 0;
    int kd = 0;
    int div = 1;
    for (int d=0; d<div; d++){
      k_start += kd;
      kd = k/div + (d < k%div);
      int k_end = k_start + kd;

      Timer t_solve_remap("Solve_remap_mats");
      t_solve_remap.start();
      for (int i=0; i<T->order; i++){
        Tensor<dtype> mmat;
        Tensor<dtype> * mat ;
        Tensor <dtype> rrmat;
        Tensor <dtype> * rmat;
        mat = mat_list[i];
        int64_t tot_sz;
        if (is_vec)
          tot_sz = T->lens[i];
        else
          tot_sz = T->lens[i]*kd;
        if (div>1){
          if (aux_mode_first){
            slice_st[0] = k_start;
            slice_st[1] = 0;
            slice_end[0] = k_end;
            slice_end[1] = T->lens[i];
            mat_strides[2*i+0] = kd;
            mat_strides[2*i+1] = 1;
          } else {
            slice_st[1] = k_start;
            slice_st[0] = 0;
            slice_end[1] = k_end;
            slice_end[0] = T->lens[i];
            mat_strides[2*i+0] = 1;
            mat_strides[2*i+1] = T->lens[i];
          }
          mmat = mat_list[i]->slice(slice_st, slice_end);
          mat = &mmat;
          if (i== mode){
            rrmat = RHS->slice(slice_st, slice_end);
            rmat = &rrmat;
          }
        } else if (!is_vec) {
          if (aux_mode_first){
            mat_strides[2*i+0] = k;
            mat_strides[2*i+1] = 1;
          } else {
            mat_strides[2*i+0] = 1;
            mat_strides[2*i+1] = T->lens[i];
          }
        }
        int nrow, ncol;
        if (aux_mode_first){
          nrow = kd;
          ncol = T->lens[i];
        } else {
          nrow = T->lens[i];
          ncol = kd;
        }
        if (phys_phase[i] == 1){
          redist_mats[i] = NULL;
          if (T->wrld->np == 1){
            IASSERT(div == 1);
            arrs[i] = (dtype*)mat_list[i]->data;
            if (i==mode){
                rhs_arr = (dtype*)RHS->data;
            }
          }
          else if (i!=mode) {
            arrs[i] = (dtype*)T->sr->alloc(tot_sz);
            mat->read_all(arrs[i], true);
          } else{
            if (is_vec){
              Vector<dtype> * v = new Vector<dtype>(mat_list[i]->lens[0], 'a'-1, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
              v->operator[]("i") += mat->operator[]("i");
              redist_mats[i] = v;
              arrs[i] = (dtype*)v->data;

              Vector<dtype> * v_rhs = new Vector<dtype>(mat_list[i]->lens[0], 'a'-1, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
              v_rhs->operator[]("i") += rmat->operator[]("i");
              redist_rhs = v_rhs;
              rhs_arr = (dtype*)v_rhs->data;
            }
            else {
              char nonastr[2];
              nonastr[0] = 'a'-1;
              nonastr[1] = 'a'-2;
              Matrix<dtype> * m = new Matrix<dtype>(nrow, ncol, nonastr, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
              m->operator[]("ij") += mat->operator[]("ij");
              redist_mats[i] = m;
              arrs[i] = (dtype*)m->data;

              Matrix<dtype> * m_rhs = new Matrix<dtype>(nrow, ncol, nonastr, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
              m_rhs->operator[]("ij") += rmat->operator[]("ij");
              redist_rhs = m_rhs;
              rhs_arr = (dtype*)m_rhs->data;
            }
            arrs[i] = (dtype*)redist_mats[i]->data;
            mat->read_all(arrs[i], true);
          }
        }
        else{
          int topo_dim = T->edge_map[i].cdt;
          IASSERT(T->edge_map[i].type == CTF_int::PHYSICAL_MAP);
          IASSERT(!T->edge_map[i].has_child || T->edge_map[i].child->type != CTF_int::PHYSICAL_MAP);
          if (aux_mode_first){
            mat_idx[0] = 'a';
            mat_idx[1] = par_idx[topo_dim];
          } else {
            mat_idx[0] = par_idx[topo_dim];
            mat_idx[1] = 'a';
          }
          int comm_lda = 1;
          for (int l=0; l<topo_dim; l++){
            comm_lda *= T->topo->dim_comm[l].np;
          }
          CTF_int::CommData cmdt(T->wrld->rank-comm_lda*T->topo->dim_comm[topo_dim].rank,T->topo->dim_comm[topo_dim].rank,T->wrld->cdt);
          if (is_vec){
            Vector<dtype> * v = new Vector<dtype>(mat_list[i]->lens[0], par_idx[topo_dim], par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            v->operator[]("i") += mat_list[i]->operator[]("i");
            redist_mats[i] = v;
            arrs[i] = (dtype*)v->data;
            if (i ==mode){
                Vector<dtype> * v_rhs = new Vector<dtype>(mat_list[i]->lens[0], par_idx[topo_dim], par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
                v_rhs->operator[]("i") += rmat->operator[]("i");
                redist_rhs = v_rhs;
                rhs_arr = (dtype*)v_rhs->data;
            }
            else {
              cmdt.bcast(v->data,v->size,T->sr->mdtype(),0);
            }
           } else {
            Matrix<dtype> * m = new Matrix<dtype>(nrow, ncol, mat_idx, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            m->operator[]("ij") += mat->operator[]("ij");
            redist_mats[i] = m;
            arrs[i] = (dtype*)m->data;
            if (i == mode){
                Matrix<dtype> * m_rhs = new Matrix<dtype>(nrow, ncol, mat_idx, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
                m_rhs->operator[]("ij") += rmat->operator[]("ij");
                redist_rhs = m_rhs;
                rhs_arr = (dtype*)m_rhs->data;
            }
            else{
              cmdt.bcast(m->data,m->size,T->sr->mdtype(),0);
            }
            if (aux_mode_first){
              mat_strides[2*i+0] = kd;
              mat_strides[2*i+1] = 1;
            } else {
              mat_strides[2*i+0] = 1;
              mat_strides[2*i+1] = m->pad_edge_len[0]/phys_phase[i];
            }
          }
        }
      }
      t_solve_remap.stop();

      int jr = T->edge_map[mode].calc_phys_rank(T->topo);
      MPI_Comm slice_comm;
      MPI_Comm_split(T->wrld->comm, jr, T->wrld->rank, &slice_comm);
      int cm_rank,cm_size;
      MPI_Comm_rank(slice_comm, &cm_rank);
      MPI_Comm_size(slice_comm,&cm_size);
      
      int64_t I = T->pad_edge_len[mode]/T->edge_map[mode].np ;
      int R = mat_list[0]->lens[1-aux_mode_first];
      
      
      Timer t_trav("Sort_nnz");
      Timer t_LHS_work("LHS_work");
      Timer t_solve_lhs("LHS_solves");
      t_trav.start() ; 

      /*
      std::sort(pairs,pairs+ npair,[T,phys_phase,mode,ldas](Pair<dtype> i1, Pair<dtype> i2){
        return (((i1.k/ldas[mode])%T->lens[mode])/phys_phase[mode] < ((i2.k/ldas[mode])%T->lens[mode])/phys_phase[mode] ) ; 
      }) ;
      */
      

      Pair<dtype> * pairs_copy = (Pair<dtype>*)malloc(npair*2*sizeof(int64_t)) ;
      int * indices = (int *)malloc(npair*sizeof(int64_t)) ;
      int64_t * c = (int64_t *) calloc(I+1,sizeof(int64_t));
      int64_t * count = (int64_t *) calloc(I,sizeof(int64_t));


      
      for (int64_t i=0; i<npair ; i++){
        int64_t key = pairs[i].k/ldas[mode] ; 
        indices[i] = (key%T->lens[mode])/phys_phase[mode];
        ++c[indices[i]];
        ++count[indices[i]] ; 
      }

      for(int64_t i=1;i<=I;i++){
        c[i]+=c[i-1];            
      }

      for(int64_t i=npair-1;i>=0;i--){
        pairs_copy[c[indices[i]]-1] = pairs[i];
        --c[indices[i]] ;       
      } 

      std::copy(pairs_copy, pairs_copy+npair,pairs)  ;

      free(c);
      free(pairs_copy);
      free(indices);

      t_trav.stop();


      //int64_t I_s = std::ceil(float(I)/cm_size) ;
      int batches = 1 ;
      int64_t batched_I = I ; 
      int64_t max_memuse = CTF_int::proc_bytes_available() ;
      int64_t I_s ;
      int64_t rows ;
      int buffer = 2048*5; 

      while (true){
        if (max_memuse > ((std::ceil(float(I/batches)/cm_size)*cm_size*(R+3)*R)*2 + buffer*(R+1) + 100 )*(int64_t)sizeof(dtype) *(int64_t)sizeof(dtype) ) {
          break ;
        }
        else{
          batches +=1 ;  
        }
      }
      MPI_Allreduce(MPI_IN_PLACE, &batches, 1, MPI_INT, MPI_MAX, T->wrld->comm);
      batched_I = I/batches ; 

      int64_t total= 0;
      for (int b =0 ; b<batches ; b++){
        if (b != batches-1){
          rows = batched_I; 
        } 
        else{
          rows = batched_I + I%batches ;
        }

        I_s = std::ceil(float(rows)/cm_size) ;

        dtype * LHS_list = (dtype *) calloc(I_s*cm_size*R*R,sizeof(dtype) );
          
          if (LHS_list == 0){
            printf("Memory full LHS for proc [%d] \n",T->wrld->rank);
          }
        
        //define how the symmetric arrays are referenced, keep this consistent throughout
        char* uplo = "L" ;
        char* trans = "N" ; //if want to incorporate column major then change this 
        int scale = 1 ;
        double alpha = 1.0 ; 
        int info =0 ; 
        

        t_LHS_work.start();
        Timer t_scatter("Scatter and Sc_reduce");
        //int * inds = (int*)malloc(T->order*sizeof(int));
        
        int sweeps ; 
          
        //double * row = (double *) malloc(R* sizeof(double) );
        dtype * H = (dtype *) calloc(buffer*R,sizeof(dtype) ) ;
        
        for (int64_t j =0  ; j< rows ; j++){ 
          sweeps = count[j+ b*batched_I]/buffer ;

          if (sweeps >0){
            for (int s = 0 ; s< sweeps ; s++){
#ifdef _OPENMP
              #pragma omp parallel for 
#endif
              for(int q = 0 ; q<buffer ; q++){
                int64_t key = pairs[total + q].k ;
                std::fill(
                H + q*R,
                H + (q+1)*R ,
                std ::sqrt(pairs[total + q].d)) ;
                for(int i = 0 ; i < T->order ; i++){
                  if (i!=mode){
                    int64_t ke= key/ldas[i];
                    int index = (ke%T->lens[i])/phys_phase[i];
                    CTF_int::default_vec_mul(&arrs[i][index*R], H+q*R, H+q*R, R) ;
                  }
                }
              }
              CTF_BLAS::syrk<dtype>(uplo,trans,&R,&buffer,&alpha,H,&R,&alpha,&LHS_list[j*R*R],&R) ;
              
              std::fill(
                  H,
                  H+ buffer*R,
                  0.);
              total+=buffer ; 
            }
          }
          sweeps = count[j+ b*batched_I]%buffer ;
          if (sweeps>0){
#ifdef _OPENMP
            #pragma omp parallel for
#endif
              
            for(int q = 0 ; q<sweeps ; q++){
              int64_t key = pairs[total + q].k ;
              std::fill(
              H + q*R,
              H + (q+1)*R ,
              std ::sqrt(pairs[total + q].d)) ;
              for(int i = 0 ; i < T->order ; i++){
                if (i!=mode){
                  int64_t ke= key/ldas[i];
                  int index = (ke%T->lens[i])/phys_phase[i];
                  CTF_int::default_vec_mul(&arrs[i][index*R], H+q*R, H+q*R, R) ;
                }
              }
            }
            CTF_BLAS::syrk<dtype>(uplo,trans,&R,&sweeps,&alpha,H,&R,&alpha,&LHS_list[j*R*R],&R) ;
            std::fill(
                H,
                H+ sweeps*R,
                0.);
            total+=sweeps ; 
          } 
        }

        
        free(H) ;
        //free(inds) ;
        t_LHS_work.stop();

        t_scatter.start() ;
        dtype * combined_LHS_RHS = (dtype *) malloc(I_s*cm_size*R*(R+1)*sizeof(dtype) );
          
        std::copy(LHS_list, LHS_list + I_s*cm_size*R*R, combined_LHS_RHS);
        std::copy(rhs_arr + b*batched_I*R, rhs_arr + b*batched_I*R + rows*R,combined_LHS_RHS + I_s*cm_size*R*R);

        //scatter reduce left hand sides and scatter right hand sides in a buffer
        int* Recv_count = (int*) malloc(sizeof(int)*cm_size) ; 
        std::fill(
         Recv_count,
         Recv_count + cm_size,
         I_s*R*(R+1));
          
          
        // MPI_Reduce_scatter( MPI_IN_PLACE, LHS_list, Recv_count , MPI_DOUBLE, MPI_SUM, slice_comm );
        // MPI_Reduce_scatter( MPI_IN_PLACE, RHS_buf, Recv_count , MPI_DOUBLE, MPI_SUM, slice_comm );
        MPI_Reduce_scatter( MPI_IN_PLACE, combined_LHS_RHS, Recv_count , MPI_DOUBLE, MPI_SUM, slice_comm );

        free(LHS_list) ;
        free(Recv_count);


        dtype * barrier_mat = (dtype *) calloc(I_s*cm_size*R, sizeof(dtype) );
        std::copy(arrs[mode] + b*batched_I*R, arrs[mode] + b*batched_I*R + rows*R, barrier_mat) ;
        
          
        if (cm_rank == 0){
          MPI_Scatter(barrier_mat, I_s*R, MPI_DOUBLE, MPI_IN_PLACE, I_s*R,  
                     MPI_DOUBLE, 0, slice_comm);
        }
        else{
          MPI_Scatter(NULL, I_s*R, MPI_DOUBLE, barrier_mat, I_s*R,  
                     MPI_DOUBLE, 0, slice_comm);
        }
          
        for (int i = 0; i < I_s; i++) {
            dtype* rhs_block = combined_LHS_RHS + I_s * R * R + i * R;
            dtype* mat_block = barrier_mat + i * R;
            dtype* lhs_block = combined_LHS_RHS + i * R * R;

            // Update RHS
            for (int r = 0; r < R; r++) {
                rhs_block[r] += barrier / (mat_block[r] + 1e-10);
            }

            // Update LHS diagonal only
            for (int d = 0; d < R; d++) {
                dtype mat_val = mat_block[d];
                dtype diag_update = barrier / (mat_val * mat_val + 1e-10) + regu;
                lhs_block[d * R + d] += diag_update;  // Only diagonal
            }
        }

        free(barrier_mat);


        t_scatter.stop() ; 

        //call local spd solve on I/cm_size different systems locally (avoid calling solve on padding in lhs)
        t_solve_lhs.start() ;
        for (int i=0; i<I_s; i++){
          if (i + cm_rank*I_s + b*batched_I < I - (T->lens[mode] % T->edge_map[mode].np > 0 )  + (jr< T->lens[mode] % T->edge_map[mode].np )){
            //CTF_BLAS::posv<dtype>(uplo,&R,&scale,&LHS_list[i*R*R],&R,&RHS_buf[i*R],&R,&info) ;
            dtype * rhs_block = combined_LHS_RHS + I_s * R * R + i * R;
            dtype * lhs_block = combined_LHS_RHS + i * R * R;

            CTF_BLAS::posv<dtype>(uplo, &R, &scale, lhs_block, &R, rhs_block, &R, &info);
          }
        }
        t_solve_lhs.stop();
        
        
        dtype* solved_rhs_buf = combined_LHS_RHS + I_s * R * R;
        dtype* recv_buf = (cm_rank == 0) ? &arrs[mode][b * batched_I * R] : NULL;

        MPI_Gather(solved_rhs_buf, I_s * R, MPI_DOUBLE,
                   recv_buf,       I_s * R, MPI_DOUBLE,
                   0, slice_comm);

        free(combined_LHS_RHS);
        
      }
      
      free(count) ;
      
      MPI_Comm_free(&slice_comm);

        
      for (int j=0 ; j< T->order ; j++){
        if (j==mode){
          if (redist_mats[j] != NULL){
            mat_list[j]->set_zero();
            mat_list[j]->operator[]("ij") += redist_mats[j]->operator[]("ij");
            delete redist_mats[j];
          }
          else {
             IASSERT((dtype*)mat_list[j]->data == arrs[j]);
          }
         if (redist_rhs != NULL){
            if (redist_rhs->data != (char*)rhs_arr)
              T->sr->dealloc((char*)rhs_arr);
            delete redist_rhs;
          } 
         else {
             IASSERT((dtype*)RHS->data == rhs_arr);
         }

       }
        
        else {
          if (redist_mats[j] != NULL){
            if (redist_mats[j]->data != (char*)arrs[j])
              T->sr->dealloc((char*)arrs[j]);
            delete redist_mats[j];
          } else {
            if (arrs[j] != (dtype*)mat_list[j]->data)
              T->sr->dealloc((char*)arrs[j]);
          }
        }
      }
    }
    free(redist_mats);
    if (mat_strides != NULL) free(mat_strides);
    free(par_idx);
    free(phys_phase);
    free(ldas);
    free(arrs);
    if (!T->is_sparse){
      T->sr->pair_dealloc((char*)pairs);
    }
    t_solve_factor.stop();
   }
  
  template<typename dtype>
void Solve_Factor_Tucker(Tensor<dtype> * T, Tensor<dtype> ** mat_list, Tensor<dtype> * core, Tensor<dtype> * RHS, int mode, Tensor <dtype>* regu, double epsilon, double barrier,  bool proj, bool add_ones, bool aux_mode_first){
    // Mode defines what factor index we're computing
    IASSERT(T->order == RHS->order) ;
    int64_t npair ;
    int64_t npair_RHS;
    Pair<dtype> * pairs ;
    Pair<dtype> * pairs_RHS;
    if (T->is_sparse){
      IASSERT(RHS->is_sparse);
      pairs = (Pair<dtype>*)T->data;
      npair = T->nnz_loc;
      pairs_RHS = (Pair<dtype>*)RHS->data;
      npair_RHS = RHS->nnz_loc;
    } else{
      T->get_local_pairs(&npair, &pairs, true, false);
      RHS->get_local_pairs(&npair_RHS, &pairs_RHS, true, false);
    }
    IASSERT(npair == npair_RHS);
    for (int i=0; i<T->order; i++){
      IASSERT(T->edge_map[i].calc_phys_phase() == RHS->edge_map[i].calc_phys_phase());
    }
    Timer t_solve_factor("Solve_Factor_Tucker");
    t_solve_factor.start();
    int k[T->order];
    bool is_vec[T->order];
    for (int i=0; i<T->order; i++){
      is_vec[i] = mat_list[i]->order == 1;
      if (!is_vec[i]){
        k[i] = mat_list[i]->lens[1-aux_mode_first];
      }
      else{
        k[i] = -1;
      }
    }
    IASSERT(mode >= 0 && mode < T->order);
    for (int i=0; i<T->order; i++){
      IASSERT(is_vec[i] || T->lens[i] == mat_list[i]->lens[aux_mode_first]);
      IASSERT(!mat_list[i]->is_sparse);
    }
    dtype ** arrs = (dtype**)malloc(sizeof(dtype*)*T->order);
    dtype * regu_arr;
    int64_t * ldas = (int64_t*)malloc(T->order*sizeof(int64_t));
    int * phys_phase = (int*)malloc(T->order*sizeof(int));
    int** mat_strides = (int**)malloc(T->order * sizeof(int*));
    for (int i = 0; i < T->order; ++i) {
        if (!is_vec[i]) {
            mat_strides[i] = (int*)malloc(2 * T->order * sizeof(int));
        } else {
            mat_strides[i] = NULL;
        }
    }
    for (int i=0; i<T->order; i++){
      phys_phase[i] = T->edge_map[i].calc_phys_phase();
    }

    ldas[0] = 1;
    for (int i=1; i<T->order; i++){
      ldas[i] = ldas[i-1] * T->lens[i-1];
    }

    Tensor<dtype> ** redist_mats = (Tensor<dtype>**)malloc(sizeof(Tensor<dtype>*)*T->order);
    Tensor<dtype> * redist_regu = NULL;

    Partition par(T->topo->order, T->topo->lens);
    char * par_idx = (char*)malloc(sizeof(char)*T->topo->order);
    for (int i=0; i<T->topo->order; i++){
      par_idx[i] = 'a'+i+1;
    }
    char mat_idx[2];
    int slice_st[2];
    int slice_end[2];
    int k_start = 0;
    int kd = 0;
    int div = 1;
    for (int d=0; d<div; d++){
      Timer t_solve_remap("Solve_remap_mats");
      t_solve_remap.start();
      for (int i=0; i<T->order; i++){
        k_start += kd;
        kd = k[i]/div + (d < k[i]%div);
        int k_end = k_start + kd;
        Tensor<dtype> mmat;
        Tensor<dtype> * mat ;
        mat = mat_list[i];
        int64_t tot_sz;
        if (is_vec[i])
          tot_sz = T->lens[i];
        else
          tot_sz = T->lens[i]*kd;
        if (div>1){
          if (aux_mode_first){
            slice_st[0] = k_start;
            slice_st[1] = 0;
            slice_end[0] = k_end;
            slice_end[1] = T->lens[i];
            mat_strides[i][2*i+0] = kd;
            mat_strides[i][2*i+1] = 1;
          } else {
            slice_st[1] = k_start;
            slice_st[0] = 0;
            slice_end[1] = k_end;
            slice_end[0] = T->lens[i];
            mat_strides[i][2*i+0] = 1;
            mat_strides[i][2*i+1] = T->lens[i];
          }
          mmat = mat_list[i]->slice(slice_st, slice_end);
          mat = &mmat;
        } else if (!is_vec[i]) {
          if (aux_mode_first){
            mat_strides[i][2*i+0] = k[i];
            mat_strides[i][2*i+1] = 1;
          } else {
            mat_strides[i][2*i+0] = 1;
            mat_strides[i][2*i+1] = T->lens[i];
          }
        }
        int nrow, ncol;
        if (aux_mode_first){
          nrow = kd;
          ncol = T->lens[i];
        } else {
          nrow = T->lens[i];
          ncol = kd;
        }
        if (phys_phase[i] == 1){
          redist_mats[i] = NULL;
          if (T->wrld->np == 1){
            IASSERT(div == 1);
            arrs[i] = (dtype*)mat_list[i]->data;
            if (i==mode){
                regu_arr = (dtype*)regu->data;
            }
          }
          else if (i!=mode) {
            arrs[i] = (dtype*)T->sr->alloc(tot_sz);
            mat->read_all(arrs[i], true);
          } else{
            if (is_vec[i]){
              Vector<dtype> * v = new Vector<dtype>(mat_list[i]->lens[0], 'a'-1, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
              v->operator[]("i") += mat_list[i]->operator[]("i");
              redist_mats[i] = v;
              arrs[i] = (dtype*)v->data;
            }
            else {
              char nonastr[2];
              nonastr[0] = 'a'-1;
              nonastr[1] = 'a'-2;
              Matrix<dtype> * m = new Matrix<dtype>(nrow, ncol, nonastr, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
                m->operator[]("ij") += mat->operator[]("ij");
                redist_mats[i] = m;
                arrs[i] = (dtype*)m->data;
            }
            arrs[i] = (dtype*)redist_mats[i]->data;
            mat->read_all(arrs[i], true);
            Vector<dtype> * v_reg = new Vector<dtype>(regu->lens[0], 'a'-1, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            v_reg->operator[]("i") += regu->operator[]("i");
            redist_regu = v_reg;
            regu_arr = (dtype*)v_reg->data;
          }
        } 
        else{
          int topo_dim = T->edge_map[i].cdt;
          IASSERT(T->edge_map[i].type == CTF_int::PHYSICAL_MAP);
          IASSERT(!T->edge_map[i].has_child || T->edge_map[i].child->type != CTF_int::PHYSICAL_MAP);
          if (aux_mode_first){
            mat_idx[0] = 'a';
            mat_idx[1] = par_idx[topo_dim];
          } else {
            mat_idx[0] = par_idx[topo_dim];
            mat_idx[1] = 'a';
          }
          int comm_lda = 1;
          for (int l=0; l<topo_dim; l++){
            comm_lda *= T->topo->dim_comm[l].np;
          }
          CTF_int::CommData cmdt(T->wrld->rank-comm_lda*T->topo->dim_comm[topo_dim].rank,T->topo->dim_comm[topo_dim].rank,T->wrld->cdt);
          if (is_vec[i]){
            Vector<dtype> * v = new Vector<dtype>(mat_list[i]->lens[0], par_idx[topo_dim], par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            v->operator[]("i") += mat_list[i]->operator[]("i");
            redist_mats[i] = v;
            arrs[i] = (dtype*)v->data;
            if (i != mode)
              cmdt.bcast(v->data,v->size,T->sr->mdtype(),0);
           } else {
            Matrix<dtype> * m = new Matrix<dtype>(nrow, ncol, mat_idx, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            m->operator[]("ij") += mat->operator[]("ij");
            redist_mats[i] = m;
            arrs[i] = (dtype*)m->data;
            if (i != mode)
              cmdt.bcast(m->data,m->size,T->sr->mdtype(),0);
            if (aux_mode_first){
              mat_strides[i][2*i+0] = kd;
              mat_strides[i][2*i+1] = 1;
            } else {
              mat_strides[i][2*i+0] = 1;
              mat_strides[i][2*i+1] = m->pad_edge_len[0]/phys_phase[i];
            }
          }
          if (i== mode){
            Vector<dtype> * v_reg = new Vector<dtype>(regu->lens[0], par_idx[topo_dim], par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            v_reg->operator[]("i") += regu->operator[]("i");
            redist_regu = v_reg;
            regu_arr = (dtype*)v_reg->data;
          }
        }
      }
      //CTF::Tensor< dtype >::Tensor  ( tensor const &  A = core, World &   wrld = current rank )
      //World local_world = CTF::World(MPI_COMM_SELF);
      int total_size ; 
      MPI_Comm_size(MPI_COMM_WORLD, &total_size);
      Partition Partition1D(1, &total_size);


      char nonastr_core[core->order];
      for(int i = 0 ; i < core->order ; i++){
        nonastr_core[i] = 'a' - i-1 ;
      }
      char par_idx_core;
      par_idx_core = 'a';



      //B = new Tensor <dtype>(..., Partition1D["i"], Idx_Partition(), World(MPI_COMM_WORLD))

      Tensor<dtype>* l_core1 = new Tensor<dtype>(core->order,
                                          core->lens,
                                          core->sym,
                                          *T->wrld,
                                          nonastr_core,
                                          Partition1D[&par_idx_core],
                                          Idx_Partition(),
                                          NULL,
                                          0,
                                          *core->sr
                                          );

      l_core1->operator[]("ijk") = core->operator[]("ijk");


      int len_core=1;
      for(int i =0; i <l_core1->order ; i++){
          len_core = len_core*l_core1->lens[i];
      }
      
      MPI_Bcast((dtype*)l_core1->data,len_core, MPI_DOUBLE,0, MPI_COMM_WORLD);

      dtype * l_core_data = (dtype *) malloc((int64_t)sizeof(dtype)*len_core);

      memcpy(l_core_data, (dtype *)l_core1->data, len_core*sizeof(dtype)) ;

      t_solve_remap.stop();

      int jr = T->edge_map[mode].calc_phys_rank(T->topo);
      MPI_Comm slice_comm;
      MPI_Comm_split(T->wrld->comm, jr, T->wrld->rank, &slice_comm);
      int cm_rank,cm_size;
      MPI_Comm_rank(slice_comm, &cm_rank);
      MPI_Comm_size(slice_comm,&cm_size);
      
      int64_t I = T->pad_edge_len[mode]/T->edge_map[mode].np ;
      int R = l_core1->lens[mode];

      
      
      Timer t_trav("Sort_nnz");
      Timer t_LHS_work("LHS_work");
      Timer t_solve_lhs("LHS_solves");
      Timer t_scatter("Scatter");
      Timer t_gather("Gather");
    
      t_trav.start() ; 

      /*
      std::sort(pairs,pairs+ npair,[T,phys_phase,mode,ldas](Pair<dtype> i1, Pair<dtype> i2){
        return (((i1.k/ldas[mode])%T->lens[mode])/phys_phase[mode] < ((i2.k/ldas[mode])%T->lens[mode])/phys_phase[mode] ) ; 
      }) ;
      */
      

      Pair<dtype> * pairs_copy = (Pair<dtype>*)malloc(npair*2*sizeof(int64_t)) ;
      Pair<dtype> * pairs_copy_RHS = (Pair<dtype>*)malloc(npair*2*sizeof(int64_t)) ;
      int * indices = (int *)malloc(npair*sizeof(int64_t)) ;
      int64_t * c = (int64_t *) calloc(I+1,sizeof(int64_t));
      int64_t * count = (int64_t *) calloc(I,sizeof(int64_t));

      std::copy(pairs_RHS, pairs_RHS + npair, pairs_copy_RHS) ;

      
      for (int64_t i=0; i<npair ; i++){
        int64_t key = pairs[i].k/ldas[mode] ; 
        indices[i] = (key%T->lens[mode])/phys_phase[mode];
        ++c[indices[i]];
        ++count[indices[i]] ; 
      }

      for(int64_t i=1;i<=I;i++){
        c[i]+=c[i-1];            
      }

      for(int64_t i=npair-1;i>=0;i--){
        pairs_copy[c[indices[i]]-1] = pairs[i];
        pairs_RHS[c[indices[i]]-1] = pairs_copy_RHS[i];
        --c[indices[i]] ;       
      } 
      std::copy(pairs_copy, pairs_copy + npair,pairs)  ;
      //std::copy(pairs_copy_RHS, pairs_copy_RHS + npair,pairs_RHS)  ;

      free(c);
      free(pairs_copy);
      free(pairs_copy_RHS);
      free(indices);

      t_trav.stop();
      
      //int64_t I_s = std::ceil(float(I)/cm_size) ;
      int batches = 1 ;
      int64_t batched_I = I ; 
      int64_t max_memuse = CTF_int::proc_bytes_available() ;
      int64_t I_s ;
      int64_t rows ;
      int buffer = 2048*5;
      int loop_size;

      while (true){
        if (max_memuse > ((std::ceil(float(I/batches)/cm_size)*cm_size*(R+3)*R)*2 + buffer*(R+1) + 10*R + 2*I + 100 )*(int64_t)sizeof(dtype) *(int64_t)sizeof(dtype) ) {
          break ;
        }
        else{
          batches +=1 ;  
        }
      }
      MPI_Allreduce(MPI_IN_PLACE, &batches, 1, MPI_INT, MPI_MAX, T->wrld->comm);
      batched_I = I/batches ; 

      int64_t total= 0;
      for (int b =0 ; b<batches ; b++){
        if (b != batches-1){
          rows = batched_I; 
        } 
        else{
          rows = batched_I + I%batches ;
        }

        I_s = std::ceil(float(rows)/cm_size) ;

        dtype * LHS_list = (dtype *) calloc(I_s*cm_size*R*R,sizeof(dtype) );
        dtype * RHS_buf = (dtype *) calloc(I_s*cm_size*R,sizeof(dtype) );
          
          if (LHS_list == 0){
            printf("Memory full LHS for proc [%d] \n",T->wrld->rank);
          }
        
        
        //define how the symmetric arrays are referenced, keep this consistent throughout
        char* uplo = "L" ;
        char* trans = "N" ;
        char* trans_gem = "T" ; //For GEMV
        int scale = 1 ;
        double alpha= 1.0 ; 
        double beta = 0.0;
        int info =0 ;
        int INCY =1;
        int INCX=1 ; 
        int M,N ;
        int sweeps,outer ;
        double scalar ;
        //int indices[3];
        
        
        t_LHS_work.start();
        

        

        dtype * H = (dtype *) calloc(buffer*R,sizeof(dtype) ) ;
        dtype * input_buf = (dtype *) malloc(len_core*sizeof(dtype));
        dtype * output_buf = (dtype *) malloc(len_core*sizeof(dtype));
        dtype * entries = (dtype *) calloc(buffer,sizeof(dtype) ) ;
        


        for (int64_t j =0  ; j< rows ; j++){ 
          sweeps = count[j+ b*batched_I]/buffer ;
          if (sweeps >0){
            loop_size = buffer ;
            outer = sweeps ;
          }
          else{
            loop_size = count[j+ b*batched_I]%buffer ;
            outer = 1 ;
          }

          for (int s = 0 ; s< outer ; s++){
            for(int q = 0 ; q<loop_size ; q++){
              int64_t key = pairs[total + q].k ;
              M = len_core;
              if (mode != T->order -1){
                N = l_core1->lens[T->order -1];
                M = M/N;
                int64_t ke= key/ldas[T->order -1];
                int index = (ke%T->lens[T->order -1])/phys_phase[T->order -1];
                CTF_BLAS::gemv<dtype>(trans, &M,&N, &alpha, l_core_data, &M,&arrs[T->order -1][index*l_core1->lens[T->order -1]],&INCX,&beta,output_buf,&INCY);
                std::copy(output_buf, output_buf + M, input_buf);

                for (int i = T->order -2 ; i > mode ; i--){
                  int64_t ke= key/ldas[i];
                  int index = (ke%T->lens[i])/phys_phase[i];
                  N = l_core1->lens[i];
                  M = M/N;
                  
                  CTF_BLAS::gemv<dtype>(trans, &M,&N, &alpha, input_buf, &M,&arrs[i][index*l_core1->lens[i]],&INCX,&beta,output_buf,&INCY);
                  std::copy(output_buf, output_buf + M, input_buf);
                }

                for (int i=0 ; i<mode ; i++){
                  int64_t ke= key/ldas[i];
                  int index = (ke%T->lens[i])/phys_phase[i];
                  N = l_core1->lens[i];
                  M = M/N;
                  
                  CTF_BLAS::gemv<dtype>(trans_gem, &N,&M, &alpha, input_buf, &N,&arrs[i][index*l_core1->lens[i]],&INCX,&beta,output_buf,&INCY);
                  std::copy(output_buf, output_buf + M, input_buf);
                }

              }
              else{
                N = l_core1->lens[0];
                M = M/N ;
                int64_t ke= key/ldas[0];
                int index = (ke%T->lens[0])/phys_phase[0];
                CTF_BLAS::gemv<dtype>(trans_gem, &N,&M, &alpha, l_core_data, &N,&arrs[0][index*l_core1->lens[0]],&INCX,&beta,output_buf,&INCY);
                std::copy(output_buf, output_buf + M, input_buf);
                for (int i =1 ; i<mode ; i++){
                  int64_t ke= key/ldas[i];
                  int index = (ke%T->lens[i])/phys_phase[i];
                  N = l_core1->lens[i];
                  M = M/N;
                  
                  CTF_BLAS::gemv<dtype>(trans_gem, &N,&M, &alpha, input_buf, &N,&arrs[i][index*l_core1->lens[i]],&INCX,&beta,output_buf,&INCY);
                  std::copy(output_buf, output_buf + M, input_buf);
                }
              }
              
              scalar = std ::sqrt(pairs[total + q].d) ;
              entries[q] = pairs_RHS[total + q].d ;
              
              std::transform(output_buf, output_buf + R, output_buf,
                   [scalar](double val) { return val * scalar; }) ;
              
              
              std::copy(output_buf, output_buf + R, H + q*R);
              
            }
            CTF_BLAS::syrk<dtype>(uplo,trans,&R,&loop_size,&alpha,H,&R,&alpha,&LHS_list[j*R*R],&R) ;
            CTF_BLAS::gemv<dtype>(trans,  // No transpose
                  &R,            // m: rows = R
                  &loop_size,       // n: cols = buffer
                  &alpha,
                  H,             // H is buffer x R row-major
                  &R,            // ld = R (row stride in row-major layout)
                  entries,       // vector of length sweeps
                  &scale,
                  &alpha,
                  RHS_buf + j*R, // output vector of length R
                  &scale);
            std::fill(
                H,
                H+ loop_size*R,
                0.);
            std::fill(
                  entries,
                  entries + loop_size,
                  0.);
            total+=loop_size ; 
          }
        }
          
        delete l_core1;
        free(H) ;
        free(l_core_data);
        //free(Gem_buf);
        free(input_buf);
        free(output_buf);
        free(entries);



        
        t_LHS_work.stop();


        
        t_scatter.start() ;

        int R_sym = R * (R + 1) / 2;

        dtype* combined_LHS_RHS = (dtype*) malloc(I_s * cm_size * (R_sym + R) * sizeof(dtype));
          
        for (int j = 0; j < I_s * cm_size; j++) {
            dtype* lhs_src = LHS_list + j * R * R;                    // full matrix block
            dtype* rhs_src = RHS_buf + j * R;                         // full RHS block
            dtype* out     = combined_LHS_RHS + j * (R_sym + R);      // packed block
        
            // Copy upper triangle row-major into packed format
            int idx = 0;
            for (int row = 0; row < R; row++) {
                for (int col = row; col < R; col++) {  // upper triangle
                    out[idx++] = lhs_src[row * R + col];
                }
            }
        
            // Copy RHS
            std::copy(rhs_src, rhs_src + R, out + R_sym);
        }


        //scatter reduce left hand sides and scatter right hand sides in a buffer
        int* Recv_count = (int*) malloc(sizeof(int)*cm_size) ; 
        std::fill(Recv_count, Recv_count + cm_size, I_s * (R_sym + R));

        MPI_Reduce_scatter( MPI_IN_PLACE, combined_LHS_RHS, Recv_count , MPI_DOUBLE, MPI_SUM, slice_comm );

        free(LHS_list) ;
        free(RHS_buf) ;

        free(Recv_count);
        int barrier_block_len;
        dtype* barrier_mat = nullptr;
        
        if (barrier == -1.0 && !proj) {
            // Only regularization is needed
            barrier_block_len = 1;
            barrier_mat = (dtype*) calloc(I_s * cm_size * barrier_block_len, sizeof(dtype));
            
            if (cm_rank == 0) {
                std::copy(regu_arr + b * batched_I,
                          regu_arr + b * batched_I + rows,
                          barrier_mat);
            }
            
        } else {
            // Need arrs + regu
            barrier_block_len = R + 1;
            barrier_mat = (dtype*) calloc(I_s * cm_size * barrier_block_len, sizeof(dtype));
        
            if (cm_rank == 0) {
                int64_t start_idx = b * batched_I;
            
                for (int i = 0; i < rows; i++) {
                    int64_t global_cm_idx = start_idx + i;
            
                    // Copy R entries from arrs[mode]
                    std::copy(arrs[mode] + global_cm_idx * R,
                              arrs[mode] + (global_cm_idx + 1) * R,
                              barrier_mat + i * barrier_block_len);
                    
                    // Copy regularization value
                    barrier_mat[i * barrier_block_len + R] = regu_arr[global_cm_idx];
                }
            }
        }
        
        // Scatter based on actual block length
        MPI_Scatter(barrier_mat,
                    I_s * barrier_block_len, MPI_DOUBLE,
                    (cm_rank == 0 ? MPI_IN_PLACE : barrier_mat),
                    I_s * barrier_block_len, MPI_DOUBLE,
                    0, slice_comm);


        std::vector<dtype> thresholds(I_s, 0.0);
        dtype* row_ptr ;
        dtype* mat_block ;
        dtype regular ;
        for (int i = 0; i < I_s; i++) {
            dtype* block_base = combined_LHS_RHS + i * (R_sym + R);
            dtype* lhs_block = block_base;         // packed upper triangle
            dtype* rhs_block = block_base + R_sym; // full RHS vector

            if (barrier != -1.0 || proj){
                row_ptr = barrier_mat + i * (R + 1);
                mat_block = row_ptr;
                regular = row_ptr[R];
            }
            

        
            if (barrier != -1.0) {
                // Barrier term added to RHS and LHS diagonal
                
                for (int r = 0; r < R; r++) {
                    rhs_block[r] += barrier / (mat_block[r] + 1e-10);
                }
        
                // Update diagonal entries of LHS in packed format
                int idx = 0;
                for (int d = 0; d < R; d++) {
                    dtype mat_val = mat_block[d];
                    dtype diag_update = barrier / (mat_val * mat_val + 1e-10) + regular;
                    lhs_block[idx] += diag_update;
                    idx += R - d;
                }
            }

            else if (proj){
                // Update diagonal using the regularization pointer above
                int idx = 0;
                for (int d = 0; d < R; d++) {
                    lhs_block[idx] += regular;
                    idx += R - d;
                }
            }
            else {
                // No barrier, only regularization
                regular  = barrier_mat[i] ;
                int idx = 0;
                for (int d = 0; d < R; d++) {
                    lhs_block[idx] += regular;
                    idx += R - d;
                }
            }
        
            // Optional constant addition
            if (add_ones) {
                for (int r = 0; r < R; r++) {
                    rhs_block[r] -= 1.0;
                }
            }
        
            if (proj) {
                double sq_sum = 0.0;
                for (int r = 0; r < R; r++) {
                    dtype val = mat_block[r] + rhs_block[r];
                    dtype clipped_val = val > 0.0 ? val : 0.0;
                    dtype diff = mat_block[r] - clipped_val;
                    sq_sum += diff * diff;
                }
                dtype nrm = std::sqrt(sq_sum);
                thresholds[i] = std::min(nrm, epsilon);
            }
        }
          
        
              
        
        t_scatter.stop() ; 
        // Predeclare vectors once, outside all conditionals
        // Declare once outside the loop
        std::vector<int> selected_indices;
        std::vector<int> unselected_indices;
        std::vector<dtype> lhs_selected(R * R, 0.0);
        std::vector<dtype> rhs_selected;

        // Preallocate with maximum possible sizes (R and R x R)
        // Optionally preallocate to max size if you expect reuse
        if (proj) {
            rhs_selected.resize(R);
        }
        // Call local spd solve on I/cm_size different systems locally
        t_solve_lhs.start();
          
        for (int i = 0; i < I_s; i++) {
            if (i + cm_rank * I_s + b * batched_I < I - (T->lens[mode] % T->edge_map[mode].np > 0) + (jr < T->lens[mode] % T->edge_map[mode].np)) {
                dtype* block_base = combined_LHS_RHS + i * (R_sym + R);
                dtype* lhs_block = block_base;         // packed upper triangle
                dtype* rhs_block = block_base + R_sym; // RHS vector
                
                dtype* row_ptr = barrier_mat + i * (R + 1);
                dtype* mat_block = row_ptr;            // arrs[mode][i * R]
                
                
                if (proj) {
        
                    selected_indices.clear();
                    unselected_indices.clear();
                    // negative gradient pointing to make things negative
                    // not selected if current iterate already 0 or <= threshold
                    // Classify entries once
                    for (int r = 0; r < R; r++) {
                        bool is_rhs_nonpos = rhs_block[r] <= 0.0;
                        bool is_mat_zero = mat_block[r] == 0.0;
                        bool is_mat_small = mat_block[r] > 0.0 && mat_block[r] <= thresholds[i];
                    
                        if (is_rhs_nonpos && (is_mat_zero || is_mat_small)) {
                            unselected_indices.push_back(r);  // Collect unselected
                        } else {
                            selected_indices.push_back(r);    // Collect selected
                        }
                    }
        
                    int R_selected = selected_indices.size();

        
                    if (R_selected > 0) {
                        lhs_selected.resize(R_selected * R_selected);
                        rhs_selected.resize(R_selected);
        
        
                        for (int ii = 0; ii < R_selected; ii++) {
                            int idx_i = selected_indices[ii];
                            rhs_selected[ii] = rhs_block[idx_i];
                        
                            for (int jj = ii; jj < R_selected; jj++) {
                                int idx_j = selected_indices[jj];
                        
                                // Access (idx_i, idx_j) from global packed matrix
                                int row = std::min(idx_i, idx_j); // ensure row ≤ col
                                int col = std::max(idx_i, idx_j);
                                int packed_idx = row * R - (row * (row - 1)) / 2 + (col - row);
                        
                                lhs_selected[ii * R_selected + jj] = lhs_block[packed_idx];
                            }
                        }
        
                        // printf("=== [Rank %d, Local i = %d] Projected Solve ===\n", cm_rank, i);
                        // printf("RHS (rhs_selected):\n");
                        // for (int ii = 0; ii < R_selected; ii++) {
                        //     printf("%.6e ", rhs_selected[ii]);
                        // }
                        // printf("\nLHS (lhs_selected, upper triangle only):\n");
                        // for (int ii = 0; ii < R_selected; ii++) {
                        //     for (int jj = 0; jj < R_selected; jj++) {
                        //         if (jj >= ii)
                        //             printf("%.6e ", lhs_selected[ii * R_selected + jj]);
                        //         else
                        //             printf("    .     ");  // visually denote lower triangle
                        //     }
                        //     printf("\n");
                        // }
                        // Solve system
                        CTF_BLAS::posv<dtype>(uplo, &R_selected, &scale,
                                              lhs_selected.data(), &R_selected,
                                              rhs_selected.data(), &R_selected, &info);
        
                        // Write solved entries back in correct order
                        for (int ii = 0; ii < R_selected; ii++) {
                            mat_block[selected_indices[ii]] = rhs_selected[ii];
                        }
                    }
        
                    // Fallback handling for unselected entries
                    for (int idx : unselected_indices) {
                        mat_block[idx] = (mat_block[idx] == 0.0) ? 0.0 : rhs_block[idx];
                    }

                    double sum = 0.0;
                    // We are computing 1/2 d^T H d - rhs^T d
                    for (int ii = 0; ii < R_selected; ++ii) {
                        double di = rhs_selected[ii];  // Packed solution vector
                        int idx_i = selected_indices[ii];
                    
                        // Diagonal term
                        dtype regs = barrier_mat[i * (R + 1) + R];
                        sum += 0.5 * (lhs_selected[ii * R_selected + ii] - regs) * di * di;
                    
                        // Off-diagonal symmetric terms
                        for (int jj = ii + 1; jj < R_selected; ++jj) {
                            double dj = rhs_selected[jj];
                            sum += lhs_selected[ii * R_selected + jj] * di * dj;
                        }
                        // Dot product with original RHS (rhs_orig is full-sized and negated)
                        sum -= di * rhs_block[idx_i];
                    }
                    barrier_mat[i * (R + 1) + R] = (dtype) sum;
        
                } else {
                    // No projection, full system solve
                    // Fill upper triangle from packed `lhs_block`
                    // Posv wants full R xR working memory
                    
                    int packed_idx = 0;
                    for (int ii = 0; ii < R; ii++) {
                        int row_offset = ii * R;
                        for (int jj = ii; jj < R; jj++) {
                            lhs_selected[row_offset + jj] = lhs_block[packed_idx++];
                        }
                    }

                    CTF_BLAS::posv<dtype>(uplo, &R, &scale, lhs_selected.data(), &R, rhs_block, &R, &info);

                }
                
            }
        }
        t_solve_lhs.stop();

        t_gather.start();
        
        if (proj){
            free(combined_LHS_RHS);
            if (cm_rank==0){
              MPI_Gather(MPI_IN_PLACE, I_s*(R+1), MPI_DOUBLE, 
                         barrier_mat, I_s*(R+1), MPI_DOUBLE, 
                         0, slice_comm);
            }
            else{
              MPI_Gather(barrier_mat, I_s*(R+1), MPI_DOUBLE, 
                         NULL, I_s*(R+1), MPI_DOUBLE,
                         0, slice_comm);
            }
              
              
            if (cm_rank == 0) {
                int64_t start_idx = b * batched_I;
            
                for (int i = 0; i < rows; i++) {
                    int64_t global_idx = start_idx + i;
            
                    std::copy(
                        barrier_mat + i * (R + 1),
                        barrier_mat + i * (R + 1) + R,
                        arrs[mode] + global_idx * R
                    );
            
                    regu_arr[global_idx] = barrier_mat[i * (R + 1) + R];
                }
            }
            free(barrier_mat);
        }

        else{
            free(barrier_mat);
            dtype* right_sides = (dtype*) calloc(I_s * cm_size * R, sizeof(dtype));
            
            for (int i = 0; i < rows; i++) {
                dtype* block_base = combined_LHS_RHS + i * (R_sym + R);
                dtype* rhs_block = block_base + R_sym;
                std::copy(rhs_block, rhs_block + R, right_sides + i * R);
            }
            free(combined_LHS_RHS);
            
            if (cm_rank == 0) {
                MPI_Gather(MPI_IN_PLACE, I_s * R, MPI_DOUBLE,
                           right_sides,  I_s * R, MPI_DOUBLE,
                           0, slice_comm);
            } else {
                MPI_Gather(right_sides, I_s * R, MPI_DOUBLE,
                           NULL, I_s * R, MPI_DOUBLE,
                           0, slice_comm);
            }

            if (cm_rank == 0) {
                std::copy(right_sides, right_sides + rows * R,
                          arrs[mode] + b * batched_I * R);
            }
            free(right_sides);
        }
          
        
      }
      
      free(count) ;
        
      
      MPI_Comm_free(&slice_comm);
      t_gather.stop();


      for (int j=0 ; j< T->order ; j++){
        if (j==mode){
          if (redist_mats[j] != NULL){
            mat_list[j]->set_zero();
            if (is_vec){
                mat_list[j]->operator[]("i") += redist_mats[j]->operator[]("i");
            }
            else{
              mat_list[j]->operator[]("ij") += redist_mats[j]->operator[]("ij");
            }
            delete redist_mats[j];
          }
          else {
            IASSERT((dtype*)mat_list[j]->data == arrs[j]);
          } 
        }
        else {
          if (redist_mats[j] != NULL){
            if (redist_mats[j]->data != (char*)arrs[j])
              T->sr->dealloc((char*)arrs[j]);
            delete redist_mats[j];
          } else {
            if (arrs[j] != (dtype*)mat_list[j]->data)
              T->sr->dealloc((char*)arrs[j]);
          }
        }
      }
      if (redist_regu != NULL){
        regu->set_zero();
        regu->operator[]("i") += redist_regu->operator[]("i");
        delete redist_regu;
      }
      else{
        IASSERT((dtype*)regu->data == regu_arr);
      }
    }
    free(redist_mats);

    for (int i = 0; i < T->order; ++i) {
        if (mat_strides[i] != NULL) {
            free(mat_strides[i]);
        }
    }
    free(mat_strides);
    free(par_idx);
    free(phys_phase);
    free(ldas);
    free(arrs);
    if (!T->is_sparse){
      T->sr->pair_dealloc((char*)pairs);
      RHS->sr->pair_dealloc((char*)pairs_RHS);
    }
      
    t_solve_factor.stop();
  }

template<typename dtype>
  void Sparse_add(Tensor<dtype> * T, Tensor<dtype> * M,double alpha, double beta){
    IASSERT(T->order == M->order) ;
    IASSERT(T->is_sparse && M->is_sparse) ;

    int64_t npair1,npair2;
    Pair<dtype> * pairs1 ; 
    Pair<dtype> * pairs2 ;

    npair1 = T->nnz_loc ;
    npair2 = M->nnz_loc ;
    IASSERT(npair1==npair2);
    for (int i=0; i<T->order; i++){
      IASSERT(T->edge_map[i].calc_phys_phase() == M->edge_map[i].calc_phys_phase());
    }
    pairs1 = (Pair<dtype> *)T->data;
    pairs2 = (Pair<dtype> *)M->data;

    /*CTF_int::default_axpy<dtype>(npair1,
                   dtype         alpha,
                   dtype const * X,
                    int           incX,
                    dtype *       Y,
                    int           incY)
                    */
    for(int64_t i=0;i<npair1;i++){
       pairs1[i].d = alpha*pairs1[i].d + beta*pairs2[i].d;
    }

  }

  template<typename dtype>
  void Sparse_mul(Tensor<dtype> * T, Tensor<dtype> * M){
    IASSERT(T->order == M->order) ;
    IASSERT(T->is_sparse && M->is_sparse) ;

    int64_t npair1,npair2;
    Pair<dtype> * pairs1 ; 
    Pair<dtype> * pairs2 ;

    npair1 = T->nnz_loc ;
    npair2 = M->nnz_loc ;
    IASSERT(npair1==npair2);
    for (int i=0; i<T->order; i++){
      IASSERT(T->edge_map[i].calc_phys_phase() == M->edge_map[i].calc_phys_phase());
    }
    pairs1 = (Pair<dtype> *)T->data;
    pairs2 = (Pair<dtype> *)M->data;

    /*CTF_int::default_vec_mul
                    */
    for(int64_t i=0;i<npair1;i++){
       pairs1[i].d *= pairs2[i].d;
    }

  }
  
  template<typename dtype>
  void Sparse_div(Tensor<dtype> * T, Tensor<dtype> * M){
    IASSERT(T->order == M->order) ;
    IASSERT(T->is_sparse && M->is_sparse) ;

    int64_t npair1,npair2;
    Pair<dtype> * pairs1 ; 
    Pair<dtype> * pairs2 ;

    npair1 = T->nnz_loc ;
    npair2 = M->nnz_loc ;
    IASSERT(npair1==npair2);
    for (int i=0; i<T->order; i++){
      IASSERT(T->edge_map[i].calc_phys_phase() == M->edge_map[i].calc_phys_phase());
    }
    pairs1 = (Pair<dtype> *)T->data;
    pairs2 = (Pair<dtype> *)M->data;

    /*CTF_int::default_vec_mul
                    */
    for(int64_t i=0;i<npair1;i++){
       pairs1[i].d /= pairs2[i].d;
    }

  }

  template<typename dtype>
  double Sparse_inner_prod(Tensor<dtype> * T, Tensor<dtype> * M){
    IASSERT(T->order == M->order) ;
    IASSERT(T->is_sparse && M->is_sparse) ;

    int64_t npair1,npair2;
    Pair<dtype> * pairs1 ; 
    Pair<dtype> * pairs2 ;

    npair1 = T->nnz_loc ;
    npair2 = M->nnz_loc ;
    IASSERT(npair1==npair2);
    for (int i=0; i<T->order; i++){
      IASSERT(T->edge_map[i].calc_phys_phase() == M->edge_map[i].calc_phys_phase());
    }
    pairs1 = (Pair<dtype> *)T->data;
    pairs2 = (Pair<dtype> *)M->data;

    double val=0.0;

    /*CTF_int::default_vec_mul
                    */
    for(int64_t i=0;i<npair1;i++){
       val+= pairs1[i].d *pairs2[i].d;
    }

    MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE, MPI_SUM, T->wrld->comm);

    return val;
  }

  template<typename dtype>
  void Sparse_exp(Tensor<dtype> * T, double alpha){
    IASSERT(T->is_sparse) ;

    int64_t npair ;
    Pair<dtype> * pairs ;

    npair = T->nnz_loc ;
    
    pairs = (Pair<dtype> *)T->data;

    /*CTF_int::default_vec_mul
                    */
    for(int64_t i=0;i<npair;i++){
       pairs[i].d = std::exp(alpha*pairs[i].d);
    }

  }

  template<typename dtype>
  void Sparse_log(Tensor<dtype> * T){
    IASSERT(T->is_sparse) ;

    int64_t npair ;
    Pair<dtype> * pairs ;

    npair = T->nnz_loc ;
    
    pairs = (Pair<dtype> *)T->data;

    /*CTF_int::default_vec_mul
                    */
    for(int64_t i=0;i<npair;i++){
       pairs[i].d = std::log(pairs[i].d);
    }

  }

  template<typename dtype>
  void Sparse_sigmoid(Tensor<dtype> * T, double alpha, double eps){
    IASSERT(T->is_sparse) ;

    int64_t npair ;
    Pair<dtype> * pairs ;

    npair = T->nnz_loc ;
    
    pairs = (Pair<dtype> *)T->data;

    /*CTF_int::default_vec_mul
                    */
    for(int64_t i=0;i<npair;i++){
       pairs[i].d = (1 + eps)/ (1.0 + alpha*std::exp(-1.0 *pairs[i].d) + eps) ;
    }

  }

  template<typename dtype>
  void get_index_tensor(Tensor<dtype> * T){
    IASSERT(T->is_sparse) ;
    int64_t npair ;
    Pair<dtype> * pairs ;

    npair = T->nnz_loc ;
    
    pairs = (Pair<dtype> *)T->data;


    for(int64_t i=0;i<npair;i++){
       pairs[i].d = 1.0;
    }

  }
}