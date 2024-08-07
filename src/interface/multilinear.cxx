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
  void Solve_Factor(Tensor<dtype> * T, Tensor<dtype> ** mat_list, Tensor<dtype> * RHS, int mode, double regu, bool aux_mode_first){
    // Mode defines what factor index we're computing

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

      Timer t_solve_remap("Solve_remap_mats");
      t_solve_remap.start();
      for (int i=0; i<T->order; i++){
        Tensor<dtype> mmat;
        Tensor<dtype> * mat ; 
        if (i != mode){
           mat = mat_list[i];
        }
        else{
           mat = RHS;
        }
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
          if (i!=mode){
            mmat = mat_list[i]->slice(slice_st, slice_end);
          }
          else{
            mmat = RHS->slice(slice_st, slice_end);
          }
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
            if (i!= mode)
              arrs[i] = (dtype*)mat_list[i]->data;
            else{
              arrs[i] = (dtype*)mat_list[i]->data;
              mat->read_all(arrs[i], true);
            } 
          }
          else if (i!=mode) {
            arrs[i] = (dtype*)T->sr->alloc(tot_sz);
            mat->read_all(arrs[i], true);
          } else{
            if (is_vec)
              redist_mats[i] = new Vector<dtype>(mat_list[i]->lens[0], 'a'-1, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            else {
              char nonastr[2];
              nonastr[0] = 'a'-1;
              nonastr[1] = 'a'-2;
              redist_mats[i] = new Matrix<dtype>(nrow, ncol, nonastr, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            }
            arrs[i] = (dtype*)redist_mats[i]->data;
            mat->read_all(arrs[i], true);
          }
        } 
        else {
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
            else{
              v->operator[]("i") += RHS->operator[]("i");
            }
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
        pairs_copy[c[indices[i]]-1]=pairs[i];        
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
        if (max_memuse > ((std::ceil(float(I/batches)/cm_size)*cm_size*(R+1)*R) + buffer*R + 10 )*(int64_t)sizeof(dtype) *(int64_t)sizeof(dtype) ) {
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

        //scatter reduce left hand sides and scatter right hand sides in a buffer
        int* Recv_count = (int*) malloc(sizeof(int)*cm_size) ; 
        std::fill(
         Recv_count,
         Recv_count + cm_size,
         I_s*R*R);

        
        t_scatter.start() ; 
        MPI_Reduce_scatter( MPI_IN_PLACE, LHS_list, Recv_count , MPI_DOUBLE, MPI_SUM, slice_comm );
        free(Recv_count);

        for(int64_t i =0 ; i< I_s ; i++){
          for(int r = 0 ; r<R ; r++){
            LHS_list[R*R*i +r*R +r]+=regu ; 
          }
        }
        
        

        dtype * arrs_buf = (dtype *) calloc(I_s*cm_size*R,sizeof(dtype) );

        if (b == batches-1){
          std::copy(&arrs[mode][b*batched_I*R],&arrs[mode][I*R],arrs_buf) ;
        }
        else{
          std::copy(&arrs[mode][b*batched_I*R],&arrs[mode][(b+1)*batched_I*R],arrs_buf) ; 
        }
          
        if (cm_rank == 0){
          MPI_Scatter(arrs_buf, I_s*R, MPI_DOUBLE, MPI_IN_PLACE, I_s*R,  
                     MPI_DOUBLE, 0, slice_comm);
        }
        else{
          MPI_Scatter(NULL, I_s*R, MPI_DOUBLE, arrs_buf, I_s*R,  
                     MPI_DOUBLE, 0, slice_comm);
        }
        t_scatter.stop() ; 
        //call local spd solve on I/cm_size different systems locally (avoid calling solve on padding in lhs)
        t_solve_lhs.start() ;
        for (int i=0; i<I_s; i++){
          if (i + cm_rank*I_s + b*batched_I < I - (T->lens[mode] % T->edge_map[mode].np > 0 )  + (jr< T->lens[mode] % T->edge_map[mode].np ))
            CTF_BLAS::posv<dtype>(uplo,&R,&scale,&LHS_list[i*R*R],&R,&arrs_buf[i*R],&R,&info) ;
        }
        t_solve_lhs.stop();

        free(LHS_list) ;

        //allgather on slice_comm should be used for preserving the mttkrp like mapping
        if (cm_rank==0){
          MPI_Gather(MPI_IN_PLACE, I_s*R, MPI_DOUBLE, arrs_buf, I_s*R, MPI_DOUBLE, 0, slice_comm);
        }
        else{
          MPI_Gather(arrs_buf, I_s*R, MPI_DOUBLE, NULL, I_s*R, MPI_DOUBLE, 0, slice_comm);
        }
        
        std::copy(arrs_buf, arrs_buf + rows*R, &arrs[mode][b*batched_I*R]) ; 
        
        free(arrs_buf) ;
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
    if (!T->is_sparse)
      T->sr->pair_dealloc((char*)pairs);
    t_solve_factor.stop();
  }

  template<typename dtype>
void Solve_Factor_Tucker(Tensor<dtype> * T, Tensor<dtype> ** mat_list, Tensor<dtype> * core, Tensor<dtype> * RHS, int mode, double regu, bool aux_mode_first){
    // Mode defines what factor index we're computing

    Timer t_solve_factor("Solve_Factor_Tucker");
    t_solve_factor.start();
    int k[T->order];
    bool is_vec = mat_list[0]->order == 1;
    if (!is_vec){
      for (int i=0; i<T->order; i++){
        k[i] = mat_list[i]->lens[1-aux_mode_first];
      }
    }
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
      Timer t_solve_remap("Solve_remap_mats");
      t_solve_remap.start();
      for (int i=0; i<T->order; i++){
        k_start += kd;
        kd = k[i]/div + (d < k[i]%div);
        int k_end = k_start + kd;
        Tensor<dtype> mmat;
        Tensor<dtype> * mat ; 
        if (i != mode){
           mat = mat_list[i];
        }
        else{
           mat = RHS;
        }
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
          if (i!=mode){
            mmat = mat_list[i]->slice(slice_st, slice_end);
          }
          else{
            mmat = RHS->slice(slice_st, slice_end);
          }
          mat = &mmat;
        } else if (!is_vec) {
          if (aux_mode_first){
            mat_strides[2*i+0] = k[i];
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
            if (i!= mode)
              arrs[i] = (dtype*)mat_list[i]->data;
            else{
              arrs[i] = (dtype*)mat_list[i]->data;
              mat->read_all(arrs[i], true);
            } 
          }
          else if (i!=mode) {
            arrs[i] = (dtype*)T->sr->alloc(tot_sz);
            mat->read_all(arrs[i], true);
          } else{
            if (is_vec)
              redist_mats[i] = new Vector<dtype>(mat_list[i]->lens[0], 'a'-1, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            else {
              char nonastr[2];
              nonastr[0] = 'a'-1;
              nonastr[1] = 'a'-2;
              redist_mats[i] = new Matrix<dtype>(nrow, ncol, nonastr, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            }
            arrs[i] = (dtype*)redist_mats[i]->data;
            mat->read_all(arrs[i], true);
          }
        } 
        else {
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
            else{
              v->operator[]("i") += RHS->operator[]("i");
            }
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
      t_trav.start() ; 
      

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
        pairs_copy[c[indices[i]]-1]=pairs[i];        
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
      int loop_size ;

      while (true){
        if (max_memuse > ((std::ceil(float(I/batches)/cm_size)*cm_size*(R+1)*R) + buffer*R + 10 )*(int64_t)sizeof(dtype)  ) {
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
        Timer t_scatter("Scatter and Sc_reduce");

        

        dtype * H = (dtype *) calloc(buffer*R,sizeof(dtype) ) ;
        dtype * input_buf = (dtype *) malloc(len_core*sizeof(dtype));
        dtype * output_buf = (dtype *) malloc(len_core*sizeof(dtype));
        // dtype * output1 = (dtype *) calloc(l_core1->lens[1]*l_core1->lens[2]*l_core1->lens[0],sizeof(dtype));
        // dtype * output2 = (dtype *) calloc(l_core1->lens[2],sizeof(dtype));
        // dtype * test_mat = (dtype *) calloc(2*3*4, sizeof(dtype));
        // dtype * vec1 = (dtype *) calloc(4, sizeof(dtype));
        // dtype * vec2 = (dtype *) calloc(3, sizeof(dtype));
        // printf("----------\n");
        
        // for (int p=0; p<2*3*4 ; p++){
        //   test_mat[p] = p+1 ;
        // }
        // for (int p=0 ; p<4 ; p++){
        //   vec1[p] = p+1 ;
        // }

        // M = 2*3;
        // N = 4;

        // CTF_BLAS::gemv<dtype>(trans, &M,&N, &alpha, test_mat, &M, vec1,&INCX,&beta,test_mat,&INCY);

        // M = 2;
        // N = 3;
        // CTF_BLAS::gemv<dtype>(trans, &M,&N, &alpha, test_mat, &M, vec1,&INCX,&beta,vec2,&INCY);
        // for (int p=0 ; p<2*3 ; p++){
        //   cout<<test_mat[p]<<endl;
        // }
        // printf("-----\n");

        // for (int p=0 ; p<2 ; p++){
        //   cout<<vec2[p]<<endl;
        // }
        // fflush(stdout);
        // printf("-----\n");


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
              
              std::transform(output_buf, output_buf + R, output_buf,
                   [scalar](double val) { return val * scalar; }) ;
              
              std::copy(output_buf, output_buf + R, H + q*R);

              
              // for (int i =0 ; i<T->order ; i++){
              //   int64_t ke= key/ldas[i];
              //   indices[i] = (ke%T->lens[i])/phys_phase[i];
              // }
              // if (mode ==0){
              //       //H[q*R + a] += l_core_data[a + b* l_core1->lens[0] + c*l_core1->lens[0]*l_core1->lens[1]]*arrs[1][indices[1]*l_core1->lens[1] + b]*arrs[2][indices[2]*l_core1->lens[2]+c];
              //       M = l_core1->lens[0]*l_core1->lens[1];
              //       N = l_core1->lens[2] ;
              //       CTF_BLAS::gemv<dtype>(trans, &M,&N, &alpha, l_core_data, &M,&arrs[2][indices[2]*l_core1->lens[2]],&INCX,&beta,output_buf,&INCY);
              //       std::copy(output_buf, output_buf + M, input_buf);

              //       M = l_core1->lens[0];
              //       N = l_core1->lens[1] ;
              //       CTF_BLAS::gemv<dtype>(trans, &M,&N, &alpha, input_buf, &M,&arrs[1][indices[1]*l_core1->lens[1]],&INCX,&beta,H+q*R,&INCY);
              // }
              // else if (mode ==1){
              //       M = l_core1->lens[0]*l_core1->lens[1];
              //       N = l_core1->lens[2] ;
              //       CTF_BLAS::gemv<dtype>(trans, &M,&N, &alpha, l_core_data, &M,&arrs[2][indices[2]*l_core1->lens[2]],&INCX,&beta,output_buf,&INCY);
              //       std::copy(output_buf, output_buf + M, input_buf);

              //       M = l_core1->lens[1];
              //       N = l_core1->lens[0];
              //       CTF_BLAS::gemv<dtype>(trans_gem, &N,&M, &alpha, input_buf, &N,&arrs[0][indices[0]*l_core1->lens[0]],&INCX,&beta,output_buf,&INCY);
              //       std::copy(output_buf, output_buf + R, H + q*R);
              // }
              // else{
              //   M = l_core1->lens[1]*l_core1->lens[2];
              //   N = l_core1->lens[0] ;
              //   CTF_BLAS::gemv<dtype>(trans_gem, &N,&M, &alpha, l_core_data, &N,&arrs[0][indices[0]*l_core1->lens[0]],&INCX,&beta,output_buf,&INCY);
              //   std::copy(output_buf, output_buf + M, input_buf);
              //   M = l_core1->lens[1];
              //   N = l_core1->lens[2];
              //   CTF_BLAS::gemv<dtype>(trans_gem, &M,&N, &alpha, input_buf, &M,&arrs[1][indices[1]*l_core1->lens[1]],&INCX,&beta,H+q*R,&INCY);
              // }

              // for (int c=0 ; c< l_core1->lens[2] ; c++){
              //   for (int b=0 ; b< l_core1->lens[1] ; b++){
              //     for (int a=0 ;a< l_core1->lens[0] ; a++){
              //       if (mode ==1){
              //         H[q*R + b] += l_core_data[a + b* l_core1->lens[0] + c*l_core1->lens[0]*l_core1->lens[1]]*arrs[0][indices[0]*l_core1->lens[0] + a]*arrs[2][indices[2]*l_core1->lens[2]+c];
              //       }
              //       else{
              //         H[q*R + c] += l_core_data[a + b* l_core1->lens[0] + c*l_core1->lens[0]*l_core1->lens[1]]*arrs[0][indices[0]*l_core1->lens[0] + a]*arrs[1][indices[1]*l_core1->lens[1] + b];
              //       }
              //     }
              //   }
              // }
              
            }
            CTF_BLAS::syrk<dtype>(uplo,trans,&R,&loop_size,&alpha,H,&R,&alpha,&LHS_list[j*R*R],&R) ;
            std::fill(
                H,
                H+ loop_size*R,
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



        
        t_LHS_work.stop();

        //scatter reduce left hand sides and scatter right hand sides in a buffer
        int* Recv_count = (int*) malloc(sizeof(int)*cm_size) ; 
        std::fill(
         Recv_count,
         Recv_count + cm_size,
         I_s*R*R);

        
        t_scatter.start() ; 
        MPI_Reduce_scatter( MPI_IN_PLACE, LHS_list, Recv_count , MPI_DOUBLE, MPI_SUM, slice_comm );
        free(Recv_count);

        for(int64_t i =0 ; i< I_s ; i++){
          for(int r = 0 ; r<R ; r++){
            LHS_list[R*R*i +r*R +r]+=regu ; 
          }
        }
        

        dtype * arrs_buf = (dtype *) calloc(I_s*cm_size*R,sizeof(dtype) );

        if (b == batches-1){
          std::copy(&arrs[mode][b*batched_I*R],&arrs[mode][I*R],arrs_buf) ;
        }
        else{
          std::copy(&arrs[mode][b*batched_I*R],&arrs[mode][(b+1)*batched_I*R],arrs_buf) ; 
        }
          
        if (cm_rank == 0){
          MPI_Scatter(arrs_buf, I_s*R, MPI_DOUBLE, MPI_IN_PLACE, I_s*R,  
                     MPI_DOUBLE, 0, slice_comm);
        }
        else{
          MPI_Scatter(NULL, I_s*R, MPI_DOUBLE, arrs_buf, I_s*R,  
                     MPI_DOUBLE, 0, slice_comm);
        }
        t_scatter.stop() ; 
        //call local spd solve on I/cm_size different systems locally (avoid calling solve on padding in lhs)
        t_solve_lhs.start() ;

        for (int i=0; i<I_s; i++){
          if (i + cm_rank*I_s + b*batched_I < I - (T->lens[mode] % T->edge_map[mode].np > 0 )  + (jr< T->lens[mode] % T->edge_map[mode].np ))
            CTF_BLAS::posv<dtype>(uplo,&R,&scale,&LHS_list[i*R*R],&R,&arrs_buf[i*R],&R,&info) ;
        }
        t_solve_lhs.stop();


        free(LHS_list) ;

        //allgather on slice_comm should be used for preserving the mttkrp like mapping
        if (cm_rank==0){
          MPI_Gather(MPI_IN_PLACE, I_s*R, MPI_DOUBLE, arrs_buf, I_s*R, MPI_DOUBLE, 0, slice_comm);
        }
        else{
          MPI_Gather(arrs_buf, I_s*R, MPI_DOUBLE, NULL, I_s*R, MPI_DOUBLE, 0, slice_comm);
        }
        
        std::copy(arrs_buf, arrs_buf + rows*R, &arrs[mode][b*batched_I*R]) ; 
        
        free(arrs_buf) ;
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
    if (!T->is_sparse)
      T->sr->pair_dealloc((char*)pairs);
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
  void Sparse_exp(Tensor<dtype> * T){
    IASSERT(T->is_sparse) ;

    int64_t npair ;
    Pair<dtype> * pairs ;

    npair = T->nnz_loc ;
    
    pairs = (Pair<dtype> *)T->data;

    /*CTF_int::default_vec_mul
                    */
    for(int64_t i=0;i<npair;i++){
       pairs[i].d = std::exp(pairs[i].d);
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
