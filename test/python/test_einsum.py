#!/usr/bin/env python

import unittest
import numpy as np
import ctf
import os
import sys
import numpy.linalg as la

def allclose(a, b):
    return abs(ctf.to_nparray(a) - ctf.to_nparray(b)).sum() <= 1e-5


class KnowValues(unittest.TestCase):
    def test_einsum_views(self):
        a0 = np.arange(27.).reshape(3,3,3)
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(ctf.einsum("jii->ij", a1), np.einsum("jii->ij", a0)))
        self.assertTrue(allclose(ctf.einsum("iii->i", a1), np.einsum("iii->i", a0)))
        self.assertTrue(allclose(ctf.einsum("iii", a1), np.einsum("iii", a0)))

        a0 = np.arange(6.)
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(ctf.einsum("i,i,i->i", a1, a1, a1),
                                 np.einsum("i,i,i->i", a0, a0, a0)))

        # swap axes
        a0 = np.arange(24.).reshape(4,3,2)
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(ctf.einsum("ijk->kji", a1),
                                 np.einsum("ijk->kji", a0)))

    def test_einsum_sums(self):
        # outer(a,b)
        for n in range(1, 17):
            a0 = np.arange(3, dtype=np.double)+1
            b0 = np.arange(n, dtype=np.double)+1
            a1 = ctf.astensor(a0)
            b1 = ctf.astensor(b0)
            self.assertTrue(allclose(ctf.einsum("i,j", a1, b1), np.outer(a0, b0)))

        # matvec(a,b) / a.dot(b) where a is matrix, b is vector
        for n in range(1, 17):
            a0 = np.arange(4*n, dtype=np.double).reshape(n, 4)
            b0 = np.arange(n, dtype=np.double)
            a1 = ctf.astensor(a0)
            b1 = ctf.astensor(b0)
            self.assertTrue(allclose(ctf.einsum("ji,j", a1, b1), np.dot(b0.T, a0)))
            self.assertTrue(allclose(ctf.einsum("ji,j->", a1, b1), np.dot(b0.T, a0).sum()))

        # matmat(a,b) / a.dot(b) where a is matrix, b is matrix
        for n in range(1, 17):
            a0 = np.arange(4*n, dtype=np.double).reshape(n, 4)
            b0 = np.arange(6*n, dtype=np.double).reshape(n, 6)
            a1 = ctf.astensor(a0)
            b1 = ctf.astensor(b0)
            self.assertTrue(allclose(ctf.einsum("ji,jk", a1, b1), np.dot(a0.T, b0)))
            self.assertTrue(allclose(ctf.einsum("ji,jk->", a1, b1), np.dot(a0.T, b0).sum()))


        # matrix triple product (note this is not currently an efficient
        # way to multiply 3 matrices)
        a0 = np.arange(12.).reshape(3, 4)
        b0 = np.arange(20.).reshape(4, 5)
        c0 = np.arange(30.).reshape(5, 6)
        a1 = ctf.astensor(a0)
        b1 = ctf.astensor(b0)
        c1 = ctf.astensor(c0)
        self.assertTrue(allclose(ctf.einsum("ij,jk,kl", a1, b1, c1),
                                 np.einsum("ij,jk,kl", a0, b0, c0)))

        # tensordot(a, b)
        a0 = np.arange(27.).reshape(3, 3, 3)
        b0 = np.arange(27.).reshape(3, 3, 3)
        a1 = ctf.astensor(a0)
        b1 = ctf.astensor(b0)
        self.assertTrue(allclose(ctf.einsum("ijk, jli -> kl", a1, b1),
                                 np.einsum("ijk, jli -> kl", a0, b0)))
        self.assertTrue(allclose(ctf.einsum("ijk, jli -> lk", a1, b1),
                                 np.einsum("ijk, jli -> lk", a0, b0)))
        self.assertTrue(allclose(ctf.einsum("ikj, jli -> kl", a1, b1),
                                 np.einsum("ikj, jli -> kl", a0, b0)))
        self.assertTrue(allclose(ctf.einsum("kij, lij -> lk", a1, b1),
                                 np.einsum("kij, lij -> lk", a0, b0)))

    def test_einsum_misc(self):
        # The iterator had an issue with buffering this reduction
        a0 = np.ones((5, 12, 4, 2, 3))
        b0 = np.ones((5, 12, 11))
        a1 = ctf.astensor(a0)
        b1 = ctf.astensor(b0)
        self.assertTrue(allclose(ctf.einsum('ijklm,ijn->', a1, b1),
                                 np.einsum('ijklm,ijn->', a0, b0)))
        self.assertTrue(allclose(ctf.einsum('ijklm,ijn,ijn->', a1, b1, b1),
                                 #np.einsum('ijklm,ijn,ijn->', a0, b0, b0)))
                                 np.einsum('ijklm,ijn->', a0, b0)))

        # inner loop implementation
        a0 = np.arange(1., 3.)
        b0 = np.arange(1., 5.).reshape(2, 2)
        c0 = np.arange(1., 9.).reshape(4, 2)
        a1 = ctf.astensor(a0)
        b1 = ctf.astensor(b0)
        c1 = ctf.astensor(c0)
        self.assertTrue(allclose(ctf.einsum('x,yx,zx->xzy', a1, b1, c1),
                                 np.einsum('x,yx,zx->xzy', a0, b0, c0)))

        a0 = np.random.normal(0, 1, (5, 5, 5, 5))
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(ctf.einsum('aabb->ab', a1),
                                 np.einsum('aabb->ab', a0)))

        a0 = np.arange(25.).reshape(5, 5)
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(ctf.einsum('mi,mi,mi->m', a1, a1, a1),
                                 np.einsum('mi,mi,mi->m', a0, a0, a0)))

    def test_einsum_mix_types(self):
        a0 = np.random.random((5, 1, 4, 2, 3)).astype(complex)+1j
        b0 = np.random.random((5, 1, 11)).astype(np.float32)
        a1 = ctf.astensor(a0)
        b1 = ctf.astensor(b0)
        c0 = np.einsum('ijklm,ijn->', a0, b0)
        c1 = ctf.einsum('ijklm,ijn->', a1, b1)
        self.assertTrue(allclose(c0, c1))
        c1 = ctf.einsum('ijklm,ijn->', a1, b0)
        self.assertTrue(allclose(c0, c1))
        c1 = ctf.einsum('ijklm,ijn->', a0, b1)
        self.assertTrue(allclose(c0, c1))
        c0 = np.einsum('ijklm,->ij', a0, 3.)
        c1 = ctf.einsum('ijklm,->ij', a1, 3.)
        self.assertTrue(allclose(c0, c1))

    def test_MTTKRP_vec(self):
        for N in range(2,5):
            lens = np.random.randint(3, 4, N)
            A = ctf.tensor(lens)
            A.fill_sp_random(-1.,1.,.5)
            mats = []
            for i in range(N):
                mats.append(ctf.random.random([lens[i]]))
            for i in range(N):
                ctr = A.i("ijklm"[0:N])
                for j in range(N):
                    if i != j:
                        ctr *= mats[j].i("ijklm"[j])
                ans = ctf.zeros(mats[i].shape)
                ans.i("ijklm"[i]) << ctr
                ctf.MTTKRP(A, mats, i)
                self.assertTrue(allclose(ans, mats[i]))


    def test_MTTKRP_mat(self):
        k = 9
        for N in range(2,5):
            lens = np.random.randint(3, 4, N)
            A = ctf.tensor(lens)
            A.fill_sp_random(-1.,1.,.5)
            mats = []
            for i in range(N):
                mats.append(ctf.random.random([lens[i],k]))
            for i in range(N):
                ctr = A.i("ijklm"[0:N])
                for j in range(N):
                    if i != j:
                        ctr *= mats[j].i("ijklm"[j]+"r")
                ans = ctf.zeros(mats[i].shape)
                ans.i("ijklm"[i]+"r") << ctr
                ctf.MTTKRP(A, mats, i)
                self.assertTrue(allclose(ans, mats[i]))

    # def test_Solve_Factor_mat(self):
    #     R = 10
    #     for N in range(3,6):
    #         mats = []
    #         num = np.random.randint(N)
    #         lens = np.random.randint(10,20,N)
    #         regu = 1e-04
    #         for i in range(N):
    #             if i !=num:
    #                 mats.append(ctf.random.random([lens[i],R]))
    #             else:
    #                 mats.append(ctf.tensor([lens[i],R]))
    #         RHS = ctf.random.random([lens[num],R])
    #         A = ctf.tensor(lens,sp=1)
    #         A.fill_sp_random(1., 1., 0.5)
    #         lst_mat = []
    #         T_inds = "".join([chr(ord('a')+i) for i in range(A.ndim)])
    #         einstr=""
    #         for i in range(N):
    #             if i != num:
    #                 einstr+=chr(ord('a')+i) + 'r' + ','
    #                 lst_mat.append(mats[i].to_nparray())
    #                 einstr+=chr(ord('a')+i) + 'z' + ','
    #                 lst_mat.append(mats[i].to_nparray())
    #         einstr+= T_inds + "->"+chr(ord('a')+num)+'rz'
    #         lst_mat.append(A.to_nparray())
    #         lhs_np =np.einsum(einstr,*lst_mat,optimize=True)
    #         rhs_np = RHS.to_nparray()
    #         ans = np.zeros_like(rhs_np)
    #         for i in range(mats[num].shape[0]):
    #             ans[i,:] = la.solve(lhs_np[i]+regu*np.eye(R),rhs_np[i,:])
    #         ctf.Solve_Factor(A,mats,RHS,num,regu)
    #         self.assertTrue(np.allclose(ans, mats[num].to_nparray()))
    
    def test_Solve_Factor_mat(self):
        R = 8
        for N in range(3,6):
            mats = []
            num = np.random.randint(N)
            lens = np.random.randint(10,20,N)
            regu = 1e-4
            for i in range(N):
                mats.append(ctf.random.random([lens[i],R]))
            RHS = ctf.tensor(lens,sp=1)
            RHS.fill_sp_random(-1., 1., 0.2)
            A = RHS.copy()
            ctf.get_index_tensor(A)
            lst_mat = []
            T_inds = "".join([chr(ord('a')+i) for i in range(A.ndim)])
            einstr=""
            for i in range(N):
                if i != num:
                    einstr+=chr(ord('a')+i) + 'r' + ','
                    lst_mat.append(mats[i].to_nparray())
                    einstr+=chr(ord('a')+i) + 'z' + ','
                    lst_mat.append(mats[i].to_nparray())
            einstr+= T_inds + "->"+chr(ord('a')+num)+'rz'
            lst_mat.append(A.to_nparray())
            lhs_np = np.einsum(einstr,*lst_mat,optimize=True)

            lst_mat= []
            T_inds = "".join([chr(ord('a')+i) for i in range(A.ndim)])
            einstr=""
            for i in range(N):
                if i != num:
                    einstr+=chr(ord('a')+i) + 'r' + ','
                    lst_mat.append(mats[i].to_nparray())
            einstr+= T_inds + "->"+chr(ord('a')+num)+'r'
            lst_mat.append(RHS.to_nparray())
            rhs_np = np.einsum(einstr,*lst_mat,optimize=True)
            barrier = None
            proj = True
            add_ones = True
            eps = 0.9
            mats_np = mats[num].to_nparray()
            if add_ones:
                rhs_np -= np.ones_like(rhs_np)
            ans = np.zeros_like(rhs_np)
            nrms = la.norm(( mats_np -  (mats_np + rhs_np).clip(0)) , axis=1)
            thresholds = np.minimum(nrms , eps)
            regu_arr = regu*ctf.random.random(mats[num].shape[0])
            for i in range(mats_np.shape[0]):
                # Cache original RHS before modifications
                rhs_orig = rhs_np[i].copy()
            
                if barrier is not None:
                    assert proj is False, "Projection must be off when barrier is not none"
                    reg_matrix = regu_arr.to_nparray()[i] * np.eye(R)
                    barrier_diag = np.diag(barrier / (mats_np[i] * mats_np[i] + 1e-10))
                    lhs = lhs_np[i] + reg_matrix + barrier_diag
                    rhs = rhs_np[i] + barrier/ (mats_np[i] + 1e-10)
            
            
                    ans[i] = la.solve(lhs, rhs)
            
                elif proj:
                    assert barrier is None, "barrier must be None when projection is enabled"
            
                    # Selection logic
                    is_rhs_nonpos = rhs_np[i] <= 0.0
                    is_mat_zero = mats_np[i] == 0.0
                    is_mat_small = (mats_np[i] > 0.0) & (mats_np[i] <= thresholds[i])
                    unselected_mask = is_rhs_nonpos & (is_mat_zero | is_mat_small)
                    selected_mask = ~unselected_mask
                    selected_indices = np.where(selected_mask)[0]
            
                    if selected_indices.size > 0:
                        lhs_active = lhs_np[i][np.ix_(selected_indices, selected_indices)]
                        rhs_active = rhs_orig[selected_indices]
                        lhs = lhs_active + regu_arr.to_nparray()[i] * np.eye(selected_indices.size)
                        rhs = rhs_active
            
                        sol_active = la.solve(lhs, rhs)
                        ans[i, selected_indices] = sol_active
            
                    for idx in np.where(unselected_mask)[0]:
                        ans[i, idx] = rhs_orig[idx] if mats_np[i, idx] > 0.0 else 0.0
            
                else:
                    assert barrier is None, "barrier must be None when projection is enabled"
                    assert proj is False, "No projection as well"
                    lhs = lhs_np[i] + regu_arr.to_nparray()[i] * np.eye(R)
                    rhs = rhs_np[i]
            
            
                    ans[i] = la.solve(lhs, rhs)
            ctf.Solve_Factor(A,mats,RHS,num,regu_arr,eps, barrier, proj, add_ones)
            
            self.assertTrue(np.allclose(ans, mats[num].to_nparray()))

    # def test_Solve_Factor_with_RHS(self):
    #     R = 10
    #     for N in range(3,6):
    #         mats = []
    #         num = np.random.randint(N)
    #         lens = np.random.randint(10,20,N)
    #         regu = 1e-04
    #         for i in range(N):
    #             mats.append(ctf.random.random([lens[i],R]))
    #         A = ctf.tensor(lens,sp=1)
    #         A.fill_sp_random(0.1, 1., 0.2)
    #         lst_mat = []
    #         T_inds = "".join([chr(ord('a')+i) for i in range(A.ndim)])
    #         einstr=""
    #         for i in range(N):
    #             if i != num:
    #                 einstr+=chr(ord('a')+i) + 'r' + ','
    #                 lst_mat.append(mats[i].to_nparray())
    #                 einstr+=chr(ord('a')+i) + 'z' + ','
    #                 lst_mat.append(mats[i].to_nparray())
    #         einstr+= T_inds + "->"+chr(ord('a')+num)+'rz'
    #         lst_mat.append(A.to_nparray())
    #         lhs_np = np.einsum(einstr,*lst_mat,optimize=True)
    #         RHS = ctf.random.random(mats[num].shape)
    #         rhs_np = RHS.to_nparray()
    #         barrier = 0.01
    #         mats_np = mats[num].to_nparray()
    #         rhs_np += barrier/(mats_np + 1e-10)
    #         ans = np.zeros_like(rhs_np)
    #         for i in range(mats_np.shape[0]):
    #             ans[i,:] = la.solve(lhs_np[i]+regu*np.eye(R) + np.diag(barrier/ (mats_np[i,:]*mats_np[i,:] + 1e-10) ),
    #                                 rhs_np[i,:] )
    #         ctf.Solve_Factor_with_RHS(A,mats,RHS,num,regu,barrier)
    #         self.assertTrue(np.allclose(ans, mats[num].to_nparray()))
            
    def test_Solve_Factor_Tucker(self):
        for N in range(3,6):
            mats = []
            num = np.random.randint(N)
            lens = np.random.randint(10,20,N)
            ranks = np.random.randint(2,6,N)
            regu = 1e-4
            for i in range(N):
                mats.append(ctf.random.random([lens[i],ranks[i]]))
            RHS = ctf.tensor(lens,sp=1)
            RHS.fill_sp_random(-1., 1., 0.2)
            core = ctf.random.random(ranks)
            A = RHS.copy()
            ctf.get_index_tensor(A)
            T_inds = "".join([chr(ord('a')+i) for i in range(A.ndim)])
            core_inds = "".join([chr(ord('r')+i) for i in range(core.ndim)])
            core_inds2 = "".join([chr(ord('l')+i) for i in range(core.ndim)])
            einstr= T_inds+','
            lst_mat = []
            lst_mat.append(A.to_nparray())
            for i in range(N):
                if i != num:
                    einstr+=chr(ord('a')+i) + chr(ord('r')+i) + ','
                    lst_mat.append(mats[i].to_nparray())
                    einstr+=chr(ord('a')+i) + chr(ord('l')+i) + ','
                    lst_mat.append(mats[i].to_nparray())
            lst_mat.append(core.to_nparray())
            lst_mat.append(core.to_nparray())
            einstr+= core_inds + ','+core_inds2+'->'+chr(ord('a')+num)+chr(ord('r')+num)+chr(ord('l')+num)
            lhs_np = np.einsum(einstr,*lst_mat,optimize=True)

            lst_mat= []
            T_inds = "".join([chr(ord('a')+i) for i in range(A.ndim)])
            einstr=""
            for i in range(N):
                if i != num:
                    einstr+=chr(ord('a')+i) + chr(ord('r')+i) + ','
                    lst_mat.append(mats[i].to_nparray())
            core_inds = "".join([chr(ord('r')+i) for i in range(core.ndim)])
            lst_mat.append(core.to_nparray())
            einstr+= core_inds + ','
            einstr+= T_inds + "->"+chr(ord('a')+num)+ chr(ord('r')+num)
            lst_mat.append(RHS.to_nparray())
            rhs_np = np.einsum(einstr,*lst_mat,optimize=True)
            barrier = None
            proj = True
            add_ones = True
            eps = 0.9
            mats_np = mats[num].to_nparray()
            if add_ones:
                rhs_np -= np.ones_like(rhs_np)
            ans = np.zeros_like(rhs_np)
            nrms = la.norm(( mats_np -  (mats_np + rhs_np).clip(0)) , axis=1)
            thresholds = np.minimum(nrms , eps)
            regu_arr = regu*ctf.random.random(mats[num].shape[0])
            for i in range(mats_np.shape[0]):
                # Cache original RHS before modifications
                rhs_orig = rhs_np[i].copy()
            
                if barrier is not None:
                    assert proj is False, "Projection must be off when barrier is not none"
                    reg_matrix = regu_arr.to_nparray()[i] * np.eye(R)
                    barrier_diag = np.diag(barrier / (mats_np[i] * mats_np[i] + 1e-10))
                    lhs = lhs_np[i] + reg_matrix + barrier_diag
                    rhs = rhs_np[i] + barrier/ (mats_np[i] + 1e-10)
            
                    ans[i] = la.solve(lhs, rhs)
            
                elif proj:
                    assert barrier is None, "barrier must be None when projection is enabled"
            
                    # Selection logic
                    is_rhs_nonpos = rhs_np[i] <= 0.0
                    is_mat_zero = mats_np[i] == 0.0
                    is_mat_small = (mats_np[i] > 0.0) & (mats_np[i] <= thresholds[i])
                    unselected_mask = is_rhs_nonpos & (is_mat_zero | is_mat_small)
                    selected_mask = ~unselected_mask
                    selected_indices = np.where(selected_mask)[0]
            
                    if selected_indices.size > 0:
                        lhs_active = lhs_np[i][np.ix_(selected_indices, selected_indices)]
                        rhs_active = rhs_orig[selected_indices]
                        lhs = lhs_active + regu_arr.to_nparray()[i] * np.eye(selected_indices.size)
                        rhs = rhs_active
            
                        sol_active = la.solve(lhs, rhs)
                        ans[i, selected_indices] = sol_active
            
                    for idx in np.where(unselected_mask)[0]:
                        ans[i, idx] = rhs_orig[idx] if mats_np[i, idx] > 0.0 else 0.0
            
                else:
                    assert barrier is None, "barrier must be None when projection is enabled"
                    assert proj is False, "No projection as well"
                    lhs = lhs_np[i] + regu_arr.to_nparray()[i] * np.eye(R)
                    rhs = rhs_np[i]
            
            
                    ans[i] = la.solve(lhs, rhs)
            ctf.Solve_Factor_Tucker(A,mats,core,RHS,num,regu_arr,eps, barrier, proj, add_ones)
            self.assertTrue(np.allclose(ans, mats[num].to_nparray()))

    def test_TTTP_vec(self):
        A = np.random.random((4, 3, 5))
        u = np.random.random((4,))
        v = np.random.random((5,))
        ans = np.einsum("ijk,i,k->ijk",A,u,v)
        cA = ctf.astensor(A)
        cu = ctf.astensor(u)
        cv = ctf.astensor(v)
        cans = ctf.TTTP(cA,[cu,None,cv])
        self.assertTrue(allclose(ans, cans))

    def test_TTTP_mat(self):
        A = np.random.random((5, 1, 4, 2, 3))
        u = np.random.random((5, 3))
        v = np.random.random((1, 3))
        w = np.random.random((4, 3))
        x = np.random.random((2, 3))
        y = np.random.random((3, 3))
        ans = np.einsum("ijklm,ia,ja,ka,la,ma->ijklm",A,u,v,w,x,y)
        cA = ctf.astensor(A)
        cu = ctf.astensor(u)
        cv = ctf.astensor(v)
        cw = ctf.astensor(w)
        cx = ctf.astensor(x)
        cy = ctf.astensor(y)
        cans = ctf.TTTP(cA,[cu,cv,cw,cx,cy])
        self.assertTrue(allclose(ans, cans))

    def test_sp_TTTP_mat(self):
        A = ctf.tensor((5, 1, 4, 2, 3),sp=True)
        A.fill_sp_random(0.,1.,.2)
        u = ctf.random.random((5, 3))
        v = ctf.random.random((1, 3))
        w = ctf.random.random((4, 3))
        x = ctf.random.random((2, 3))
        y = ctf.random.random((3, 3))
        ans = ctf.einsum("ijklm,ia,ja,ka,la,ma->ijklm",A,u,v,w,x,y)
        cans = ctf.TTTP(A,[u,v,w,x,y])
        self.assertTrue(allclose(ans, cans))

    def test_tree_ctr(self):
        X = []
        for i in range(10):
            X.append(ctf.random.random((8, 8)))
        scl = ctf.einsum("ab,ac,ad,ae,af,bg,cg,dg,eg,fg",X[0],X[1],X[2],X[3],X[4],X[5],X[6],X[7],X[8],X[9])
        C = ctf.dot(X[0],X[5])
        for i in range(1,5):
            C = C * ctf.dot(X[i],X[5+i])
        scl2 = ctf.vecnorm(C,1)
        self.assertTrue(np.abs(scl-scl2)<1.e-4) 


def run_tests():
    np.random.seed(5330);
    wrld = ctf.comm()
    if wrld.rank() != 0:
        result = unittest.TextTestRunner(stream = open(os.devnull, 'w')).run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    else:
        print("Tests for einsum")
        result = unittest.TextTestRunner().run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    return result

if __name__ == "__main__":
    result = run_tests()
    ctf.MPI_Stop()
    sys.exit(not result)
    