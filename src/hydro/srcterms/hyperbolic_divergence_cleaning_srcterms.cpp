//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hyperbolic_divergence_cleaning_srcterm.cpp
//! \brief source terms related to Dedner's divB cleaning

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../gravity/gravity.hpp"
#include "../../mesh/mesh.hpp"
#include "../hydro.hpp"
#include "hydro_srcterms.hpp"

//----------------------------------------------------------------------------------------
//! \fn void HydroSourceTerms::HyperbolicDivergenceCleaning
//! \brief Adds source terms for hyperbolic divergence cleaning 
//! \note

void HydroSourceTerms::HyperbolicDivergenceCleaning(const Real dt,
                       const AthenaArray<Real> *flx, const AthenaArray<Real> &prim,
                       AthenaArray<Real> &cons) {
  MeshBlock *pmb = pmy_hydro_->pmy_block;
  Mesh *pm = pmb->pmy_mesh;
  Coordinates *pco = pmb->pcoord;
  const AthenaArray<Real> &x1flux = flx[X1DIR];
  const AthenaArray<Real> &x2flux = flx[X2DIR];
  const AthenaArray<Real> &x3flux = flx[X3DIR];
  Real ch = -pmy_hydro_->ch_;

  // assuming L = 1
  Real df = std::exp(-ch/pmy_hydro_->cr_*dt);
  // assuming L = dx
  //   Real df = std::exp(-ch/(pmy_hydro_->cr_*pmy_hydro_->mindx_)*dt);

  // assuming Cartesian
  if (divbsrc_) {
    Real dtoch2 = dt / SQR(ch);
    if (pmb->block_size.nx3 > 1) { // 3D
      for (int k = pmb->ks; k <= pmb->ke; ++k) {
        Real idz = 1.0 / pco->x3f(k);
        for (int j = pmb->js; j <= pmb->je; ++j) {
          Real idy = 1.0 / pco->x2f(j);
#pragma omp simd
          for (int i = pmb->is; i <= pmb->ie; ++i) {
            Real idx = 1.0 / pco->x1f(i);
            Real divBdt = ((x1flux(IPS,k,  j,  i+1) - x1flux(IPS,k,j,i)) * idx
                         + (x2flux(IPS,k,  j+1,i)   - x2flux(IPS,k,j,i)) * idy
                         + (x3flux(IPS,k+1,j,  i)   - x3flux(IPS,k,j,i)) * idz)
                         * dtoch2;
            cons(IM1,k,j,i) -= divBdt*prim(IBX1,k,j,i);
            cons(IM2,k,j,i) -= divBdt*prim(IBX2,k,j,i);
            cons(IM3,k,j,i) -= divBdt*prim(IBX3,k,j,i);
            cons(IBX1,k,j,i) -= divBdt*prim(IVX,k,j,i);
            cons(IBX2,k,j,i) -= divBdt*prim(IVY,k,j,i);
            cons(IBX3,k,j,i) -= divBdt*prim(IVZ,k,j,i);
            cons(IPS,k,j,i) *= df;
            if (NON_BAROTROPIC_EOS)
              cons(IEN,k,j,i) -= divBdt
                              * (prim(IBX1,k,j,i) * prim(IVX,k,j,i)
                               + prim(IBX2,k,j,i) * prim(IVY,k,j,i)
                               + prim(IBX3,k,j,i) * prim(IVZ,k,j,i));
          }
        }
      }
    } else if (pmb->block_size.nx2 > 1) {
      int k = pmb->ks;
      for (int j = pmb->js; j <= pmb->je; ++j) {
        Real idy = 1.0 / pco->x2f(j);
#pragma omp simd
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          Real idx = 1.0 / pco->x1f(i);
          Real divBdt = ((x1flux(IPS,k,  j,  i+1) - x1flux(IPS,k,j,i)) * idx
                       + (x2flux(IPS,k,  j+1,i)   - x2flux(IPS,k,j,i)) * idy) * dtoch2;
          cons(IM1,k,j,i) -= divBdt*prim(IBX1,k,j,i);
          cons(IM2,k,j,i) -= divBdt*prim(IBX2,k,j,i);
          cons(IM3,k,j,i) -= divBdt*prim(IBX3,k,j,i);
          cons(IBX1,k,j,i) -= divBdt*prim(IVX,k,j,i);
          cons(IBX2,k,j,i) -= divBdt*prim(IVY,k,j,i);
          cons(IBX3,k,j,i) -= divBdt*prim(IVZ,k,j,i);
          cons(IPS,k,j,i) *= df;
          if (NON_BAROTROPIC_EOS)
            cons(IEN,k,j,i) -= divBdt
                            * (prim(IBX1,k,j,i) * prim(IVX,k,j,i)
                             + prim(IBX2,k,j,i) * prim(IVY,k,j,i)
                             + prim(IBX3,k,j,i) * prim(IVZ,k,j,i));
        }
      }
    } else { // 1D
      int k = pmb->ks, j = pmb->js;
#pragma omp simd
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        Real idx = 1.0 / pco->x1f(i);
        Real divBdt = ((x1flux(IPS,k,  j,  i+1) - x1flux(IPS,k,j,i)) * idx) * dtoch2;
        cons(IM1,k,j,i) -= divBdt*prim(IBX1,k,j,i);
        cons(IM2,k,j,i) -= divBdt*prim(IBX2,k,j,i);
        cons(IM3,k,j,i) -= divBdt*prim(IBX3,k,j,i);
        cons(IBX1,k,j,i) -= divBdt*prim(IVX,k,j,i);
        cons(IBX2,k,j,i) -= divBdt*prim(IVY,k,j,i);
        cons(IBX3,k,j,i) -= divBdt*prim(IVZ,k,j,i);
        cons(IPS,k,j,i) *= df;
        if (NON_BAROTROPIC_EOS)
          cons(IEN,k,j,i) -= divBdt
                          * (prim(IBX1,k,j,i) * prim(IVX,k,j,i)
                           + prim(IBX2,k,j,i) * prim(IVY,k,j,i)
                           + prim(IBX3,k,j,i) * prim(IVZ,k,j,i));
      }
    }
  } else {
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
#pragma omp simd
        for (int i = pmb->is; i <= pmb->ie; ++i)
          cons(IPS,k,j,i) *= df;
      }
    }
  }
  return;
}
