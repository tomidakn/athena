//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file self_gravity.cpp
//! \brief source terms due to self-gravity

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
//! \fn void HydroSourceTerms::SelfGravity
//! \brief Adds source terms for self-gravitational acceleration to conserved variables
//! \note
//! This implements the source term formula in Mullen, Hanawa and Gammie 2020, but only
//! for the momentum part. The energy source term is not conservative in this version.
//! I leave the fully conservative formula for later as it requires design consideration.
//! Also note that this implementation is not exactly conservative when the potential
//! contains a residual error (Multigrid has small but non-zero residual).

void HydroSourceTerms::SelfGravity(const Real dt,const AthenaArray<Real> *flux,
                                   const AthenaArray<Real> &prim,
                                   AthenaArray<Real> &cons) {
  MeshBlock *pmb = pmy_hydro_->pmy_block;
  Gravity *pgrav = pmb->pgrav;

  // assume 3D

  // acceleration in 1-direction
  if (NON_BAROTROPIC_EOS) {
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma clang loop vectorize(assume_safety)
#pragma fj loop loop_fission_target
#pragma fj loop loop_fission_threshold 1
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real den = prim(IDN,k,j,i);
          Real dx1 = pmb->pcoord->dx1v(i);
          Real dtodx1 = dt/dx1;
          Real phic = pgrav->phi(k,j,i);
          Real phil1 = 0.5*(pgrav->phi(k,j,i-1)+phic);
          Real phir1 = 0.5*(phic+pgrav->phi(k,j,i+1));
          cons(IM1,k,j,i) -= dtodx1*den*(phir1-phil1);
          cons(IEN,k,j,i) -= dtodx1*(flux[X1DIR](IDN,k,j,i  )*(phic - phil1) +
                                     flux[X1DIR](IDN,k,j,i+1)*(phir1 - phic));
          Real dx2 = pmb->pcoord->dx2v(j);
          Real dtodx2 = dt/dx2;
          Real phil2 = 0.5*(pgrav->phi(k,j-1,i)+phic);
          Real phir2 = 0.5*(phic+pgrav->phi(k,j+1,i));
          cons(IM2,k,j,i) -= dtodx2*den*(phir2-phil2);
          cons(IEN,k,j,i) -= dtodx2*(flux[X2DIR](IDN,k,j  ,i)*(phic - phil2) +
                                     flux[X2DIR](IDN,k,j+1,i)*(phir2 - phic));
          Real dx3 = pmb->pcoord->dx3v(k);
          Real dtodx3 = dt/dx3;
          Real phil3 = 0.5*(pgrav->phi(k-1,j,i)+phic);
          Real phir3 = 0.5*(phic+pgrav->phi(k+1,j,i));
          cons(IM3,k,j,i) -= dtodx3*den*(phir3-phil3);
          cons(IEN,k,j,i) -= dtodx3*(flux[X3DIR](IDN,k  ,j,i)*(phic - phil3) +
                                     flux[X3DIR](IDN,k+1,j,i)*(phir3 - phic));
        }
      }
    }
  } else {
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma clang loop vectorize(assume_safety)
#pragma fj loop loop_fission_target
#pragma fj loop loop_fission_threshold 1
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real den = prim(IDN,k,j,i);
          Real dx1 = pmb->pcoord->dx1v(i);
          Real dtodx1 = dt/dx1;
          Real phic = pgrav->phi(k,j,i);
          Real phil1 = 0.5*(pgrav->phi(k,j,i-1)+phic);
          Real phir1 = 0.5*(phic+pgrav->phi(k,j,i+1));
          cons(IM1,k,j,i) -= dtodx1*den*(phir1-phil1);

          Real dx2 = pmb->pcoord->dx2v(j);
          Real dtodx2 = dt/dx2;
          Real phil2 = 0.5*(pgrav->phi(k,j-1,i)+phic);
          Real phir2 = 0.5*(phic+pgrav->phi(k,j+1,i));
          cons(IM2,k,j,i) -= dtodx2*den*(phir2-phil2);

          Real dx3 = pmb->pcoord->dx3v(k);
          Real dtodx3 = dt/dx3;
          Real phil3 = 0.5*(pgrav->phi(k-1,j,i)+phic);
          Real phir3 = 0.5*(phic+pgrav->phi(k+1,j,i));
          cons(IM3,k,j,i) -= dtodx3*den*(phir3-phil3);
        }
      }
    }
  }

  return;
}
