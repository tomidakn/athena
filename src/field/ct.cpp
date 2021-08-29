//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file ct.cpp
//! \brief

// C headers

// C++ headers
#include <algorithm>  // max(), min()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "field.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn  void Field::CT
//! \brief Constrained Transport implementation of dB/dt = -Curl(E), where E=-(v X B)

void Field::CT(const Real wght, FaceField &b_out) {
  MeshBlock *pmb=pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  AthenaArray<Real> &e1 = e.x1e, &e2 = e.x2e, &e3 = e.x3e;
  AthenaArray<Real> &area = face_area_, &len = edge_length_, &len_p1 = edge_length_p1_;

  //---- update B1
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      // add curl(E) in 2D and 3D problem
      if (pmb->block_size.nx2 > 1) {
        pmb->pcoord->Face1Area(k,j,is,ie+1,area);
        pmb->pcoord->Edge3Length(k,j  ,is,ie+1,len);
        pmb->pcoord->Edge3Length(k,j+1,is,ie+1,len_p1);
#pragma clang loop vectorize(assume_safety)
        for (int i=is; i<=ie+1; ++i) {
          b_out.x1f(k,j,i) -=
              (wght/area(i))*(len_p1(i)*e3(k,j+1,i) - len(i)*e3(k,j,i));
        }

        if (pmb->block_size.nx3 > 1) {
          pmb->pcoord->Edge2Length(k  ,j,is,ie+1,len);
          pmb->pcoord->Edge2Length(k+1,j,is,ie+1,len_p1);
#pragma clang loop vectorize(assume_safety)
          for (int i=is; i<=ie+1; ++i) {
            b_out.x1f(k,j,i) +=
                (wght/area(i))*(len_p1(i)*e2(k+1,j,i) - len(i)*e2(k,j,i));
          }
        }
      }
    }
  }

  //---- update B2 (curl terms in 1D and 3D problems)
  for (int k=ks; k<=ke; ++k) {
    // reset loop limits for polar boundary
    int jl=js; int ju=je+1;
    if (pmb->pbval->block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::polar
        || pmb->pbval->block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::polar_wedge)
      jl=js+1;
    if (pmb->pbval->block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::polar
        || pmb->pbval->block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::polar_wedge)
      ju=je;
    for (int j=jl; j<=ju; ++j) {
      pmb->pcoord->Face2Area(k,j,is,ie,area);
      pmb->pcoord->Edge3Length(k,j,is,ie+1,len);
#pragma clang loop vectorize(assume_safety)
      for (int i=is; i<=ie; ++i) {
        b_out.x2f(k,j,i) += (wght/area(i))*(len(i+1)*e3(k,j,i+1) - len(i)*e3(k,j,i));
      }
      if (pmb->block_size.nx3 > 1) {
        pmb->pcoord->Edge1Length(k  ,j,is,ie,len);
        pmb->pcoord->Edge1Length(k+1,j,is,ie,len_p1);
#pragma clang loop vectorize(assume_safety)
        for (int i=is; i<=ie; ++i) {
          b_out.x2f(k,j,i) -=
              (wght/area(i))*(len_p1(i)*e1(k+1,j,i) - len(i)*e1(k,j,i));
        }
      }
    }
  }

  //---- update B3 (curl terms in 1D and 2D problems)
  for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
      pmb->pcoord->Face3Area(k,j,is,ie,area);
      pmb->pcoord->Edge2Length(k,j,is,ie+1,len);
#pragma clang loop vectorize(assume_safety)
      for (int i=is; i<=ie; ++i) {
        b_out.x3f(k,j,i) -= (wght/area(i))*(len(i+1)*e2(k,j,i+1) - len(i)*e2(k,j,i));
      }
      if (pmb->block_size.nx2 > 1) {
        pmb->pcoord->Edge1Length(k,j  ,is,ie,len);
        pmb->pcoord->Edge1Length(k,j+1,is,ie,len_p1);
#pragma clang loop vectorize(assume_safety)
        for (int i=is; i<=ie; ++i) {
          b_out.x3f(k,j,i) +=
              (wght/area(i))*(len_p1(i)*e1(k,j+1,i) - len(i)*e1(k,j,i));
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Field::CT_STS
//! \brief Constrained Transport implementation of dB/dt = -Curl(E), where E=-(v X B)
//! for STS. RKL2 registers are set to -Curl(E) update if first stage of RKL2 STS.

void Field::CT_STS(const Real wght, int stage,
                   FaceField &b_out, FaceField &ct_update_out) {
  MeshBlock *pmb=pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  AthenaArray<Real> &e1 = e.x1e, &e2 = e.x2e, &e3 = e.x3e;
  AthenaArray<Real> &area = face_area_, &len = edge_length_, &len_p1 = edge_length_p1_;

  if (stage == 1 && pmb->pmy_mesh->sts_integrator == "rkl2") {
  //---- update B1
    if (pmb->block_size.nx2 > 1) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
        // add curl(E) in 2D and 3D problem
          pmb->pcoord->Face1Area(k,j,is,ie+1,area);
          pmb->pcoord->Edge3Length(k,j  ,is,ie+1,len);
          pmb->pcoord->Edge3Length(k,j+1,is,ie+1,len_p1);
#pragma clang loop vectorize(assume_safety)
          for (int i=is; i<=ie+1; ++i) {
            Real db13 = (len_p1(i)*e3(k,j+1,i) - len(i)*e3(k,j,i))/area(i);
            b_out.x1f(k,j,i) -= wght*db13;
            ct_update_out.x1f(k,j,i) = -0.5*pmb->pmy_mesh->dt*db13;
          }
          if (pmb->block_size.nx3 > 1) {
            pmb->pcoord->Edge2Length(k  ,j,is,ie+1,len);
            pmb->pcoord->Edge2Length(k+1,j,is,ie+1,len_p1);
#pragma clang loop vectorize(assume_safety)
            for (int i=is; i<=ie+1; ++i) {
              Real db12 = (len_p1(i)*e2(k+1,j,i) - len(i)*e2(k,j,i))/area(i);
              b_out.x1f(k,j,i) += wght*db12;
              ct_update_out.x1f(k,j,i) += 0.5*pmb->pmy_mesh->dt*db12;
            }
          }
        }
      }
    }

    //---- update B2 (curl terms in 1D and 3D problems)
    for (int k=ks; k<=ke; ++k) {
      // reset loop limits for polar boundary
      int jl=js; int ju=je+1;
      if (pmb->pbval->block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::polar
          || pmb->pbval->block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::polar_wedge)
        jl=js+1;
      if (pmb->pbval->block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::polar
          || pmb->pbval->block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::polar_wedge)
        ju=je;
      for (int j=jl; j<=ju; ++j) {
        pmb->pcoord->Face2Area(k,j,is,ie,area);
        pmb->pcoord->Edge3Length(k,j,is,ie+1,len);
#pragma clang loop vectorize(assume_safety)
        for (int i=is; i<=ie; ++i) {
          Real db23 = (len(i+1)*e3(k,j,i+1) - len(i)*e3(k,j,i))/area(i);
          b_out.x2f(k,j,i) += wght*db23;
          ct_update_out.x2f(k,j,i) = 0.5*pmb->pmy_mesh->dt*db23;
        }
        if (pmb->block_size.nx3 > 1) {
          pmb->pcoord->Edge1Length(k  ,j,is,ie,len);
          pmb->pcoord->Edge1Length(k+1,j,is,ie,len_p1);
#pragma clang loop vectorize(assume_safety)
          for (int i=is; i<=ie; ++i) {
            Real db21 = (len_p1(i)*e1(k+1,j,i) - len(i)*e1(k,j,i))/area(i);
            b_out.x2f(k,j,i) -= wght*db21;
            ct_update_out.x2f(k,j,i) -= 0.5*pmb->pmy_mesh->dt*db21;
          }
        }
      }
    }

    //---- update B3 (curl terms in 1D and 2D problems)
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        pmb->pcoord->Face3Area(k,j,is,ie,area);
        pmb->pcoord->Edge2Length(k,j,is,ie+1,len);
#pragma clang loop vectorize(assume_safety)
        for (int i=is; i<=ie; ++i) {
          Real db32 = (len(i+1)*e2(k,j,i+1) - len(i)*e2(k,j,i))/area(i);
          b_out.x3f(k,j,i) -= wght*db32;
          ct_update_out.x3f(k,j,i) = -0.5*pmb->pmy_mesh->dt*db32;
        }
        if (pmb->block_size.nx2 > 1) {
          pmb->pcoord->Edge1Length(k,j  ,is,ie,len);
          pmb->pcoord->Edge1Length(k,j+1,is,ie,len_p1);
#pragma clang loop vectorize(assume_safety)
          for (int i=is; i<=ie; ++i) {
            Real db31 = (len_p1(i)*e1(k,j+1,i) - len(i)*e1(k,j,i))/area(i);
            b_out.x3f(k,j,i) += wght*db31;
            ct_update_out.x3f(k,j,i) += 0.5*pmb->pmy_mesh->dt*db31;
          }
        }
      }
    }
  } else {
  //---- update B1
    if (pmb->block_size.nx2 > 1) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
        // add curl(E) in 2D and 3D problem
          pmb->pcoord->Face1Area(k,j,is,ie+1,area);
          pmb->pcoord->Edge3Length(k,j  ,is,ie+1,len);
          pmb->pcoord->Edge3Length(k,j+1,is,ie+1,len_p1);
#pragma clang loop vectorize(assume_safety)
          for (int i=is; i<=ie+1; ++i) {
            b_out.x1f(k,j,i) -=
                (wght/area(i))*(len_p1(i)*e3(k,j+1,i) - len(i)*e3(k,j,i));
          }

          if (pmb->block_size.nx3 > 1) {
            pmb->pcoord->Edge2Length(k  ,j,is,ie+1,len);
            pmb->pcoord->Edge2Length(k+1,j,is,ie+1,len_p1);
#pragma clang loop vectorize(assume_safety)
            for (int i=is; i<=ie+1; ++i) {
              b_out.x1f(k,j,i) +=
                (wght/area(i))*(len_p1(i)*e2(k+1,j,i) - len(i)*e2(k,j,i));
            }
          }
        }
      }
    }

    //---- update B2 (curl terms in 1D and 3D problems)
    for (int k=ks; k<=ke; ++k) {
      // reset loop limits for polar boundary
      int jl=js; int ju=je+1;
      if (pmb->pbval->block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::polar
          || pmb->pbval->block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::polar_wedge)
        jl=js+1;
      if (pmb->pbval->block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::polar
          || pmb->pbval->block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::polar_wedge)
        ju=je;
      for (int j=jl; j<=ju; ++j) {
        pmb->pcoord->Face2Area(k,j,is,ie,area);
        pmb->pcoord->Edge3Length(k,j,is,ie+1,len);
#pragma clang loop vectorize(assume_safety)
        for (int i=is; i<=ie; ++i) {
          b_out.x2f(k,j,i) += (wght/area(i))*(len(i+1)*e3(k,j,i+1) - len(i)*e3(k,j,i));
        }
        if (pmb->block_size.nx3 > 1) {
          pmb->pcoord->Edge1Length(k  ,j,is,ie,len);
          pmb->pcoord->Edge1Length(k+1,j,is,ie,len_p1);
#pragma clang loop vectorize(assume_safety)
          for (int i=is; i<=ie; ++i) {
            b_out.x2f(k,j,i) -=
              (wght/area(i))*(len_p1(i)*e1(k+1,j,i) - len(i)*e1(k,j,i));
          }
        }
      }
    }

    //---- update B3 (curl terms in 1D and 2D problems)
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        pmb->pcoord->Face3Area(k,j,is,ie,area);
        pmb->pcoord->Edge2Length(k,j,is,ie+1,len);
#pragma clang loop vectorize(assume_safety)
        for (int i=is; i<=ie; ++i) {
          b_out.x3f(k,j,i) -= (wght/area(i))*(len(i+1)*e2(k,j,i+1) - len(i)*e2(k,j,i));
        }
        if (pmb->block_size.nx2 > 1) {
          pmb->pcoord->Edge1Length(k,j  ,is,ie,len);
          pmb->pcoord->Edge1Length(k,j+1,is,ie,len_p1);
#pragma clang loop vectorize(assume_safety)
          for (int i=is; i<=ie; ++i) {
            b_out.x3f(k,j,i) +=
                (wght/area(i))*(len_p1(i)*e1(k,j+1,i) - len(i)*e1(k,j,i));
          }
        }
      }
    }
  }
  return;
}
