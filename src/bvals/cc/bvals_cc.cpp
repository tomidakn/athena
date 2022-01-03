//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_cc.cpp
//! \brief functions that apply BCs for CELL_CENTERED variables

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>    // memcpy()
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../eos/eos.hpp"
#include "../../field/field.hpp"
#include "../../globals.hpp"
#include "../../hydro/hydro.hpp"
#include "../../mesh/mesh.hpp"
#include "../../parameter_input.hpp"
#include "../../utils/buffer_utils.hpp"
#include "../bvals.hpp"
#include "bvals_cc.hpp"

// MPI header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifdef UTOFU_PARALLEL
#include <utofu.h>
#endif

//! constructor

CellCenteredBoundaryVariable::CellCenteredBoundaryVariable(
    MeshBlock *pmb, AthenaArray<Real> *var, AthenaArray<Real> *coarse_var,
    AthenaArray<Real> *var_flux, bool fflux)
    : BoundaryVariable(pmb, fflux), var_cc(var), coarse_buf(coarse_var),
      x1flux(var_flux[X1DIR]), x2flux(var_flux[X2DIR]), x3flux(var_flux[X3DIR]),
      nl_(0), nu_(var->GetDim4() -1), flip_across_pole_(nullptr) {
  //! \note
  //! CellCenteredBoundaryVariable should only be used w/ 4D or 3D (nx4=1) AthenaArray
  //! For now, assume that full span of 4th dim of input AthenaArray should be used:
  //! ---> get the index limits directly from the input AthenaArray
  //! <=nu_ (inclusive), <nx4 (exclusive)
  if (nu_ < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in CellCenteredBoundaryVariable constructor" << std::endl
        << "An 'AthenaArray<Real> *var' of nx4_ = " << var->GetDim4() << " was passed\n"
        << "Should be nx4 >= 1 (likely uninitialized)." << std::endl;
    ATHENA_ERROR(msg);
  }

  // KT: fflux is a flag and it is true (false) when flux correction is (not) needed.
  //     I have not implemented it for shearing box, leaving it to Tomohiro.


  InitBoundaryData(bd_var_, BoundaryQuantity::cc);
#ifdef MPI_PARALLEL
  // KGF: dead code, leaving for now:
  // cc_phys_id_ = pbval_->ReserveTagVariableIDs(1);
  cc_phys_id_ = pbval_->bvars_next_phys_id_;
#endif
  if (fflux_ && ((pmy_mesh_->multilevel)
      || (pbval_->shearing_box != 0))) { // SMR or AMR or SHEARING_BOX
    fflux_ = true;
    InitBoundaryData(bd_var_flcor_, BoundaryQuantity::cc_flcor);
#ifdef MPI_PARALLEL
    cc_flx_phys_id_ = cc_phys_id_ + 1;
#endif
  } else {
    fflux_ = false;
  }

  if (pbval_->shearing_box != 0) {
#ifdef MPI_PARALLEL
    shear_cc_phys_id_ = cc_phys_id_ + 2;
    shear_flx_phys_id_ = shear_cc_phys_id_ + 1;
#endif
    int nc2 = pmb->ncells2;
    int nc3 = pmb->ncells3;
    int nx3 = pmb->block_size.nx3;
    int &xgh = pbval_->xgh_;
    for (int upper=0; upper<2; upper++) {
      if (pbval_->is_shear[upper]) {
        shear_cc_[upper].NewAthenaArray(nu_+1, nc3, NGHOST, nc2+2*xgh+1);
        shear_var_flx_[upper].NewAthenaArray(nu_+1, nc3, nc2);
        shear_map_flx_[upper].NewAthenaArray(nu_+1, nc3, 1, nc2+2*xgh+1);

        // TODO(KGF): the rest of this should be a part of InitBoundaryData()

        int bsize = pmb->block_size.nx2*pbval_->ssize_*(nu_ + 1);
        int fsize = pmb->block_size.nx2*nx3*(nu_ + 1);
        for (int n=0; n<4; n++) {
          shear_bd_var_[upper].send[n] = new Real[bsize];
          shear_bd_var_[upper].recv[n] = new Real[bsize];
          shear_bd_var_[upper].flag[n] = BoundaryStatus::waiting;
#ifdef MPI_PARALLEL
          shear_bd_var_[upper].req_send[n] = MPI_REQUEST_NULL;
          shear_bd_var_[upper].req_recv[n] = MPI_REQUEST_NULL;
#endif
        }
        for (int n=0; n<3; n++) {
          shear_bd_flux_[upper].send[n] = new Real[fsize];
          shear_bd_flux_[upper].recv[n] = new Real[fsize];
          shear_bd_flux_[upper].flag[n] = BoundaryStatus::waiting;
#ifdef MPI_PARALLEL
          shear_bd_flux_[upper].req_send[n] = MPI_REQUEST_NULL;
          shear_bd_flux_[upper].req_recv[n] = MPI_REQUEST_NULL;
#endif
        }
      } // end "if is a shearing boundary"
    }  // end loop over inner, outer shearing boundaries
  } // end shearing box component
}

//! destructor

CellCenteredBoundaryVariable::~CellCenteredBoundaryVariable() {
  DestroyBoundaryData(bd_var_);
  if (fflux_ && ((pmy_mesh_->multilevel)
      || (pbval_->shearing_box != 0)))
    DestroyBoundaryData(bd_var_flcor_);

  // TODO(KGF): this should be a part of DestroyBoundaryData()
  if (pbval_->shearing_box == 1) {
    for (int upper=0; upper<2; upper++) {
      if (pbval_->is_shear[upper]) { // if true for shearing inner blocks
        for (int n=0; n<4; n++) {
          delete[] shear_bd_var_[upper].send[n];
          delete[] shear_bd_var_[upper].recv[n];
        }
        for (int n=0; n<3; n++) {
          delete[] shear_bd_flux_[upper].send[n];
          delete[] shear_bd_flux_[upper].recv[n];
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn int CellCenteredBoundaryVariable::ComputeVariableBufferSize(
//!     const NeighborIndexes& ni, int cng)
//! \brief

int CellCenteredBoundaryVariable::ComputeVariableBufferSize(const NeighborIndexes& ni,
                                                            int cng) {
  MeshBlock *pmb = pmy_block_;
  int cng1, cng2, cng3;
  cng1 = cng;
  cng2 = cng*(pmb->block_size.nx2 > 1 ? 1 : 0);
  cng3 = cng*(pmb->block_size.nx3 > 1 ? 1 : 0);

  int size = ((ni.ox1 == 0) ? pmb->block_size.nx1 : NGHOST)
           *((ni.ox2 == 0) ? pmb->block_size.nx2 : NGHOST)
           *((ni.ox3 == 0) ? pmb->block_size.nx3 : NGHOST);
  if (pmy_mesh_->multilevel) {
    int f2c = ((ni.ox1 == 0) ? ((pmb->block_size.nx1+1)/2) : NGHOST)
            *((ni.ox2 == 0) ? ((pmb->block_size.nx2+1)/2) : NGHOST)
            *((ni.ox3 == 0) ? ((pmb->block_size.nx3+1)/2) : NGHOST);
    int c2f = ((ni.ox1 == 0) ?((pmb->block_size.nx1+1)/2 + cng1) : cng)
            *((ni.ox2 == 0) ? ((pmb->block_size.nx2+1)/2 + cng2) : cng)
            *((ni.ox3 == 0) ? ((pmb->block_size.nx3+1)/2 + cng3) : cng);
    size = std::max(size, c2f);
    size = std::max(size, f2c);
  }
  size *= nu_ + 1;
  return size;
}

int CellCenteredBoundaryVariable::ComputeFluxCorrectionBufferSize(
    const NeighborIndexes& ni, int cng) {
  MeshBlock *pmb = pmy_block_;
  int size1(0), size2(0);
  if ((pbval_->shearing_box == 1) && (ni.ox1 != 0)) {
    size1 = pmb->block_size.nx2*pmb->block_size.nx3*(nu_ + 1);
  }
  if (pmy_mesh_->multilevel) {
    if (ni.ox1 != 0)
      size2 = (pmb->block_size.nx2 + 1)/2*(pmb->block_size.nx3 + 1)/2*(nu_ + 1);
    if (ni.ox2 != 0)
      size2 = (pmb->block_size.nx1 + 1)/2*(pmb->block_size.nx3 + 1)/2*(nu_ + 1);
    if (ni.ox3 != 0)
      size2 = (pmb->block_size.nx1 + 1)/2*(pmb->block_size.nx2 + 1)/2*(nu_ + 1);
  }
  return std::max(size1,size2);
}

//----------------------------------------------------------------------------------------
//! \fn int CellCenteredBoundaryVariable::LoadBoundaryBufferSameLevel(Real *buf,
//!                                                             const NeighborBlock& nb)
//! \brief Set cell-centered boundary buffers for sending to a block on the same level

int CellCenteredBoundaryVariable::LoadBoundaryBufferSameLevel(Real *buf,
                                                              const NeighborBlock& nb) {
  MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;

  si = (nb.ni.ox1 > 0) ? (pmb->ie - NGHOST + 1) : pmb->is;
  ei = (nb.ni.ox1 < 0) ? (pmb->is + NGHOST - 1) : pmb->ie;
  sj = (nb.ni.ox2 > 0) ? (pmb->je - NGHOST + 1) : pmb->js;
  ej = (nb.ni.ox2 < 0) ? (pmb->js + NGHOST - 1) : pmb->je;
  sk = (nb.ni.ox3 > 0) ? (pmb->ke - NGHOST + 1) : pmb->ks;
  ek = (nb.ni.ox3 < 0) ? (pmb->ks + NGHOST - 1) : pmb->ke;
  int p = 0;
  AthenaArray<Real> &var = *var_cc;
  BufferUtility::PackData(var, buf, nl_, nu_, si, ei, sj, ej, sk, ek, p);
  return p;
}

//----------------------------------------------------------------------------------------
//! \fn int CellCenteredBoundaryVariable::LoadBoundaryBufferToCoarser(Real *buf,
//!                                                             const NeighborBlock& nb)
//! \brief Set cell-centered boundary buffers for sending to a block on the coarser level

int CellCenteredBoundaryVariable::LoadBoundaryBufferToCoarser(Real *buf,
                                                              const NeighborBlock& nb) {
  MeshBlock *pmb = pmy_block_;
  MeshRefinement *pmr = pmb->pmr;
  int si, sj, sk, ei, ej, ek;
  int cn = NGHOST - 1;
  AthenaArray<Real> &var = *var_cc;
  AthenaArray<Real> &coarse_var = *coarse_buf;

  si = (nb.ni.ox1 > 0) ? (pmb->cie - cn) : pmb->cis;
  ei = (nb.ni.ox1 < 0) ? (pmb->cis + cn) : pmb->cie;
  sj = (nb.ni.ox2 > 0) ? (pmb->cje - cn) : pmb->cjs;
  ej = (nb.ni.ox2 < 0) ? (pmb->cjs + cn) : pmb->cje;
  sk = (nb.ni.ox3 > 0) ? (pmb->cke - cn) : pmb->cks;
  ek = (nb.ni.ox3 < 0) ? (pmb->cks + cn) : pmb->cke;

  int p = 0;
  pmr->RestrictCellCenteredValues(var, coarse_var, nl_, nu_, si, ei, sj, ej, sk, ek);
  BufferUtility::PackData(coarse_var, buf, nl_, nu_, si, ei, sj, ej, sk, ek, p);
  return p;
}

//----------------------------------------------------------------------------------------
//! \fn int CellCenteredBoundaryVariable::LoadBoundaryBufferToFiner(Real *buf,
//!                                                             const NeighborBlock& nb)
//! \brief Set cell-centered boundary buffers for sending to a block on the finer level

int CellCenteredBoundaryVariable::LoadBoundaryBufferToFiner(Real *buf,
                                                            const NeighborBlock& nb) {
  MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;
  int cn = pmb->cnghost - 1;
  AthenaArray<Real> &var = *var_cc;

  si = (nb.ni.ox1 > 0) ? (pmb->ie - cn) : pmb->is;
  ei = (nb.ni.ox1 < 0) ? (pmb->is + cn) : pmb->ie;
  sj = (nb.ni.ox2 > 0) ? (pmb->je - cn) : pmb->js;
  ej = (nb.ni.ox2 < 0) ? (pmb->js + cn) : pmb->je;
  sk = (nb.ni.ox3 > 0) ? (pmb->ke - cn) : pmb->ks;
  ek = (nb.ni.ox3 < 0) ? (pmb->ks + cn) : pmb->ke;

  // send the data first and later prolongate on the target block
  // need to add edges for faces, add corners for edges
  if (nb.ni.ox1 == 0) {
    if (nb.ni.fi1 == 1)   si += pmb->block_size.nx1/2 - pmb->cnghost;
    else            ei -= pmb->block_size.nx1/2 - pmb->cnghost;
  }
  if (nb.ni.ox2 == 0 && pmb->block_size.nx2 > 1) {
    if (nb.ni.ox1 != 0) {
      if (nb.ni.fi1 == 1) sj += pmb->block_size.nx2/2 - pmb->cnghost;
      else          ej -= pmb->block_size.nx2/2 - pmb->cnghost;
    } else {
      if (nb.ni.fi2 == 1) sj += pmb->block_size.nx2/2 - pmb->cnghost;
      else          ej -= pmb->block_size.nx2/2 - pmb->cnghost;
    }
  }
  if (nb.ni.ox3 == 0 && pmb->block_size.nx3 > 1) {
    if (nb.ni.ox1 != 0 && nb.ni.ox2 != 0) {
      if (nb.ni.fi1 == 1) sk += pmb->block_size.nx3/2 - pmb->cnghost;
      else          ek -= pmb->block_size.nx3/2 - pmb->cnghost;
    } else {
      if (nb.ni.fi2 == 1) sk += pmb->block_size.nx3/2 - pmb->cnghost;
      else          ek -= pmb->block_size.nx3/2 - pmb->cnghost;
    }
  }

  int p = 0;
  BufferUtility::PackData(var, buf, nl_, nu_, si, ei, sj, ej, sk, ek, p);
  return p;
}


//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::SetBoundarySameLevel(Real *buf,
//!                                                             const NeighborBlock& nb)
//! \brief Set cell-centered boundary received from a block on the same level

void CellCenteredBoundaryVariable::SetBoundarySameLevel(Real *buf,
                                                        const NeighborBlock& nb) {
  MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;
  AthenaArray<Real> &var = *var_cc;

  if (nb.ni.ox1 == 0)     si = pmb->is,        ei = pmb->ie;
  else if (nb.ni.ox1 > 0) si = pmb->ie + 1,      ei = pmb->ie + NGHOST;
  else              si = pmb->is - NGHOST, ei = pmb->is - 1;
  if (nb.ni.ox2 == 0)     sj = pmb->js,        ej = pmb->je;
  else if (nb.ni.ox2 > 0) sj = pmb->je + 1,      ej = pmb->je + NGHOST;
  else              sj = pmb->js - NGHOST, ej = pmb->js - 1;
  if (nb.ni.ox3 == 0)     sk = pmb->ks,        ek = pmb->ke;
  else if (nb.ni.ox3 > 0) sk = pmb->ke + 1,      ek = pmb->ke + NGHOST;
  else              sk = pmb->ks - NGHOST, ek = pmb->ks - 1;

  int p = 0;

  if (nb.polar) {
    for (int n=nl_; n<=nu_; ++n) {
      Real sign = 1.0;
      if (flip_across_pole_ != nullptr) sign = flip_across_pole_[n] ? -1.0 : 1.0;
      for (int k=sk; k<=ek; ++k) {
        for (int j=ej; j>=sj; --j) {
#pragma omp simd linear(p)
          for (int i=si; i<=ei; ++i) {
            var(n,k,j,i) = sign * buf[p++];
          }
        }
      }
    }
  } else {
    BufferUtility::UnpackData(buf, var, nl_, nu_, si, ei, sj, ej, sk, ek, p);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::SetBoundaryFromCoarser(Real *buf,
//!                                                               const NeighborBlock& nb)
//! \brief Set cell-centered prolongation buffer received from a block on a coarser level

void CellCenteredBoundaryVariable::SetBoundaryFromCoarser(Real *buf,
                                                          const NeighborBlock& nb) {
  MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;
  int cng = pmb->cnghost;
  AthenaArray<Real> &coarse_var = *coarse_buf;

  if (nb.ni.ox1 == 0) {
    si = pmb->cis, ei = pmb->cie;
    if ((pmb->loc.lx1 & 1LL) == 0LL) ei += cng;
    else                             si -= cng;
  } else if (nb.ni.ox1 > 0)  {
    si = pmb->cie + 1,   ei = pmb->cie + cng;
  } else {
    si = pmb->cis - cng, ei = pmb->cis - 1;
  }
  if (nb.ni.ox2 == 0) {
    sj = pmb->cjs, ej = pmb->cje;
    if (pmb->block_size.nx2 > 1) {
      if ((pmb->loc.lx2 & 1LL) == 0LL) ej += cng;
      else                             sj -= cng;
    }
  } else if (nb.ni.ox2 > 0) {
    sj = pmb->cje + 1,   ej = pmb->cje + cng;
  } else {
    sj = pmb->cjs - cng, ej = pmb->cjs - 1;
  }
  if (nb.ni.ox3 == 0) {
    sk = pmb->cks, ek = pmb->cke;
    if (pmb->block_size.nx3 > 1) {
      if ((pmb->loc.lx3 & 1LL) == 0LL) ek += cng;
      else                             sk -= cng;
    }
  } else if (nb.ni.ox3 > 0)  {
    sk = pmb->cke + 1,   ek = pmb->cke + cng;
  } else {
    sk = pmb->cks - cng, ek = pmb->cks - 1;
  }

  int p = 0;
  if (nb.polar) {
    for (int n=nl_; n<=nu_; ++n) {
      Real sign = 1.0;
      if (flip_across_pole_ != nullptr) sign = flip_across_pole_[n] ? -1.0 : 1.0;
      for (int k=sk; k<=ek; ++k) {
        for (int j=ej; j>=sj; --j) {
#pragma omp simd linear(p)
          for (int i=si; i<=ei; ++i)
            coarse_var(n,k,j,i) = sign * buf[p++];
        }
      }
    }
  } else {
    BufferUtility::UnpackData(buf, coarse_var, nl_, nu_, si, ei, sj, ej, sk, ek, p);
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::SetBoundaryFromFiner(Real *buf,
//!                                                             const NeighborBlock& nb)
//! \brief Set cell-centered boundary received from a block on a finer level

void CellCenteredBoundaryVariable::SetBoundaryFromFiner(Real *buf,
                                                        const NeighborBlock& nb) {
  MeshBlock *pmb = pmy_block_;
  AthenaArray<Real> &var = *var_cc;
  // receive already restricted data
  int si, sj, sk, ei, ej, ek;

  if (nb.ni.ox1 == 0) {
    si = pmb->is, ei = pmb->ie;
    if (nb.ni.fi1 == 1)   si += pmb->block_size.nx1/2;
    else            ei -= pmb->block_size.nx1/2;
  } else if (nb.ni.ox1 > 0) {
    si = pmb->ie + 1,      ei = pmb->ie + NGHOST;
  } else {
    si = pmb->is - NGHOST, ei = pmb->is - 1;
  }
  if (nb.ni.ox2 == 0) {
    sj = pmb->js, ej = pmb->je;
    if (pmb->block_size.nx2 > 1) {
      if (nb.ni.ox1 != 0) {
        if (nb.ni.fi1 == 1) sj += pmb->block_size.nx2/2;
        else          ej -= pmb->block_size.nx2/2;
      } else {
        if (nb.ni.fi2 == 1) sj += pmb->block_size.nx2/2;
        else          ej -= pmb->block_size.nx2/2;
      }
    }
  } else if (nb.ni.ox2 > 0) {
    sj = pmb->je + 1,      ej = pmb->je + NGHOST;
  } else {
    sj = pmb->js - NGHOST, ej = pmb->js - 1;
  }
  if (nb.ni.ox3 == 0) {
    sk = pmb->ks, ek = pmb->ke;
    if (pmb->block_size.nx3 > 1) {
      if (nb.ni.ox1 != 0 && nb.ni.ox2 != 0) {
        if (nb.ni.fi1 == 1) sk += pmb->block_size.nx3/2;
        else          ek -= pmb->block_size.nx3/2;
      } else {
        if (nb.ni.fi2 == 1) sk += pmb->block_size.nx3/2;
        else          ek -= pmb->block_size.nx3/2;
      }
    }
  } else if (nb.ni.ox3 > 0) {
    sk = pmb->ke + 1,      ek = pmb->ke + NGHOST;
  } else {
    sk = pmb->ks - NGHOST, ek = pmb->ks - 1;
  }

  int p = 0;
  if (nb.polar) {
    for (int n=nl_; n<=nu_; ++n) {
      Real sign=1.0;
      if (flip_across_pole_ != nullptr) sign = flip_across_pole_[n] ? -1.0 : 1.0;
      for (int k=sk; k<=ek; ++k) {
        for (int j=ej; j>=sj; --j) {
#pragma omp simd linear(p)
          for (int i=si; i<=ei; ++i)
            var(n,k,j,i) = sign * buf[p++];
        }
      }
    }
  } else {
    BufferUtility::UnpackData(buf, var, nl_, nu_, si, ei, sj, ej, sk, ek, p);
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::PolarBoundarySingleAzimuthalBlock()
//! \brief polar boundary edge-case:
//!   single MeshBlock spans the entire azimuthal (x3) range

void CellCenteredBoundaryVariable::PolarBoundarySingleAzimuthalBlock() {
  MeshBlock *pmb = pmy_block_;

  if (pmb->loc.level  ==  pmy_mesh_->root_level && pmy_mesh_->nrbx3 == 1
      && pmb->block_size.nx3 > 1) {
    AthenaArray<Real> &var = *var_cc;
    if (pbval_->block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::polar) {
      int nx3_half = (pmb->ke - pmb->ks + 1) / 2;
      for (int n=nl_; n<=nu_; ++n) {
        for (int j=pmb->js-NGHOST; j<=pmb->js-1; ++j) {
          for (int i=pmb->is-NGHOST; i<=pmb->ie+NGHOST; ++i) {
            for (int k=pmb->ks-NGHOST; k<=pmb->ke+NGHOST; ++k)
              pbval_->azimuthal_shift_(k) = var(n,k,j,i);
            for (int k=pmb->ks-NGHOST; k<=pmb->ke+NGHOST; ++k) {
              int k_shift = k;
              k_shift += (k < (nx3_half + NGHOST) ? 1 : -1) * nx3_half;
              var(n,k,j,i) = pbval_->azimuthal_shift_(k_shift);
            }
          }
        }
      }
    }

    if (pbval_->block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::polar) {
      int nx3_half = (pmb->ke - pmb->ks + 1) / 2;
      for (int n=nl_; n<=nu_; ++n) {
        for (int j=pmb->je+1; j<=pmb->je+NGHOST; ++j) {
          for (int i=pmb->is-NGHOST; i<=pmb->ie+NGHOST; ++i) {
            for (int k=pmb->ks-NGHOST; k<=pmb->ke+NGHOST; ++k)
              pbval_->azimuthal_shift_(k) = var(n,k,j,i);
            for (int k=pmb->ks-NGHOST; k<=pmb->ke+NGHOST; ++k) {
              int k_shift = k;
              k_shift += (k < (nx3_half + NGHOST) ? 1 : -1) * nx3_half;
              var(n,k,j,i) = pbval_->azimuthal_shift_(k_shift);
            }
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::SetupPersistentMPI()
//! \brief Setup persistent MPI requests to be reused throughout the entire simulation

void CellCenteredBoundaryVariable::SetupPersistentMPI() {
#ifdef MPI_PARALLEL
  MeshBlock* pmb = pmy_block_;
  int &mylevel = pmb->loc.level;
  int tag;

#ifdef UTOFU_PARALLEL
  const unsigned long int uflags = UTOFU_ONESIDED_FLAG_TCQ_NOTICE
                                 | UTOFU_ONESIDED_FLAG_STRONG_ORDER
                                 | UTOFU_ONESIDED_FLAG_CACHE_INJECTION;
  // exchange VCQ and STADD - assuming one MeshBlock per process
  std::uint64_t utrbuf[bd_var_.nbmax][2], utsbuf[bd_var_.nbmax][2];
  MPI_Request utrreq[bd_var_.nbmax], utsreq[bd_var_.nbmax];
#pragma omp critical(utofu_init)
  for (int n=0; n<pbval_->nneighbor; n++) {
    NeighborBlock& nb = pbval_->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      tag = pbval_->CreateBvalsMPITag(pmb->lid, nb.bufid, cc_phys_id_);
      MPI_Irecv(utrbuf[nb.bufid], 2, MPI_UINT64_T, nb.snb.rank, tag,
                MPI_COMM_WORLD, &(utrreq[nb.bufid]));
      utsbuf[nb.bufid][0] = bd_var_.vcq_id;
      utsbuf[nb.bufid][1] = bd_var_.lra[nb.bufid];
      tag = pbval_->CreateBvalsMPITag(nb.snb.lid, nb.targetid, cc_phys_id_);
      MPI_Isend(utsbuf[nb.bufid], 2, MPI_UINT64_T, nb.snb.rank, tag,
                MPI_COMM_WORLD, &(utsreq[nb.bufid]));
    }
  }
#pragma omp barrier
  // wait for recv and set the data
#pragma omp critical(utofu_init)
  for (int n=0; n<pbval_->nneighbor; n++) {
    NeighborBlock& nb = pbval_->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      MPI_Wait(&(utrreq[nb.bufid]), MPI_STATUS_IGNORE);
      MPI_Request_free(&utrreq[nb.bufid]);
      bd_var_.rvcq[nb.bufid] = utrbuf[nb.bufid][0];
      bd_var_.rra[nb.bufid] = utrbuf[nb.bufid][1];
      utofu_set_vcq_id_path(&bd_var_.rvcq[nb.bufid], NULL);
    }
  }
#pragma omp barrier
  // wait for send
#pragma omp critical(utofu_init)
  for (int n=0; n<pbval_->nneighbor; n++) {
    NeighborBlock& nb = pbval_->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      MPI_Wait(&(utsreq[nb.bufid]), MPI_STATUS_IGNORE);
      MPI_Request_free(&utsreq[nb.bufid]);
    }
  }
#pragma omp barrier
  if (fflux_ && ((pmy_mesh_->multilevel)
      || (pbval_->shearing_box != 0))) { // SMR or AMR or SHEARING_BOX
#pragma omp critical(utofu_init)
    for (int n=0; n<pbval_->nneighbor; n++) {
      NeighborBlock& nb = pbval_->neighbor[n];
      if (nb.snb.rank != Globals::my_rank && nb.ni.type == NeighborConnect::face) {
        tag = pbval_->CreateBvalsMPITag(pmb->lid, nb.bufid, cc_flx_phys_id_);
        MPI_Irecv(utrbuf[nb.bufid], 2, MPI_UINT64_T, nb.snb.rank, tag,
                  MPI_COMM_WORLD, &(utrreq[nb.bufid]));
        utsbuf[nb.bufid][0] = bd_var_flcor_.vcq_id;
        utsbuf[nb.bufid][1] = bd_var_flcor_.lra[nb.bufid];
        tag = pbval_->CreateBvalsMPITag(nb.snb.lid, nb.targetid, cc_flx_phys_id_);
        MPI_Isend(utsbuf[nb.bufid], 2, MPI_UINT64_T, nb.snb.rank, tag,
                  MPI_COMM_WORLD, &(utsreq[nb.bufid]));
      }
    }
#pragma omp barrier
    // wait for recv and set the data
#pragma omp critical(utofu_init)
    for (int n=0; n<pbval_->nneighbor; n++) {
      NeighborBlock& nb = pbval_->neighbor[n];
      if (nb.snb.rank != Globals::my_rank && nb.ni.type == NeighborConnect::face) {
        MPI_Wait(&(utrreq[nb.bufid]), MPI_STATUS_IGNORE);
        MPI_Request_free(&utrreq[nb.bufid]);
        bd_var_flcor_.rvcq[nb.bufid] = utrbuf[nb.bufid][0];
        bd_var_flcor_.rra[nb.bufid] = utrbuf[nb.bufid][1];
        utofu_set_vcq_id_path(&bd_var_flcor_.rvcq[nb.bufid], NULL);
      }
    }
#pragma omp barrier
    // wait for send
#pragma omp critical(utofu_init)
    for (int n=0; n<pbval_->nneighbor; n++) {
      NeighborBlock& nb = pbval_->neighbor[n];
      if (nb.snb.rank != Globals::my_rank && nb.ni.type == NeighborConnect::face) {
        MPI_Wait(&(utsreq[nb.bufid]), MPI_STATUS_IGNORE);
        MPI_Request_free(&utsreq[nb.bufid]);
      }
    }
  }
#pragma omp barrier
#endif

  int f2 = pmy_mesh_->f2, f3 = pmy_mesh_->f3;
  int cng, cng1, cng2, cng3;
  cng  = cng1 = pmb->cnghost;
  cng2 = cng*f2;
  cng3 = cng*f3;
  int ssize, rsize;
  // Initialize non-polar neighbor communications to other ranks
  for (int n=0; n<pbval_->nneighbor; n++) {
    NeighborBlock& nb = pbval_->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      if (nb.snb.level == mylevel) { // same
        ssize = rsize = ((nb.ni.ox1 == 0) ? pmb->block_size.nx1 : NGHOST)
              *((nb.ni.ox2 == 0) ? pmb->block_size.nx2 : NGHOST)
              *((nb.ni.ox3 == 0) ? pmb->block_size.nx3 : NGHOST);
      } else if (nb.snb.level < mylevel) { // coarser
        ssize = ((nb.ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1)/2) : NGHOST)
              *((nb.ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1)/2) : NGHOST)
              *((nb.ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1)/2) : NGHOST);
        rsize = ((nb.ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1)/2 + cng1) : cng1)
              *((nb.ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1)/2 + cng2) : cng2)
              *((nb.ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1)/2 + cng3) : cng3);
      } else { // finer
        ssize = ((nb.ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1)/2 + cng1) : cng1)
              *((nb.ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1)/2 + cng2) : cng2)
              *((nb.ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1)/2 + cng3) : cng3);
        rsize = ((nb.ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1)/2) : NGHOST)
              *((nb.ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1)/2) : NGHOST)
              *((nb.ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1)/2) : NGHOST);
      }
      ssize *= (nu_ + 1); rsize *= (nu_ + 1);
      bd_var_.ssize[nb.bufid] = ssize;
      bd_var_.rsize[nb.bufid] = rsize;
      // specify the offsets in the view point of the target block: flip ox? signs

      // Initialize persistent communication requests attached to specific BoundaryData
      // cell-centered hydro: bd_hydro_
#ifdef UTOFU_PARALLEL
      bd_var_.recv[nb.bufid][rsize-1] = not_arrived_;
      utofu_prepare_put(bd_var_.vcq_hdl, bd_var_.rvcq[nb.bufid], bd_var_.lsa[nb.bufid],
            bd_var_.rra[nb.bufid], bd_var_.ssize[nb.bufid]*sizeof(Real), 0, uflags,
            bd_var_.toqd[nb.bufid], &bd_var_.toqdsize[nb.bufid]);
#else
      tag = pbval_->CreateBvalsMPITag(nb.snb.lid, nb.targetid, cc_phys_id_);
      if (bd_var_.req_send[nb.bufid] != MPI_REQUEST_NULL)
        MPI_Request_free(&bd_var_.req_send[nb.bufid]);
      MPI_Send_init(bd_var_.send[nb.bufid], ssize, MPI_ATHENA_REAL,
                    nb.snb.rank, tag, MPI_COMM_WORLD, &(bd_var_.req_send[nb.bufid]));
      tag = pbval_->CreateBvalsMPITag(pmb->lid, nb.bufid, cc_phys_id_);
      if (bd_var_.req_recv[nb.bufid] != MPI_REQUEST_NULL)
        MPI_Request_free(&bd_var_.req_recv[nb.bufid]);
      MPI_Recv_init(bd_var_.recv[nb.bufid], rsize, MPI_ATHENA_REAL,
                    nb.snb.rank, tag, MPI_COMM_WORLD, &(bd_var_.req_recv[nb.bufid]));
#endif
      // hydro flux correction: bd_var_flcor_
      if (fflux_ && nb.ni.type == NeighborConnect::face) {
        if (nb.snb.level != mylevel) {
          int size;
          if (nb.fid == 0 || nb.fid == 1)
            size = ((pmb->block_size.nx2 + 1)/2)*((pmb->block_size.nx3 + 1)/2);
          else if (nb.fid == 2 || nb.fid == 3)
            size = ((pmb->block_size.nx1 + 1)/2)*((pmb->block_size.nx3 + 1)/2);
          else // (nb.fid == 4 || nb.fid == 5)
            size = ((pmb->block_size.nx1 + 1)/2)*((pmb->block_size.nx2 + 1)/2);
          size *= (nu_ + 1);
          bd_var_flcor_.recv[nb.bufid][size-1] = not_arrived_;
          if (nb.snb.level < mylevel) { // send to coarser
            bd_var_flcor_.ssize[nb.bufid] = size;
#ifdef UTOFU_PARALLEL
            utofu_prepare_put(bd_var_flcor_.vcq_hdl, bd_var_flcor_.rvcq[nb.bufid],
                  bd_var_flcor_.lsa[nb.bufid], bd_var_flcor_.rra[nb.bufid],
                  bd_var_flcor_.ssize[nb.bufid]*sizeof(Real), 0, uflags,
                  bd_var_flcor_.toqd[nb.bufid], &bd_var_flcor_.toqdsize[nb.bufid]);
#else
            tag = pbval_->CreateBvalsMPITag(nb.snb.lid, nb.targetid, cc_flx_phys_id_);
            if (bd_var_flcor_.req_send[nb.bufid] != MPI_REQUEST_NULL)
              MPI_Request_free(&bd_var_flcor_.req_send[nb.bufid]);
            MPI_Send_init(bd_var_flcor_.send[nb.bufid], size, MPI_ATHENA_REAL,
                          nb.snb.rank, tag, MPI_COMM_WORLD,
                          &(bd_var_flcor_.req_send[nb.bufid]));
#endif
          } else if (nb.snb.level > mylevel) { // receive from finer
            bd_var_flcor_.rsize[nb.bufid] = size;
#ifndef UTOFU_PARALLEL
            tag = pbval_->CreateBvalsMPITag(pmb->lid, nb.bufid, cc_flx_phys_id_);
            if (bd_var_flcor_.req_recv[nb.bufid] != MPI_REQUEST_NULL)
              MPI_Request_free(&bd_var_flcor_.req_recv[nb.bufid]);
            MPI_Recv_init(bd_var_flcor_.recv[nb.bufid], size, MPI_ATHENA_REAL,
                          nb.snb.rank, tag, MPI_COMM_WORLD,
                          &(bd_var_flcor_.req_recv[nb.bufid]));
#endif
          }
        } else { // communication with same level
          if (nb.shear && (nb.fid == BoundaryFace::inner_x1
                           || nb.fid == BoundaryFace::outer_x1)
              && pbval_->shearing_box == 1) {
            int size = pmb->block_size.nx2*pmb->block_size.nx3*(nu_+1);
            tag = pbval_->CreateBvalsMPITag(nb.snb.lid, nb.targetid, cc_flx_phys_id_);
            if (bd_var_flcor_.req_send[nb.bufid] != MPI_REQUEST_NULL)
              MPI_Request_free(&bd_var_flcor_.req_send[nb.bufid]);
            MPI_Send_init(bd_var_flcor_.send[nb.bufid], size, MPI_ATHENA_REAL,
                          nb.snb.rank, tag, MPI_COMM_WORLD,
                          &(bd_var_flcor_.req_send[nb.bufid]));
            tag = pbval_->CreateBvalsMPITag(pmb->lid, nb.bufid, cc_flx_phys_id_);
            if (bd_var_flcor_.req_recv[nb.bufid] != MPI_REQUEST_NULL)
              MPI_Request_free(&bd_var_flcor_.req_recv[nb.bufid]);
            MPI_Recv_init(bd_var_flcor_.recv[nb.bufid], size, MPI_ATHENA_REAL,
                          nb.snb.rank, tag, MPI_COMM_WORLD,
                          &(bd_var_flcor_.req_recv[nb.bufid]));
          }
        }
      }
    }
  }
#ifdef UTOFU_PARALLEL
#pragma omp barrier
#pragma omp single
  MPI_Barrier(MPI_COMM_WORLD);
#pragma omp barrier
#endif
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::StartReceiving(BoundaryCommSubset phase)
//! \brief initiate MPI_Irecv()

void CellCenteredBoundaryVariable::StartReceiving(BoundaryCommSubset phase) {
  MeshBlock *pmb = pmy_block_;
#ifdef MPI_PARALLEL
  int mylevel = pmb->loc.level;
  for (int n=0; n<pbval_->nneighbor; n++) {
    NeighborBlock& nb = pbval_->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
#ifndef UTOFU_PARALLEL
      MPI_Start(&(bd_var_.req_recv[nb.bufid]));
#endif
      if (fflux_ && phase == BoundaryCommSubset::all
                 && nb.ni.type == NeighborConnect::face) {
        if ((nb.shear&&(nb.fid == BoundaryFace::inner_x1
                     || nb.fid == BoundaryFace::outer_x1)
          && pbval_->shearing_box==1) || nb.snb.level > mylevel) {
#ifndef UTOFU_PARALLEL
          MPI_Start(&(bd_var_flcor_.req_recv[nb.bufid]));
#endif
        } else { // no recv
          bd_var_flcor_.flag[nb.bufid] = BoundaryStatus::completed;
        }
      }
    }
  }
#endif
  if (pbval_->shearing_box == 1) {
    int ssize = pbval_->ssize_;
    int nx3 = pmb->block_size.nx3;
    // TODO(KGF): clear sflag arrays
    if (phase == BoundaryCommSubset::all) {
      for (int upper=0; upper<2; upper++) {
        if (pbval_->is_shear[upper]) {
          int *counts1 = pbval_->sb_flux_data_[upper].send_count;
          int *counts2 = pbval_->sb_flux_data_[upper].recv_count;
          for (int n=0; n<3; n++) {
            if (counts1[n]>0) {
              shear_send_count_flx_[upper][n] = counts1[n]*nx3;
              shear_bd_flux_[upper].sflag[n] = BoundaryStatus::waiting;
            } else {
              shear_send_count_flx_[upper][n] = 0;
              shear_bd_flux_[upper].sflag[n] = BoundaryStatus::completed;
            }
            if (counts2[n]>0) {
              shear_recv_count_flx_[upper][n] = counts2[n]*nx3;
              shear_bd_flux_[upper].flag[n] = BoundaryStatus::waiting;
            } else {
              shear_recv_count_flx_[upper][n] = 0;
              shear_bd_flux_[upper].flag[n] = BoundaryStatus::completed;
            }
          }
        }
      }
    }
    for (int upper=0; upper<2; upper++) {
      if (pbval_->is_shear[upper]) {
        int *counts1 = pbval_->sb_data_[upper].send_count;
        int *counts2 = pbval_->sb_data_[upper].recv_count;
        for (int n=0; n<4; n++) {
          if (counts1[n]>0) {
            shear_send_count_cc_[upper][n] = counts1[n]*ssize;
            shear_bd_var_[upper].sflag[n] = BoundaryStatus::waiting;
          } else {
            shear_send_count_cc_[upper][n] = 0;
            shear_bd_var_[upper].sflag[n] = BoundaryStatus::completed;
          }
          if (counts2[n]>0) {
            shear_recv_count_cc_[upper][n] = counts2[n]*ssize;
            shear_bd_var_[upper].flag[n] = BoundaryStatus::waiting;
          } else {
            shear_recv_count_cc_[upper][n] = 0;
            shear_bd_var_[upper].flag[n] = BoundaryStatus::completed;
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::ClearBoundary(BoundaryCommSubset phase)
//! \brief clean up the boundary flags after each loop

void CellCenteredBoundaryVariable::ClearBoundary(BoundaryCommSubset phase) {
  MeshBlock *pmb = pmy_block_;
  int mylevel = pmb->loc.level;

  for (int n=0; n<pbval_->nneighbor; n++) {
    NeighborBlock& nb = pbval_->neighbor[n];
    bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
    bd_var_.sflag[nb.bufid] = BoundaryStatus::waiting;

    if (fflux_ && nb.ni.type == NeighborConnect::face) {
      bd_var_flcor_.flag[nb.bufid] = BoundaryStatus::waiting;
      bd_var_flcor_.sflag[nb.bufid] = BoundaryStatus::waiting;
    }
  }

#ifdef MPI_PARALLEL
#ifdef UTOFU_PARALLEL
  int rc;
  void *cbdata;
  while (bd_var_.sentcount > 0) {
    rc = utofu_poll_tcq(bd_var_.vcq_hdl, 0, &cbdata);
    if (rc == UTOFU_SUCCESS)
      bd_var_.sentcount--;
  }
  if (fflux_ && phase == BoundaryCommSubset::all) {
    while (bd_var_flcor_.sentcount > 0) {
      rc = utofu_poll_tcq(bd_var_flcor_.vcq_hdl, 0, &cbdata);
      if (rc == UTOFU_SUCCESS)
        bd_var_flcor_.sentcount--;
    }
  }
#else
  for (int n=0; n<pbval_->nneighbor; n++) {
    NeighborBlock& nb = pbval_->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      // Wait for Isend
      MPI_Wait(&(bd_var_.req_send[nb.bufid]), MPI_STATUS_IGNORE);
      if (fflux_ && phase == BoundaryCommSubset::all
                 && nb.ni.type == NeighborConnect::face) {
        if ((nb.shear && (nb.fid == BoundaryFace::inner_x1
                       || nb.fid == BoundaryFace::outer_x1)
             && pbval_->shearing_box==1) || nb.snb.level < mylevel) {
          MPI_Wait(&(bd_var_flcor_.req_send[nb.bufid]), MPI_STATUS_IGNORE);
        }
      }
    }
  }
#endif
#endif

  // clear shearing box boundary communications
  if (pbval_->shearing_box ==1) {
    // TODO(KGF): clear sflag arrays
    if (phase == BoundaryCommSubset::all) {
      for (int upper=0; upper<2; upper++) {
        if (pbval_->is_shear[upper]) {
          for (int n=0; n<3; n++) {
            if (pbval_->sb_flux_data_[upper].send_neighbor[n].rank == -1) continue;
            shear_bd_flux_[upper].flag[n] = BoundaryStatus::waiting;
#ifdef MPI_PARALLEL
            if (pbval_->sb_flux_data_[upper].send_neighbor[n].rank != Globals::my_rank) {
              MPI_Wait(&shear_bd_flux_[upper].req_send[n], MPI_STATUS_IGNORE);
            }
#endif
          }
        }
      }
    }
    for (int upper=0; upper<2; upper++) {
      if (pbval_->is_shear[upper]) {
        for (int n=0; n<4; n++) {
          if (pbval_->sb_data_[upper].send_neighbor[n].rank == -1) continue;
          shear_bd_var_[upper].flag[n] = BoundaryStatus::waiting;
#ifdef MPI_PARALLEL
          if (pbval_->sb_data_[upper].send_neighbor[n].rank != Globals::my_rank) {
            MPI_Wait(&shear_bd_var_[upper].req_send[n], MPI_STATUS_IGNORE);
          }
#endif
        }
      }
    }
  }
  return;
}
