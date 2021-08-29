//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file new_blockdt.cpp
//! \brief computes timestep using CFL condition on a MEshBlock

// C headers

// C++ headers
#include <algorithm>  // min()
#include <cmath>      // fabs(), sqrt()
#include <limits>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../field/field_diffusion/field_diffusion.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../scalars/scalars.hpp"
#include "hydro.hpp"
#include "hydro_diffusion/hydro_diffusion.hpp"

// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn void Hydro::NewBlockTimeStep()
//! \brief calculate the minimum timestep within a MeshBlock

void Hydro::NewBlockTimeStep() {
  MeshBlock *pmb = pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  const AthenaArray<Real> &w = pmb->phydro->w;
  const AthenaArray<Real> &bcc = pmb->pfield->bcc;
  // hyperbolic timestep constraint in each (x1-slice) cell along coordinate direction:
  AthenaArray<Real> &dt1 = dt1_, &dt2 = dt2_, &dt3 = dt3_;  // (x1 slices)
  Real wi[NWAVE];

  Real real_max = std::numeric_limits<Real>::max();
  Real min_dt = real_max;
  // Note, "dt_hyperbolic" currently refers to the dt limit imposed by evoluiton of the
  // ideal hydro or MHD fluid by the main integrator (even if not strictly hyperbolic)
  Real min_dt_hyperbolic  = real_max;
  // TODO(felker): consider renaming dt_hyperbolic after general execution model is
  // implemented and flexibility from #247 (zero fluid configurations) is
  // addressed. dt_hydro, dt_main (inaccurate since "dt" is actually main), dt_MHD?
  Real min_dt_parabolic  = real_max;
  Real min_dt_user  = real_max;
  Real gamma = pmb->peos->GetGamma();

  // TODO(felker): skip this next loop if pm->fluid_setup == FluidFormulation::disabled
  FluidFormulation fluid_status = pmb->pmy_mesh->fluid_setup;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pmb->pcoord->CenterWidth1(k, j, is, ie, dt1);
      pmb->pcoord->CenterWidth2(k, j, is, ie, dt2);
      pmb->pcoord->CenterWidth3(k, j, is, ie, dt3);
#pragma clang loop vectorize(assume_safety)
      for (int i=is; i<=ie; ++i) {
        Real idn = 1.0/w(IDN,k,j,i);
        Real wvx = w(IVX,k,j,i);
        Real wvy = w(IVY,k,j,i);
        Real wvz = w(IVZ,k,j,i);
        Real wpr = w(IPR,k,j,i);
        Real wbx = bcc(IB1,k,j,i);
        Real wby = bcc(IB2,k,j,i);
        Real wbz = bcc(IB3,k,j,i);

        Real bx2 = wbx*wbx;
        Real by2 = wby*wby;
        Real bz2 = wbz*wbz;
        Real bsq = bx2 + by2 + bz2;
        Real asq = gamma*wpr;
        Real qsq = bsq + asq;
        Real tmp = bsq - asq;
        Real tsq = tmp * tmp;
        Real cfx = std::sqrt(0.5*(qsq + std::sqrt(tsq + 4.0*asq*(by2+bz2)))*idn);
        Real cfy = std::sqrt(0.5*(qsq + std::sqrt(tsq + 4.0*asq*(bx2+bz2)))*idn);
        Real cfz = std::sqrt(0.5*(qsq + std::sqrt(tsq + 4.0*asq*(bx2+by2)))*idn);
        dt1(i) /= (std::abs(wvx) + cfx);
        dt2(i) /= (std::abs(wvy) + cfy);
        dt3(i) /= (std::abs(wvz) + cfz);
      }

      // compute minimum of (v1 +/- C)
#pragma clang loop vectorize(assume_safety)
      for (int i=is; i<=ie; ++i) {
        min_dt_hyperbolic = std::min(min_dt_hyperbolic, dt1(i));
      }

      // if grid is 2D/3D, compute minimum of (v2 +/- C)
      if (pmb->block_size.nx2 > 1) {
#pragma clang loop vectorize(assume_safety)
        for (int i=is; i<=ie; ++i) {
          min_dt_hyperbolic = std::min(min_dt_hyperbolic, dt2(i));
        }
      }

      // if grid is 3D, compute minimum of (v3 +/- C)
      if (pmb->block_size.nx3 > 1) {
#pragma clang loop vectorize(assume_safety)
        for (int i=is; i<=ie; ++i) {
          min_dt_hyperbolic = std::min(min_dt_hyperbolic, dt3(i));
        }
      }
    }
  }

  // calculate the timestep limited by the diffusion processes
  if (hdif.hydro_diffusion_defined) {
    Real min_dt_vis, min_dt_cnd;
    hdif.NewDiffusionDt(min_dt_vis, min_dt_cnd);
    min_dt_parabolic = std::min(min_dt_parabolic, min_dt_vis);
    min_dt_parabolic = std::min(min_dt_parabolic, min_dt_cnd);
  } // hydro diffusion

  if (MAGNETIC_FIELDS_ENABLED &&
      pmb->pfield->fdif.field_diffusion_defined) {
    Real min_dt_oa, min_dt_hall;
    pmb->pfield->fdif.NewDiffusionDt(min_dt_oa, min_dt_hall);
    min_dt_parabolic = std::min(min_dt_parabolic, min_dt_oa);
    // Hall effect is dispersive, not diffusive:
    min_dt_hyperbolic = std::min(min_dt_hyperbolic, min_dt_hall);
  } // field diffusion

  if (NSCALARS > 0 && pmb->pscalars->scalar_diffusion_defined) {
    Real min_dt_scalar_diff = pmb->pscalars->NewDiffusionDt();
    min_dt_parabolic = std::min(min_dt_parabolic, min_dt_scalar_diff);
  } // passive scalar diffusion

  min_dt_hyperbolic *= pmb->pmy_mesh->cfl_number;
  // scale the theoretical stability limit by a safety factor = the hyperbolic CFL limit
  // (user-selected or automaticlaly enforced). May add independent parameter "cfl_diff"
  // in the future (with default = cfl_number).
  min_dt_parabolic *= pmb->pmy_mesh->cfl_number;

  // For orbital advection, give a restriction on dt_hyperbolic.
  if (pmb->porb->orbital_advection_active) {
    Real min_dt_orb = pmb->porb->NewOrbitalAdvectionDt();
    min_dt_hyperbolic = std::min(min_dt_hyperbolic, min_dt_orb);
  }

  // set main integrator timestep as the minimum of the appropriate timestep constraints:
  // hyperbolic: (skip if fluid is nonexistent or frozen)
  min_dt = std::min(min_dt, min_dt_hyperbolic);
  // user:
  if (UserTimeStep_ != nullptr) {
    min_dt_user = UserTimeStep_(pmb);
    min_dt = std::min(min_dt, min_dt_user);
  }
  // parabolic:
  // STS handles parabolic terms -> then take the smaller of hyperbolic or user timestep
  if (!STS_ENABLED) {
    // otherwise, take the smallest of the hyperbolic, parabolic, user timesteps
    min_dt = std::min(min_dt, min_dt_parabolic);
  }
  pmb->new_block_dt_ = min_dt;
  pmb->new_block_dt_hyperbolic_ = min_dt_hyperbolic;
  pmb->new_block_dt_parabolic_ = min_dt_parabolic;
  pmb->new_block_dt_user_ = min_dt_user;

  return;
}
