//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hlld.cpp
//! \brief HLLD Riemann solver for adiabatic MHD.
//!
//! REFERENCES:
//! - T. Miyoshi & K. Kusano, "A multi-state HLL approximate Riemann solver for ideal
//!   MHD", JCP, 208, 315 (2005)

// C headers

// C++ headers
#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

// Athena++ headers
#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../../../eos/eos.hpp"
#include "../../../mesh/mesh.hpp"
#include "../../hydro.hpp"

// container to store (density, momentum, total energy, tranverse magnetic field)
// minimizes changes required to adopt athena4.2 version of this solver
struct Cons1D {
  Real d, mx, my, mz, e, by, bz;
};

#define SMALL_NUMBER 1.0e-8

//----------------------------------------------------------------------------------------

void Hydro::RiemannSolver(const int k, const int j, const int il, const int iu,
                          const int ivx, const AthenaArray<Real> &bx,
                          const AthenaArray<Real> &wl, const AthenaArray<Real> &wr,
                          AthenaArray<Real> &flx,
                          AthenaArray<Real> &ey, AthenaArray<Real> &ez,
                          AthenaArray<Real> &wct, const AthenaArray<Real> &dxw) {
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  const int N = 128;
  static Real fdn[N], fvx[N], fvy[N], fvz[N], fen[N], eya[N], eza[N], wcta[N];

  EquationOfState *peos = pmy_block->peos;
  Real gm = peos->GetGamma();
  Real igm1 = 1.0 / (gm - 1.0);
  Real dt = pmy_block->pmy_mesh->dt;

//#pragma omp simd
#pragma clang loop vectorize(assume_safety)
#pragma fj loop loop_fission_target
  for (int i=il; i<=iu; ++i) {
    //--- Step 1.  Load L/R states into local variables
    Real wldn=wl(IDN,i);
    Real wlvx=wl(ivx,i);
    Real wlvy=wl(ivy,i);
    Real wlvz=wl(ivz,i);
    Real wlpr=wl(IPR,i);
    Real wlby=wl(IBY,i);
    Real wlbz=wl(IBZ,i);

    Real wrdn=wr(IDN,i);
    Real wrvx=wr(ivx,i);
    Real wrvy=wr(ivy,i);
    Real wrvz=wr(ivz,i);
    Real wrpr=wr(IPR,i);
    Real wrby=wr(IBY,i);
    Real wrbz=wr(IBZ,i);

    Real bxi = bx(k,j,i);

    // Compute L/R states for selected conserved variables
    Real bxsq = bxi*bxi;
    // (KGF): group transverse vector components for floating-point associativity symmetry
    Real pbl = 0.5*(bxsq + (SQR(wlby) + SQR(wlbz)));  // magnetic pressuren (l/r)
    Real pbr = 0.5*(bxsq + (SQR(wrby) + SQR(wrbz)));
    Real kel = 0.5*wldn*(SQR(wlvx) + (SQR(wlvy) + SQR(wlvz)));
    Real ker = 0.5*wrdn*(SQR(wrvx) + (SQR(wrvy) + SQR(wrvz)));

    const Real &uldn  = wldn;
    const Real &urdn  = wrdn;

    //--- Step 2.  Compute L & R wave speeds according to Miyoshi & Kusano, eqn. (67)

    Real asql = gm*wlpr;
    Real asqr = gm*wrpr;
    Real ct2l = wlby*wlby+wlbz*wlbz;
    Real ct2r = wrby*wrby+wrbz*wrbz;
    Real qsql = bxsq+ct2l+asql;
    Real qsqr = bxsq+ct2r+asqr;
    Real tmpl = bxsq+ct2l-asql;
    Real tmpr = bxsq+ct2r-asqr;
    Real cfl = std::sqrt(0.5*(qsql+std::sqrt(tmpl*tmpl+4.0*asql*ct2l))/wldn);
    Real cfr = std::sqrt(0.5*(qsqr+std::sqrt(tmpr*tmpr+4.0*asqr*ct2r))/wrdn);

    Real spd0 = std::min( wlvx-cfl, wrvx-cfr );
    Real spd4 = std::max( wlvx+cfl, wrvx+cfr );

    Real ptl = wlpr + pbl; // total pressurens L,R
    Real ptr = wrpr + pbr;

    //--- Step 4.  Compute middle and Alfven wave speeds

    Real sdl = spd0 - wlvx;  // S_i-u_i (i=L or R)
    Real sdr = spd4 - wrvx;

    Real ulmx = wlvx*uldn;
    Real ulmy = wlvy*uldn;
    Real ulmz = wlvz*uldn;
    Real ulen = wlpr*igm1 + kel + pbl;
    Real ulby = wlby;
    Real ulbz = wlbz;
    Real urmx = wrvx*urdn;
    Real urmy = wrvy*urdn;
    Real urmz = wrvz*urdn;
    Real uren = wrpr*igm1 + ker + pbr;
    Real urby = wrby;
    Real urbz = wrbz;
    Real udnsdl = uldn*sdl;
    Real udnsdr = urdn*sdr;

    // S_M: eqn (38) of Miyoshi & Kusano
    // (KGF): group ptl, ptr terms for floating-point associativity symmetry
    Real spd2 = (sdr*urmx - sdl*ulmx + (ptl - ptr))/(udnsdr - udnsdl);

    // eqn (23) explicitly becomes eq (41) of Miyoshi & Kusano
    // TODO(felker): place an assertion that ptstl==ptstr
    Real ptstl = ptl + udnsdl*(spd2-wlvx);
    Real ptstr = ptr + udnsdr*(spd2-wrvx);
    Real ptst = 0.5*(ptstr + ptstl);  // total pressuren (star state)

    Real umx, umy, umz, wvx, wvy, wvz, wpt, uen, uby, ubz;
    if (spd0 >= 0.0) {
      umx = ulmx;
      umy = ulmy;
      umz = ulmz;
      wvx = wlvx;
      wvy = wlvy;
      wvz = wlvz;
      wpt = ptl;
      uen = ulen;
      uby = ulby;
      ubz = ulbz;
    } else if (spd4 <= 0.0) {
      umx = urmx;
      umy = urmy;
      umz = urmz;
      wvx = wrvx;
      wvy = wrvy;
      wvz = wrvz;
      wpt = ptr;
      uen = uren;
      uby = urby;
      ubz = urbz;
    } else {
      // intermediate states
      Real sdml   = spd0 - spd2;  // S_i-S_M (i=L or R)
      Real sdmr   = spd4 - spd2;
      Real sdml_inv = 1.0/sdml;
      Real sdmr_inv = 1.0/sdmr;
      Real ulstdn = udnsdl * sdml_inv;
      Real urstdn = udnsdr * sdmr_inv;
      Real sqrtdl = std::sqrt(ulstdn);
      Real sqrtdr = std::sqrt(urstdn);
      Real spd1 = spd2 - std::abs(bxi)/sqrtdl;
      Real spd3 = spd2 + std::abs(bxi)/sqrtdr;
      wpt = ptst;

      // calculate Ul*, Ur* and Fl*, Fr*
      Real ulstmy, ulstmz, ulstby, ulstbz;
      Real urstmy, urstmz, urstby, urstbz;
      Real ulstmx = ulstdn * spd2;
      Real urstmx = urstdn * spd2;
      Real tmpl = udnsdl*sdml - bxsq;
      Real tmpr = udnsdr*sdmr - bxsq;
      if (std::abs(tmpl) < (SMALL_NUMBER)*ptst) {
        // Degenerate case
        ulstmy = ulstdn * wlvy;
        ulstmz = ulstdn * wlvz;

        ulstby = ulby;
        ulstbz = ulbz;
      } else {
        // eqns (44) and (46) of M&K
        Real tmp = bxi*(sdl - sdml)/tmpl;
        ulstmy = ulstdn * (wlvy - ulby*tmp);
        ulstmz = ulstdn * (wlvz - ulbz*tmp);

        // eqns (45) and (47) of M&K
        Real tmp2 = (uldn*SQR(sdl) - bxsq)/tmpl;
        ulstby = ulby * tmp2;
        ulstbz = ulbz * tmp2;
      }
      if (std::abs(tmpr) < (SMALL_NUMBER)*ptst) {
        // Degenerate case
        urstmy = urstdn * wrvy;
        urstmz = urstdn * wrvz;

        urstby = urby;
        urstbz = urbz;
      } else {
        // eqns (44) and (46) of M&K
        Real tmp = bxi*(sdr - sdmr)/tmpr;
        urstmy = urstdn * (wrvy - urby*tmp);
        urstmz = urstdn * (wrvz - urbz*tmp);

        // eqns (45) and (47) of M&K
        Real tmp2 = (urdn*SQR(sdr) - bxsq)/tmpr;
        urstby = urby * tmp2;
        urstbz = urbz * tmp2;
      }
      Real ulst_d_inv = 1.0/ulstdn;
      Real vbstl = (ulstmx*bxi + (ulstmy*ulstby + ulstmz*ulstbz))*ulst_d_inv;
      Real ulsten = (sdl*ulen - ptl*wlvx + ptst*spd2 +
                     bxi*(wlvx*bxi + (wlvy*ulby + wlvz*ulbz) - vbstl))*sdml_inv;
      Real urst_d_inv = 1.0/urstdn;
      Real vbstr = (urstmx*bxi+(urstmy*urstby+urstmz*urstbz))*urst_d_inv;
      Real ursten = (sdr*uren - ptr*wrvx + ptst*spd2 +
                     bxi*(wrvx*bxi + (wrvy*urby + wrvz*urbz) - vbstr))*sdmr_inv;
      if (spd1 >= 0.0) {
        umx = ulstmx;
        umy = ulstmy;
        umz = ulstmz;
        wvx = ulstmx*ulst_d_inv;
        wvy = ulstmy*ulst_d_inv;
        wvz = ulstmz*ulst_d_inv;
        uen = ulsten;
        uby = ulstby;
        ubz = ulstbz;
      } else if (spd3 <= 0.0) {
        umx = urstmx;
        umy = urstmy;
        umz = urstmz;
        wvx = urstmx*urst_d_inv;
        wvy = urstmy*urst_d_inv;
        wvz = urstmz*urst_d_inv;
        uen = ursten;
        uby = urstby;
        ubz = urstbz;
      } else {
        // ** states
        if (0.5*bxsq < (SMALL_NUMBER)*ptst) {
          if (spd2 > 0.0) {
            umx = ulstmx;
            umy = ulstmy;
            umz = ulstmz;
            wvx = ulstmx*ulst_d_inv;
            wvy = ulstmy*ulst_d_inv;
            wvz = ulstmz*ulst_d_inv;
            uen = ulsten;
            uby = ulstby;
            ubz = ulstbz;
          } else {
            umx = urstmx;
            umy = urstmy;
            umz = urstmz;
            wvx = urstmx*urst_d_inv;
            wvy = urstmy*urst_d_inv;
            wvz = urstmz*urst_d_inv;
            uen = ursten;
            uby = urstby;
            ubz = urstbz;
          }
        } else {
          Real invsumd = 1.0/(sqrtdl + sqrtdr);
          Real bxsig = (bxi > 0.0 ? 1.0 : -1.0);
          Real udstdn;
          if (spd2 > 0.0) {
            udstdn = ulstdn;
            umx = ulstmx;
          } else {
            udstdn = urstdn;
            umx = urstmx;
          }

          wvx = umx/udstdn;

          // eqn (59) of M&K
          wvy = invsumd*(sqrtdl*(ulstmy*ulst_d_inv) + sqrtdr*(urstmy*urst_d_inv) +
                         bxsig*(urstby - ulstby));
          umy = udstdn * wvy;

          // eqn (60) of M&K
          wvz = invsumd*(sqrtdl*(ulstmz*ulst_d_inv) + sqrtdr*(urstmz*urst_d_inv) +
                         bxsig*(urstbz - ulstbz));
          umz = udstdn * wvz;

          // eqn (61) of M&K
          uby = invsumd*(sqrtdl*urstby + sqrtdr*ulstby +
                bxsig*sqrtdl*sqrtdr*((urstmy*urst_d_inv) - (ulstmy*ulst_d_inv)));

          // eqn (62) of M&K
          ubz = invsumd*(sqrtdl*urstbz + sqrtdr*ulstbz +
                bxsig*sqrtdl*sqrtdr*((urstmz*urst_d_inv) - (ulstmz*ulst_d_inv)));

          // eqn (63) of M&K
          Real vbdst = spd2*bxi + (wvy*uby + wvz*ubz);
          if (spd2 > 0.0)
            uen = ulsten - sqrtdl*bxsig*(vbstl - vbdst);
          else
            uen = ursten + sqrtdr*bxsig*(vbstr - vbdst);
        }
      }
    }

    fdn[i] = umx;
    fvx[i] = umx*wvx + wpt - bxsq;
    fvy[i] = umy*wvx - bxi*uby;
    fvz[i] = umz*wvx - bxi*ubz;
    fen[i] = wvx*(uen + wpt - bxsq) - bxi*(wvy*uby + wvz*ubz);
    eya[i] = -uby*wvx + bxi*wvy;
    eza[i] =  ubz*wvx - bxi*wvz;

    Real v_over_c = (1024.0)* dt * umx / (dxw(i) * (wldn + wrdn));
    Real tmp_min = std::min(static_cast<Real>(0.5), v_over_c);
    wcta[i] = 0.5 + std::max(static_cast<Real>(-0.5), tmp_min);
  }

#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    flx(IDN,k,j,i) = fdn[i];
    flx(ivx,k,j,i) = fvx[i];
    flx(ivy,k,j,i) = fvy[i];
    flx(ivz,k,j,i) = fvz[i];
    flx(IEN,k,j,i) = fen[i];
    ey(k,j,i) =  eya[i];
    ez(k,j,i) =  eza[i];
    wct(k,j,i) = wcta[i];
  }

  return;
}
