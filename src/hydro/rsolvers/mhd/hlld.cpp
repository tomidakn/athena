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

#pragma clang loop vectorize(assume_safety)
#pragma fj loop loop_fission_target
#pragma fj loop loop_fission_threshold 1
  for (int i=il; i<=iu; ++i) {

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
    Real ct2l = wlby*wlby + wlbz*wlbz;
    Real ct2r = wrby*wrby + wrbz*wrbz;
    Real bsql = bxsq + ct2l;
    Real bsqr = bxsq + ct2r;
    Real pbl = 0.5*bsql;  // magnetic pressure (l/r)
    Real pbr = 0.5*bsqr;
    Real kel = 0.5*wldn*(SQR(wlvx) + (SQR(wlvy) + SQR(wlvz)));
    Real ker = 0.5*wrdn*(SQR(wrvx) + (SQR(wrvy) + SQR(wrvz)));

    Real uldn = wldn;
    Real ulmx = wlvx*uldn;
    Real ulmy = wlvy*uldn;
    Real ulmz = wlvz*uldn;
    Real ulen = wlpr*igm1 + kel + pbl;
    Real ulby = wlby;
    Real ulbz = wlbz;

    Real urdn = wrdn;
    Real urmx = wrvx*urdn;
    Real urmy = wrvy*urdn;
    Real urmz = wrvz*urdn;
    Real uren = wrpr*igm1 + ker + pbr;
    Real urby = wrby;
    Real urbz = wrbz;


    Real asql = gm*wlpr;
    Real asqr = gm*wrpr;
    Real qsql = bsql+asql;
    Real qsqr = bsqr+asqr;
    Real dsql = bsql-asql;
    Real dsqr = bsqr-asqr;
    Real cfl = std::sqrt(0.5*(qsql+std::sqrt(dsql*dsql+4.0*asql*ct2l))/wldn);
    Real cfr = std::sqrt(0.5*(qsqr+std::sqrt(dsqr*dsqr+4.0*asqr*ct2r))/wrdn);

    Real spd0 = std::min( wlvx-cfl, wrvx-cfr );
    Real spd4 = std::max( wlvx+cfl, wrvx+cfr );

    //--- Step 3.  Compute L/R fluxes

    Real ptl = wlpr + pbl; // total pressures L,R
    Real ptr = wrpr + pbr;

    Real fldn = ulmx;
    Real flmx = ulmx*wlvx + ptl - bxsq;
    Real flmy = ulmy*wlvx - bxi*ulby;
    Real flmz = ulmz*wlvx - bxi*ulbz;
    Real flen = wlvx*(ulen + ptl - bxsq) - bxi*(wlvy*ulby + wlvz*ulbz);
    Real flby = ulby*wlvx - bxi*wlvy;
    Real flbz = ulbz*wlvx - bxi*wlvz;

    Real frdn = urmx;
    Real frmx = urmx*wrvx + ptr - bxsq;
    Real frmy = urmy*wrvx - bxi*urby;
    Real frmz = urmz*wrvx - bxi*urbz;
    Real fren = wrvx*(uren + ptr - bxsq) - bxi*(wrvy*urby + wrvz*urbz);
    Real frby = urby*wrvx - bxi*wrvy;
    Real frbz = urbz*wrvx - bxi*wrvz;

    //--- Step 4.  Compute middle and Alfven wave speeds

    Real sdl = spd0 - wlvx;  // S_i-u_i (i=L or R)
    Real sdr = spd4 - wrvx;

    // S_M: eqn (38) of Miyoshi & Kusano
    // (KGF): group ptl, ptr terms for floating-point associativity symmetry
    Real spd2 = (sdr*urmx - sdl*ulmx + (ptl - ptr))/(sdr*urdn - sdl*uldn);

    Real sdml   = spd0 - spd2;  // S_i-S_M (i=L or R)
    Real sdmr   = spd4 - spd2;
    Real sdml_inv = 1.0/sdml;
    Real sdmr_inv = 1.0/sdmr;
    Real uldnsd = uldn * sdl;
    Real urdnsd = urdn * sdr;
    // eqn (43) of Miyoshi & Kusano
    Real ulstdn = uldnsd * sdml_inv;
    Real urstdn = urdnsd * sdmr_inv;
    Real ulst_d_inv = 1.0/ulstdn;
    Real urst_d_inv = 1.0/urstdn;
    Real sqrtdl = std::sqrt(ulstdn);
    Real sqrtdr = std::sqrt(urstdn);

    // eqn (51) of Miyoshi & Kusano
    Real spd1 = spd2 - std::abs(bxi)/sqrtdl;
    Real spd3 = spd2 + std::abs(bxi)/sqrtdr;

    //--- Step 5.  Compute intermediate states
    // eqn (23) explicitly becomes eq (41) of Miyoshi & Kusano
    // TODO(felker): place an assertion that ptstl==ptstr
    Real ptstl = ptl + uldnsd*(spd2-wlvx);
    Real ptstr = ptr + urdnsd*(spd2-wrvx);
    Real ptst = 0.5*(ptstr + ptstl);  // total pressure (star state)

    // ul* - eqn (39) of M&K
    Real ulstmx = ulstdn * spd2;
    Real ulstmy, ulstmz, ulstby, ulstbz;
    Real tmpl = uldnsd*sdml-bxsq;
    if (std::abs(tmpl) < (SMALL_NUMBER)*ptst) {
      // Degenerate case
      ulstmy = ulstdn * wlvy;
      ulstmz = ulstdn * wlvz;

      ulstby = ulby;
      ulstbz = ulbz;
    } else {
      // eqns (44) and (46) of M&K
      Real itmp = 1.0/tmpl;
      Real tmp = bxi*(sdl - sdml)*itmp;
      ulstmy = ulstdn * (wlvy - ulby*tmp);
      ulstmz = ulstdn * (wlvz - ulbz*tmp);

      // eqns (45) and (47) of M&K
      Real tmp2 = (uldn*SQR(sdl) - bxsq)*itmp;
      ulstby = ulby * tmp2;
      ulstbz = ulbz * tmp2;
    }
    // v_i* dot B_i*
    // (KGF): group transverse momenta terms for floating-point associativity symmetry
    Real vbstl = (ulstmx*bxi+(ulstmy*ulstby+ulstmz*ulstbz))*ulst_d_inv;
    // eqn (48) of M&K
    // (KGF): group transverse by, bz terms for floating-point associativity symmetry
    Real ulsten = (sdl*ulen - ptl*wlvx + ptst*spd2 +
                   bxi*(wlvx*bxi + (wlvy*ulby + wlvz*ulbz) - vbstl))*sdml_inv;

    // ur* - eqn (39) of M&K
    Real urstmx = urstdn * spd2;
    Real urstmy, urstmz, urstby, urstbz;
    Real tmpr = urdnsd*sdmr - bxsq;
    if (std::abs(tmpr) < (SMALL_NUMBER)*ptst) {
      // Degenerate case
      urstmy = urstdn * wrvy;
      urstmz = urstdn * wrvz;

      urstby = urby;
      urstbz = urbz;
    } else {
      // eqns (44) and (46) of M&K
      Real itmp = 1.0/tmpr;
      Real tmp = bxi*(sdr - sdmr)*itmp;
      urstmy = urstdn * (wrvy - urby*tmp);
      urstmz = urstdn * (wrvz - urbz*tmp);

      // eqns (45) and (47) of M&K
      Real tmp2 = (urdn*SQR(sdr) - bxsq)*itmp;
      urstby = urby * tmp2;
      urstbz = urbz * tmp2;
    }
    // v_i* dot B_i*
    // (KGF): group transverse momenta terms for floating-point associativity symmetry
    Real vbstr = (urstmx*bxi+(urstmy*urstby+urstmz*urstbz))*urst_d_inv;
    // eqn (48) of M&K
    // (KGF): group transverse by, bz terms for floating-point associativity symmetry
    Real ursten = (sdr*uren - ptr*wrvx + ptst*spd2 +
                   bxi*(wrvx*bxi + (wrvy*urby + wrvz*urbz) - vbstr))*sdmr_inv;
    // ul** and ur** - if Bx is near zero, same as *-states

    Real uldstmy, uldstmz, uldstby, uldstbz, uldsten;
    Real urdstmy, urdstmz, urdstby, urdstbz, urdsten;
    Real uldstdn = ulstdn;
    Real urdstdn = urstdn;
    Real uldstmx = ulstmx;
    Real urdstmx = urstmx;
    if (0.5*bxsq < (SMALL_NUMBER)*ptst) {
      uldstmy = ulstmy;
      uldstmz = ulstmz;
      uldstby = ulstby;
      uldstbz = ulstbz;
      uldsten = ulsten;
      urdstmy = urstmy;
      urdstmz = urstmz;
      urdstby = urstby;
      urdstbz = urstbz;
      urdsten = ursten;
    } else {
      Real invsumd = 1.0/(sqrtdl + sqrtdr);
      Real bxsig = (bxi > 0.0 ? 1.0 : -1.0);

      // eqn (59) of M&K
      Real vydst = invsumd*(sqrtdl*(ulstmy*ulst_d_inv) + sqrtdr*(urstmy*urst_d_inv) +
                            bxsig*(urstby - ulstby));
      uldstmy = uldstdn * vydst;
      urdstmy = urdstdn * vydst;

      // eqn (60) of M&K
      Real vzdst = invsumd*(sqrtdl*(ulstmz*ulst_d_inv) + sqrtdr*(urstmz*urst_d_inv) +
                            bxsig*(urstbz - ulstbz));
      uldstmz = uldstdn * vzdst;
      urdstmz = urdstdn * vzdst;

      // eqn (61) of M&K
      Real bxsdldr = bxsig*sqrtdl*sqrtdr;
      Real bydst = invsumd*(sqrtdl*urstby + sqrtdr*ulstby +
                   bxsdldr*((urstmy*urst_d_inv) - (ulstmy*ulst_d_inv)));
      uldstby = urdstby = bydst;

      // eqn (62) of M&K
      Real bzdst = invsumd*(sqrtdl*urstbz + sqrtdr*ulstbz +
                   bxsdldr*((urstmz*urst_d_inv) - (ulstmz*ulst_d_inv)));
      uldstbz = urdstbz = bzdst;

      // eqn (63) of M&K
      Real tmp = spd2*bxi + vydst*bydst + vzdst*bzdst;
      uldsten = ulsten - sqrtdl*bxsig*(vbstl - tmp);
      urdsten = ursten + sqrtdr*bxsig*(vbstr - tmp);
    }

    //--- Step 6.  Compute flux
    Real flxdn, flxvx, flxvy, flxvz, flxen, flxby, flxbz;

    if (spd0 >= 0.0) {
      // return Fl if flow is supersonic
      flxdn = fldn;
      flxvx = flmx;
      flxvy = flmy;
      flxvz = flmz;
      flxen = flen;
      flxby = flby;
      flxbz = flbz;
    } else if (spd4 <= 0.0) {
      // return Fr if flow is supersonic
      flxdn = frdn;
      flxvx = frmx;
      flxvy = frmy;
      flxvz = frmz;
      flxen = fren;
      flxby = frby;
      flxbz = frbz;
    } else if (spd1 >= 0.0) {
      // return Fl*
      flxdn = fldn + spd0 * (ulstdn - uldn);
      flxvx = flmx + spd0 * (ulstmx - ulmx);
      flxvy = flmy + spd0 * (ulstmy - ulmy);
      flxvz = flmz + spd0 * (ulstmz - ulmz);
      flxen = flen + spd0 * (ulsten - ulen);
      flxby = flby + spd0 * (ulstby - ulby);
      flxbz = flbz + spd0 * (ulstbz - ulbz);
    } else if (spd2 >= 0.0) {
      // retuln Fl**
      flxdn = fldn + spd0 * (ulstdn - uldn) + spd1 * (uldstdn - ulstdn);
      flxvx = flmx + spd0 * (ulstmx - ulmx) + spd1 * (uldstmx - ulstmx);
      flxvy = flmy + spd0 * (ulstmy - ulmy) + spd1 * (uldstmy - ulstmy);
      flxvz = flmz + spd0 * (ulstmz - ulmz) + spd1 * (uldstmz - ulstmz);
      flxen = flen + spd0 * (ulsten - ulen) + spd1 * (uldsten - ulsten);
      flxby = flby + spd0 * (ulstby - ulby) + spd1 * (uldstby - ulstby);
      flxbz = flbz + spd0 * (ulstbz - ulbz) + spd1 * (uldstbz - ulstbz);
    } else if (spd3 > 0.0) {
      // return Fr**
      flxdn = frdn + spd4 * (urstdn - urdn) + spd3 * (urdstdn - urstdn);
      flxvx = frmx + spd4 * (urstmx - urmx) + spd3 * (urdstmx - urstmx);
      flxvy = frmy + spd4 * (urstmy - urmy) + spd3 * (urdstmy - urstmy);
      flxvz = frmz + spd4 * (urstmz - urmz) + spd3 * (urdstmz - urstmz);
      flxen = fren + spd4 * (ursten - uren) + spd3 * (urdsten - ursten);
      flxby = frby + spd4 * (urstby - urby) + spd3 * (urdstby - urstby);
      flxbz = frbz + spd4 * (urstbz - urbz) + spd3 * (urdstbz - urstbz);
    } else {
      // return Fr*
      flxdn = frdn + spd4 * (urstdn - urdn);
      flxvx = frmx + spd4 * (urstmx - urmx);
      flxvy = frmy + spd4 * (urstmy - urmy);
      flxvz = frmz + spd4 * (urstmz - urmz);
      flxen = fren + spd4 * (ursten - uren);
      flxby = frby + spd4 * (urstby - urby);
      flxbz = frbz + spd4 * (urstbz - urbz);
    }

    fdn[i] = flxdn;
    fvx[i] = flxvx;
    fvy[i] = flxvy;
    fvz[i] = flxvz;
    fen[i] = flxen;
    eya[i] = -flxby;
    eza[i] =  flxbz;

    Real v_over_c = (1024.0)* dt * flxdn / (dxw(i) * (wldn + wrdn));
    Real tmp_min = std::min(static_cast<Real>(0.5), v_over_c);
    wcta[i] = 0.5 + std::max(static_cast<Real>(-0.5), tmp_min);
  }

#pragma clang loop vectorize(assume_safety)
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
