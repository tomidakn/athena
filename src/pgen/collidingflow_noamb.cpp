//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file blast.cpp
//  \brief Problem generator for spherical blast wave problem.  Works in Cartesian,
//         cylindrical, and spherical coordinates.  Contains post-processing code
//         to check whether blast is spherical for regression tests
//
// REFERENCE: P. Londrillo & L. Del Zanna, "High-order upwind schemes for
//   multidimensional MHD", ApJ, 530, 508 (2000), and references therein.

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <cstring>    // strcmp()
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

Real threshold;
Real Lunit; // unit of length 
Real tunit; // unit of time 
Real vunit; // unit of velocity 
Real rhounit; // unit of density 
Real Munit;  // unit of mass
Real Eunit; // unit of energy 
Real eunit; // unit of energy density 
Real Tunit; // unit of temperature 
Real coolunit; // unit of coolingrate
Real denthr;
Real G0;

int ntab = 64;
AthenaArray<Real> logLam, logT0;
Real logT0_min, logT0_max, dlogT0;

int ntabnum = 32;
AthenaArray<Real> shield, mol, logn0;
Real logn0_min, logn0_max, dlogn0;

const Real Kcoeff = 6.458911372764623E-4;
const Real mucoeff = 3.875346823658774E-4;
Real CFL_para;

static const int iend = 128;
static Real dxbou, Lbou, vel, b0x, b0y, P0;
static AthenaArray<Real> primbou0;
static AthenaArray<Real> xbou,ybou,zbou;

//void CoolingFunc(MeshBlock *pmb, const Real time, const Real dt,
//              const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
//              AthenaArray<Real> &cons);
void CoolingFunc(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar);

void ConductionCoeff(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);

void ViscosityCoeff(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);

Real CoolingCondTimeStep(MeshBlock *pmb);

Real net_cooling(Real num, Real tem);
Real EquilibriumTemperature(Real num);
void DefineSimulationUnits();
Real molecular_weight(Real num);

Real History_maxden(MeshBlock *pmb, int iout);

void TwoPhaseInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void TwoPhaseOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin){

        return;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
//  if (adaptive) {
//    EnrollUserRefinementCondition(RefinementCondition);
//    threshold = pin->GetReal("problem","thr");
//  }

    Real CJ = pin->GetReal("problem","CJ");
    Real dx = (mesh_size.x1max - mesh_size.x1min)/mesh_size.nx1;
    denthr = 1067.24340529533/SQR(CJ*dx); // above whith cooling/heating is switched off
    G0 = pin->GetReal("problem","G0");

    AllocateUserHistoryOutput(1);
    EnrollUserHistoryOutput(0, History_maxden, "maxden",UserHistoryOperation::max);

  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, TwoPhaseInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, TwoPhaseOuterX1);
  }

// Enrol user-dfine source function
if( pin->GetInteger("problem","switch_cooling") == 1 ){
    EnrollUserExplicitSourceFunction(CoolingFunc);
    EnrollUserTimeStepFunction(CoolingCondTimeStep);
}

// Enrol user-dfine conduction term 
if( pin->GetOrAddReal("problem","kappa_iso",0.0) == 0.0 ){
    EnrollConductionCoefficient(ConductionCoeff);
}
if( pin->GetOrAddReal("problem","nu_iso",0.0) == 0.0 ){
    EnrollViscosityCoefficient(ViscosityCoeff);
}

  DefineSimulationUnits();

  // save the initial condition
  primbou0.NewAthenaArray(iend+1,iend+1,iend+1); // IDN = 0, IVX = 1, IVY = 2, IVZ = 3, IPR = 4
  xbou.NewAthenaArray(iend+1);           // x coordinate
  ybou.NewAthenaArray(iend+1);           // y coordinate
  zbou.NewAthenaArray(iend+1);           // z coordinate

  Lbou = mesh_size.x2max - mesh_size.x2min;
  dxbou = Lbou/(Real) iend;
  xbou(0) = 0.0;
  for( int i=1; i<iend+1; i++ )
          xbou(i) = xbou(i-1) + dxbou;
  ybou(0) = - 0.5*Lbou;
  for( int i=1; i<iend+1; i++ )
          ybou(i) = ybou(i-1) + dxbou;
  zbou(0) = - 0.5*Lbou;
  for( int i=1; i<iend+1; i++ )
          zbou(i) = zbou(i-1) + dxbou;

  Real n0 = pin->GetReal("problem", "n0");
  vel = pin->GetReal("problem","v0");
  if (MAGNETIC_FIELDS_ENABLED) {
    Real b0 = pin->GetOrAddReal("problem", "b0",0.0);
    Real sinth = pin->GetOrAddReal("problem", "sinth",0.0);
    b0x = b0*sqrt(1.0 - sinth*sinth);
    b0y = b0*sinth;
  }

  FILE* fp = fopen("initial_den3d.dat", "rb");
  fread(primbou0.data(), sizeof(Real), ((long int) iend+1)*(iend+1)*(iend+1), fp);
  fclose(fp);

  for( int k=0; k<iend+1; k++) {
  for( int j=0; j<iend+1; j++) {
#pragma omp simd
  for( int i=0; i<iend+1; i++) {
          primbou0(k,j,i) *= n0*1.4;
  }}}

  if(SELF_GRAVITY_ENABLED) {
    SetGravitationalConstant(1.11142969912701e-4);
    Real eps = pin->GetOrAddReal("problem","grav_eps", 0.0);
//    SetMeanDensity(0.0);
    SetGravityThreshold(eps);
  }

  CFL_para = (1.0/6.0)*pin->GetReal("time","cfl_number_para")*pin->GetReal("time","sts_max_dt_ratio");

    logT0.NewAthenaArray(ntab);
    logLam.NewAthenaArray(ntab);
    FILE *fin = fopen("coolingfunc_KI02.dat","r");
    for(int i=0; i<ntab; i++){
        fscanf(fin,"%lf%lf",&logT0(i), &logLam(i));
    }
    fclose(fin);
    logT0_min = logT0(0);
    logT0_max = logT0(ntab-2);
    dlogT0 = (logT0(ntab-1) - logT0_min)/(Real) (ntab-1);

    logn0.NewAthenaArray(ntabnum);
    shield.NewAthenaArray(ntabnum);
    mol.NewAthenaArray(ntabnum);
    FILE *finn = fopen("shielding.dat","r");
    for(int i=0; i<ntabnum; i++){
        fscanf(finn,"%lf%lf%lf",&logn0(i), &mol(i), &shield(i));
    }
    fclose(finn);
    logn0_min = logn0(0);
    logn0_max = logn0(ntabnum-2);
    dlogn0 = (logn0(ntabnum-1) - logn0_min)/(Real) (ntabnum-1);

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Spherical blast wave test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  Real n0 = pin->GetReal("problem", "n0");
  Real v0 = pin->GetReal("problem", "v0");
  Real den;

  Real rho0 = 1.4*n0;
  Real dx = pcoord->dx1v(0);

  // setup uniform ambient medium with spherical over-pressured region
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
          Real xx  = pcoord->x1v(i);
          Real yy  = pcoord->x2v(j);
          Real zz  = pcoord->x3v(k);

          while( xx < xbou(0)    )  xx += Lbou;
          while( xx > xbou(iend) ) xx -= Lbou;

          int ii = (xx - xbou(0))/dxbou;
          int jj = (yy - ybou(0))/dxbou;
          int kk = (zz - zbou(0))/dxbou;

          int iip = ii + 1; if( iip == iend ) iip = 0;
          int jjp = jj + 1; if( jjp == iend ) jjp = 0;
          int kkp = kk + 1; if( kkp == iend ) kkp = 0;

          Real fdx = ( xx - xbou(ii) )/dxbou;
          Real fdy = ( yy - ybou(jj) )/dxbou;
          Real fdz = ( zz - zbou(kk) )/dxbou;

          Real fdxm1 = 1.0 - fdx;
          Real fdym1 = 1.0 - fdy;
          Real fdzm1 = 1.0 - fdz;

          den = fdzm1*fdym1*fdxm1*primbou0(kk ,jj ,ii ) 
              + fdzm1*fdy  *fdxm1*primbou0(kk ,jjp,ii )
              + fdzm1*fdym1*fdx  *primbou0(kk ,jj ,iip)
              + fdzm1*fdy  *fdx  *primbou0(kk ,jjp,iip)
              + fdz  *fdym1*fdxm1*primbou0(kkp,jj ,ii ) 
              + fdz  *fdy  *fdxm1*primbou0(kkp,jjp,ii )
              + fdz  *fdym1*fdx  *primbou0(kkp,jj ,iip)
              + fdz  *fdy  *fdx  *primbou0(kkp,jjp,iip);

         phydro->u(IDN,k,j,i) = den;

        if( pcoord->x1v(i) < 0 ) {
            phydro->u(IM1,k,j,i) = v0; 
            phydro->u(IM2,k,j,i) = 0.0; 
            phydro->u(IM3,k,j,i) = 0.0; 
        } else  {
            phydro->u(IM1,k,j,i) = -v0; 
            phydro->u(IM2,k,j,i) = 0.0; 
            phydro->u(IM3,k,j,i) = 0.0; 
        }

        phydro->u(IM1,k,j,i) *= den;
        phydro->u(IM2,k,j,i) *= den;
        phydro->u(IM3,k,j,i) *= den;
      }
    }
  }

  Real gamma = peos->GetGamma();
  Real gm1 = gamma - 1.0;
  if( Globals::my_rank == 0 ){
        printf("n0= %.5e\n",n0);
  }
  Real temeq = EquilibriumTemperature(n0)/Tunit;
  P0 = rho0*temeq/(0.53*(tanh( (log10(n0) - 1.3)/0.9 ) + 1.0) + 1.27);
  if( Globals::my_rank == 0 ){
        printf("T0 = %.5e\n",temeq*Tunit);
  }

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real rad;
        Real x = pcoord->x1v(i);
        Real y = pcoord->x2v(j);
        Real z = pcoord->x3v(k);
        rad = std::sqrt(SQR(x) + SQR(y) + SQR(z));

        phydro->u(IEN,k,j,i) = P0/gm1
                              + 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
                                   + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
      }
    }
  }

  // initialize interface B and total energy
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie+1; ++i) {
            pfield->b.x1f(k,j,i) = b0x;
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
        for (int i=is; i<=ie; ++i) {
            pfield->b.x2f(k,j,i) = b0y;
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
            pfield->b.x3f(k,j,i) = 0.0;
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          phydro->u(IEN,k,j,i) += 0.5*(SQR(b0x) +  SQR(b0y));
        }
      }
    }
  }

}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief Check radius of sphere to make sure it is round
//========================================================================================
void CoolingFunc(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar){

  Real gm1 = pmb->peos->GetGamma() - 1.0;

//  Real dEdt;
  Real dt_cool = 1e100;

  int ks = pmb->ks;
  int ke = pmb->ke;
  int js = pmb->js;
  int je = pmb->je;
  int is = pmb->is;
  int ie = pmb->ie;

/*
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
  for (int j=pmb->js; j<=pmb->je; ++j) {
//#pragma omp simd
#pragma clang loop vectorize(assume_safety)
  for (int i=pmb->is; i<=pmb->ie; ++i) {
*/
  for (int k=ks; k<=ke; ++k) {
  for (int j=js; j<=je; ++j) {
#pragma clang loop vectorize(assume_safety)
  for (int i=is; i<=ie; ++i) {
     if( prim(IDN,k,j,i) > denthr) continue;   
    
       Real numH = prim(IDN,k,j,i)/1.4;
       Real ln=std::max((std::min(log10(numH),logn0_max)-logn0_min)/dlogn0,0.0);
       int in=(int)ln;
       Real wn=ln-in;
       Real mu = mol(in)*(1.0 - wn) + mol(in+1)*wn;
       Real tem0 = mu*Tunit*prim(IPR,k,j,i)/prim(IDN,k,j,i);

       Real lt=std::max((std::min(log10(tem0),logT0_max)-logT0_min)/dlogT0,0.0);
       int it=(int)lt;
       Real wt=lt-it;

       Real dEdt = numH*( G0*2.0e-26*( shield(in)*(1.0 - wn) + shield(in+1)*wn ) + 3.2e-11*5.e-17 
                  - numH*exp(2.302585092994046*(logLam(it)*(1.0 - wt) + logLam(it+1)*wt)) )*1.973263510396663E+27;
       cons(IEN,k,j,i) += dEdt*dt;

       Real pb = 0.5*(SQR(bcc(IB1,k,j,i)) + SQR(bcc(IB2,k,j,i)) + SQR(bcc(IB3,k,j,i)));
       Real e_k = 0.5/cons(IDN,k,j,i)*(SQR(cons(IVX,k,j,i)) + SQR(cons(IVY,k,j,i)) + SQR(cons(IVZ,k,j,i)));
       Real temnew = mu*Tunit*gm1*(cons(IEN,k,j,i) - pb - e_k)/prim(IDN,k,j,i);

       if( temnew > 10.0 ) {
             dt_cool = std::min( dt_cool, 0.5*prim(IPR,k,j,i)/fabs(dEdt) );
       }
  }}}
  pmb->dt_cool = dt_cool;

}
//----------------------------------------------------------------------
//----------------------------------------------------------------------

Real net_cooling(Real numH, Real tem){
Real cooling, Av;

 cooling = 2.0e-26*( 1.0e7*exp( - 1.148e5/( tem + 1.0e3 ) )  
                   + 1.4e-2*sqrt(tem)*exp( - 9.2e1/tem) );

  Av = 0.05*exp(1.6*pow(numH,0.12));
  return numH*( 2.8e-26*exp(-1.8*Av)*0.5*( 1.0 + SIGN(8000.0 - tem) )  // photo-ele
                      + 3.2e-11*5.e-17  // cosmic-ray
                      - numH*cooling );
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------
Real EquilibriumTemperature(Real num){

  Real Tmin, Tcen, Tmax; 
  Real net_coolmin, net_coolcen, net_coolmax;

  Tmin = 5.0;
  Tmax = 2e4;

  if( net_cooling(num,Tmin)*net_cooling(num,Tmax)>0 ) {
      printf("something wrong in EquilibriumTemperature\n");
  }

  int itry = 0;
  do{
      Tcen = 0.5*(Tmax + Tmin);
      net_coolmin = net_cooling(num,Tmin);
      net_coolcen = net_cooling(num,Tcen);
      net_coolmax = net_cooling(num,Tmax);

      if( net_coolmin*net_coolcen < 0 ) {
          Tmax = Tcen;
      } else {
          Tmin = Tcen;
      }

      itry++;
  } while ( fabs(0.5*(Tmin+Tmax)/Tcen - 1.0) > 1e-10 );
   

  return Tcen;
}
//======================================================================
//======================================================================
Real CoolingCondTimeStep(MeshBlock *pmb)
{
  return pmb->dt_cool*pmb->pmy_mesh->cfl_number;
}
//======================================================================
Real History_maxden(MeshBlock *pmb, int iout) {
  Real momr=0;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
//  AthenaArray<Real> volume; // 1D array of volumes
//  volume.NewAthenaArray(pmb->ncells1);

  Real denmax = 0.0;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
          denmax = std::max(denmax, pmb->phydro->w(IDN,k,j,i));
      }
    }
  }

  return denmax;
}
//----------------------------------------------------------------------------------------
void ConductionCoeff(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke) {

    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i){
            Real mu = molecular_weight(prim(IDN,k,j,i)/1.4);
            phdif->kappa(HydroDiffusion::DiffProcess::iso,k,j,i) = 
                ( Kcoeff*mu*sqrt(mu)*sqrt(prim(IPR,k,j,i)/prim(IDN,k,j,i)) )/prim(IDN,k,j,i);
        }
      }
    }

  return;
}
//----------------------------------------------------------------------------------------
void ViscosityCoeff(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke) {

    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i){
            Real mu = molecular_weight(prim(IDN,k,j,i)/1.4);
            phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = 
                ( mucoeff*mu*sqrt(mu)*sqrt(prim(IPR,k,j,i)/prim(IDN,k,j,i)) )/prim(IDN,k,j,i);
        }
      }
    }

  return;
}
//----------------------------------------------------------------------------------------
void DefineSimulationUnits(){
    Lunit = 3.08567758E+18; // pc
    tunit = 3.155693E+13;  // Myr
    vunit = Lunit/tunit;
    rhounit = 1.672623E-24; // mH
    Munit = rhounit*Lunit*Lunit*Lunit; 
    Eunit = Munit*SQR(vunit);
    eunit = rhounit*SQR(vunit);
    Tunit = 115.8306644934343;
    coolunit = eunit/tunit;
}
//================================================================================
//----------------------------------------------------------------------------------------
//! \fn void TwoPhaseInnerX1()
//  \brief Sets boundary condition on left X boundary (iib) for jet problem

void TwoPhaseInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  // set primitive variables in inlet ghost zones
  int ii, jj, kk, iip, jjp, kkp;
  Real fdx, fdy, fdz, fdxm1, fdym1, fdzm1, xtmp, xx, veltime;

  veltime = vel*time - Lbou*( (int) (vel*time/Lbou) );
  
  for(int k=ks; k<=ke; ++k){
    Real zz  = pco->x3v(k);
    if ( zz < zbou(0)    ) zz += Lbou;
    if ( zz > zbou(iend) ) zz -= Lbou;
    kk = (int) ((zz - zbou(0))/dxbou );
    kkp = kk+1; if(kkp == iend) kkp = 0;
  for(int j=js; j<=je; ++j){
    Real yy  = pco->x2v(j);
    if( yy < ybou(0)    ) yy += Lbou;
    if( yy > ybou(iend) ) yy -= Lbou;
    jj = (int) ((yy - ybou(0))/dxbou );
    jjp = jj+1; if(jjp == iend) jjp = 0;
#pragma omp simd
    for(int i=1; i<=ngh; ++i){ 
        xx = pco->x1v(is-i) - veltime;
        while( xx < xbou(0)    ) xx += Lbou;
        while( xx > xbou(iend) ) xx -= Lbou;

        ii = (int) ((xx - xbou(0))/dxbou);
        iip = ii+1; if( iip == iend ) iip = 0;

        fdx = ( xx - xbou(ii) )/dxbou;
        fdy = ( yy - ybou(jj) )/dxbou;
        fdz = ( zz - zbou(kk) )/dxbou;

        fdxm1 = 1.0 - fdx;
        fdym1 = 1.0 - fdy;
        fdzm1 = 1.0 - fdz;


          prim(IDN,k,j,is-i) = fdzm1*fdym1*fdxm1*primbou0(kk ,jj ,ii ) 
                             + fdzm1*fdy  *fdxm1*primbou0(kk ,jjp,ii )
                             + fdzm1*fdym1*fdx  *primbou0(kk ,jj ,iip)
                             + fdzm1*fdy  *fdx  *primbou0(kk ,jjp,iip)
                             + fdz  *fdym1*fdxm1*primbou0(kkp,jj ,ii ) 
                             + fdz  *fdy  *fdxm1*primbou0(kkp,jjp,ii )
                             + fdz  *fdym1*fdx  *primbou0(kkp,jj ,iip)
                             + fdz  *fdy  *fdx  *primbou0(kkp,jjp,iip);
          prim(IVX,k,j,is-i) = vel;
          prim(IVY,k,j,is-i) = 0.0;
          prim(IVZ,k,j,is-i) = 0.0;
          prim(IPR,k,j,is-i) = P0;
    }
  }}

  // set magnetic field in inlet ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for(int k=ks; k<=ke; ++k){
    for(int j=js; j<=je; ++j){
#pragma omp simd
      for(int i=1; i<=ngh; ++i){
          b.x1f(k,j,is-i) = b0x;
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
          b.x2f(k,j,is-i) = b0y;
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
          b.x3f(k,j,is-i) = 0.0;
      }
    }}
  }

}
//----------------------------------------------------------------------------------------
//! \fn void TwoPhaseouterX1()
//  \brief Sets boundary condition on left X boundary (iib) for jet problem

void TwoPhaseOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  int ii, jj, kk, iip, jjp, kkp;
  Real fdx, fdy, fdz, fdxm1, fdym1, fdzm1, xtmp, xx, veltime;

  veltime = vel*time;
  veltime = veltime - Lbou*( (int) (veltime/Lbou) );
  
  for(int k=ks; k<=ke; ++k){
    Real zz  = pco->x3v(k);
    if( zz < zbou(0)    ) zz += Lbou;
    if( zz > zbou(iend) ) zz -= Lbou;
    kk = (int) ((zz - zbou(0))/dxbou );
    kkp = kk+1; if(kkp == iend) kkp = 0;
  for(int j=js; j<=je; ++j){
    Real yy  = pco->x2v(j);
    if( yy < ybou(0)    ) yy += Lbou;
    if( yy > ybou(iend) ) yy -= Lbou;
    jj = (int) ((yy - ybou(0))/dxbou );
    jjp = jj+1; if(jjp == iend) jjp = 0;
#pragma omp simd
    for(int i=1; i<=ngh; ++i){ 
            xx = pco->x1v(ie+i) + veltime;
            while( xx < xbou(0)    ) xx += Lbou;
            while( xx > xbou(iend) ) xx -= Lbou;

            ii = (int) ((xx - xbou(0))/dxbou);

            fdx = ( xx - xbou(ii) )/dxbou;
            fdy = ( yy - ybou(jj) )/dxbou;
            fdz = ( zz - zbou(kk) )/dxbou;

            fdxm1 = 1.0 - fdx;
            fdym1 = 1.0 - fdy;
            fdzm1 = 1.0 - fdz;

            iip = ii+1; if( iip == iend ) iip = 0;

          prim(IDN,k,j,ie+i) = fdzm1*fdym1*fdxm1*primbou0(kk ,jj ,ii ) 
                             + fdzm1*fdy  *fdxm1*primbou0(kk ,jjp,ii )
                             + fdzm1*fdym1*fdx  *primbou0(kk ,jj ,iip)
                             + fdzm1*fdy  *fdx  *primbou0(kk ,jjp,iip)
                             + fdz  *fdym1*fdxm1*primbou0(kkp,jj ,ii ) 
                             + fdz  *fdy  *fdxm1*primbou0(kkp,jjp,ii )
                             + fdz  *fdym1*fdx  *primbou0(kkp,jj ,iip)
                             + fdz  *fdy  *fdx  *primbou0(kkp,jjp,iip);
          prim(IVX,k,j,ie+i) = -vel;
          prim(IVY,k,j,ie+i) = 0.0;
          prim(IVZ,k,j,ie+i) = 0.0;
          prim(IPR,k,j,ie+i) = P0;
    }
  }}

  // set magnetic field in inlet ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for(int k=ks; k<=ke; ++k){
    for(int j=js; j<=je; ++j){
#pragma simd
      for(int i=1; i<=ngh; ++i){
          b.x1f(k,j,ie+i+1) = b0x;
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma simd
      for (int i=1; i<=ngh; ++i) {
          b.x2f(k,j,ie+i) = b0y;
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma simd
      for (int i=1; i<=ngh; ++i) {
          b.x3f(k,j,ie+i) = 0.0;
      }
    }}
  }


}

Real molecular_weight(Real num){
    return 0.53*tanh( (log10(num) - 1.3)/0.9 ) + 1.8;
}
