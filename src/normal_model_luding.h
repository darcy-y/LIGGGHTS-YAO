/* ----------------------------------------------------------------------
   LIGGGHTS - LAMMPS Improved for General Granular and Granular Heat
   Transfer Simulations

   LIGGGHTS is part of the CFDEMproject
   www.liggghts.com | www.cfdem.com

   Christoph Kloss, christoph.kloss@cfdem.com
   Copyright 2009-2012 JKU Linz
   Copyright 2012-     DCS Computing GmbH, Linz

   LIGGGHTS is based on LAMMPS
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   This software is distributed under the GNU General Public License.

   See the README file in the top-level directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors:
   Christoph Kloss (JKU Linz, DCS Computing GmbH, Linz)
   Richard Berger (JKU Linz)
------------------------------------------------------------------------- */
#ifdef NORMAL_MODEL
NORMAL_MODEL(LUDING,luding,10)
#else
#ifndef NORMAL_MODEL_LUDING_H_
#define NORMAL_MODEL_LUDING_H_
#include "contact_models.h"
#include <math.h>
#include "atom.h"
#include "force.h"
#include "update.h"
#include "global_properties.h"

namespace LIGGGHTS {
namespace ContactModels
{
  template<>
  class NormalModel<LUDING> : protected Pointers
  {
  public:
    static const int MASK = CM_REGISTER_SETTINGS | CM_CONNECT_TO_PROPERTIES | CM_COLLISION | CM_NO_COLLISION;

    NormalModel(LAMMPS * lmp, IContactHistorySetup * hsetup) : Pointers(lmp),
        k1(NULL),
        k2Max(NULL),
        kc(NULL),
        F0(NULL),
        phiF(NULL),
        coeffRestLog(NULL),
        ktToKn(1.0),
        tangential_damping(false),
        fixedK2(false)
    {
      deltaMax_offset = hsetup->add_history_value("deltaMax", "0");
      deltaP_offset   = hsetup->add_history_value("deltaP", "0");
      // history_offset = hsetup->add_history_value("deltaMax", "0");
      /*NL*/ if(comm->me == 0 && screen) fprintf(screen, "LUDING loaded\n");
    }

    inline void registerSettings(Settings & settings)
    {
        settings.registerOnOff("tangential_damping", tangential_damping, true);
        settings.registerOnOff("fixedK2", fixedK2);
        settings.registerDoubleSetting("ktToKn", ktToKn, 1.0);
    }

    inline void connectToProperties(PropertyRegistry & registry) {
      // NormalModel<HOOKE>::connectToProperties(registry);

      // New Luding parameters: direct input
      registry.registerProperty("k1",    &MODEL_PARAMS::createLudingK1);
      registry.registerProperty("k2Max", &MODEL_PARAMS::createLudingK2Max);
      registry.registerProperty("kc",    &MODEL_PARAMS::createLudingKc);
      registry.registerProperty("F0",    &MODEL_PARAMS::createLudingF0);

      registry.connect("k1",    k1,    "model luding");
      registry.connect("k2Max", k2Max, "model luding");
      registry.connect("kc",    kc,    "model luding");
      registry.connect("F0",    F0,    "model luding");

      // phiF is only needed when k2 varies
      if(!fixedK2)
      {
          registry.registerProperty("phiF", &MODEL_PARAMS::createLudingPhiF);
          registry.connect("phiF", phiF, "model luding");
      }

      // damping properties
      registry.registerProperty("coeffRestLog", &MODEL_PARAMS::createCoeffRestLog);
      registry.connect("coeffRestLog", coeffRestLog, "model luding");
      

      // Old parameters temporarily retained because the force formula
      // still uses them at this stage
      // registry.registerProperty("kn2kcMax", &MODEL_PARAMS::createCoeffMaxElasticStiffness);
      // registry.registerProperty("kn2kc",    &MODEL_PARAMS::createCoeffAdhesionStiffness);

      // registry.connect("kn2kcMax", kn2k2Max, "model luding");
      // registry.connect("kn2kc",    kn2kc,    "model luding");

      //NP modified C.K.
      // error checks on coarsegraining
      // if(force->cg_active())
      //   error->cg(FLERR,"model luding");
    }

    // effective exponent for stress-strain relationship
    //NP used for area correction of heat transfer
    inline double stressStrainExponent()
    {
      return 1.;
    }

    inline void collision(CollisionData & cdata, ForceData & i_forces, ForceData & j_forces)
    {

      const int itype = cdata.itype;
      const int jtype = cdata.jtype;
      const double deltan = cdata.deltan;
      double ri = cdata.radi;
      double rj = cdata.radj;
      double reff=cdata.is_wall ? cdata.radi : (ri*rj/(ri+rj));

#ifdef SUPERQUADRIC_ACTIVE_FLAG
      if(cdata.is_non_spherical && atom->superquadric_flag)
        reff = cdata.reff;
#endif

      double meff = cdata.meff;
      // double coeffRestLogChosen;

      // if (viscous)  {
      //   double ** & coeffMu = NormalModel<HOOKE>::coeffMu;
      //   double ** & coeffRestMax = NormalModel<HOOKE>::coeffRestMax;
      //   double ** & coeffStc = NormalModel<HOOKE>::coeffStc;
      //   // Stokes Number from MW Schmeeckle (2001)
      //   const double stokes=cdata.meff*cdata.vn/(6.0*M_PI*coeffMu[itype][jtype]*reff*reff);
      //   // Empirical from Legendre (2006)
      //   coeffRestLogChosen=log(coeffRestMax[itype][jtype])+coeffStc[itype][jtype]/stokes;
      // } else {
      //   double ** & coeffRestLog = NormalModel<HOOKE>::coeffRestLog;
      //   coeffRestLogChosen=coeffRestLog[itype][jtype];
      // }

      // Luding parameters for current type pair
      const double k1_now = k1[itype][jtype];
      const double k2Max_now = k2Max[itype][jtype];
      const double kc_now = kc[itype][jtype];
      const double F0_now = F0[itype][jtype];

      // --------------------------------------------------------
      // Contact history
      //
      // deltaMax = true historical maximum overlap
      // deltaP   = current plastic overlap of k2 line
      // --------------------------------------------------------
      if(cdata.touch) *cdata.touch |= TOUCH_NORMAL_MODEL;
      double & deltaMax = cdata.contact_history[deltaMax_offset];
      double & deltaP = cdata.contact_history[deltaP_offset];

      // --------------------------------------------------------
      // Update TRUE maximum overlap
      // --------------------------------------------------------

      const bool newMaximum =
          (deltan > deltaMax);

      if(newMaximum)
          deltaMax = deltan;

      // --------------------------------------------------------
      // Calculate current k2
      // --------------------------------------------------------
      double k2_now;

      if(fixedK2) {

          // constant-k2 model
          k2_now = k2Max_now;

      } else {

          // history-dependent Luding hardening

          const double phiF_now = phiF[itype][jtype];

          const double deltaMaxLim =
              k2Max_now /
              (k2Max_now-k1_now) *
              phiF_now *
              2.0*reff;
          if(deltaMax >= deltaMaxLim) {
              k2_now = k2Max_now;
          } else {
              k2_now =
                  k1_now +
                  (k2Max_now-k1_now) *
                  deltaMax /
                  deltaMaxLim;
          }
      }

      // --------------------------------------------------------
      // New maximum overlap:
      //
      // make unloading line pass through the new maximum point
      //
      // deltaP = (1-k1/k2)*deltaMax
      // --------------------------------------------------------

      if(newMaximum) {
          deltaP =
              (1.0-k1_now/k2_now) *
              deltaMax;
      }

      // --------------------------------------------------------
      // Three Luding branches WITHOUT F0 offset
      // --------------------------------------------------------

      const double fLoad =
          k1_now*deltan;

      double fUnload =
          k2_now*(deltan-deltaP);

      const double fAdhesion =
          -kc_now*deltan;

      // --------------------------------------------------------
      // Reloading directly from adhesion branch
      //
      // vn < 0  -> overlap is increasing
      //
      // If old k2 line lies below adhesion line, the contact
      // is reloading from the adhesion branch.
      //
      // Construct a new k2 line through the current point:
      //
      // k2*(delta-deltaPnew) = -kc*delta
      //
      // deltaPnew = delta*(1+kc/k2)
      // --------------------------------------------------------

      bool reloadFromAdhesion = false;

      if(!newMaximum &&
          cdata.vn < 0.0 &&
          fUnload <= fAdhesion)
      {
          deltaP =
              deltan *
              (1.0 + kc_now/k2_now);

          fUnload =
              k2_now*(deltan-deltaP);

          reloadFromAdhesion = true;
      }

      // --------------------------------------------------------
      // Select current normal branch
      // --------------------------------------------------------

      double fBase;
      double kn_current;

      if(newMaximum) {

          // A new true maximum overlap is always virgin loading.
          // Do not determine this branch from fUnload == fLoad,
          // because round-off can cause branch switching.
          fBase = fLoad;
          kn_current = k1_now;

      } else if(fUnload >= fLoad) {

          // Reloading has reached the loading envelope.
          fBase = fLoad;
          kn_current = k1_now;

      } else if(fUnload > fAdhesion ||
                reloadFromAdhesion) {

          // unloading / reloading branch
          fBase = fUnload;
          kn_current = k2_now;

      } else {

          // adhesion branch
          fBase = fAdhesion;
          kn_current = kc_now;
      }

      // --------------------------------------------------------
      // F0 offset
      //
      // F0 is INPUT as positive magnitude:
      //
      // loading:
      // FL = k1*delta - F0
      //
      // unloading:
      // FL = k2*(delta-deltaP) - F0
      //
      // adhesion:
      // FL = -kc*delta - F0
      // --------------------------------------------------------
      const double FL = fBase - F0_now;


      // --------------------------------------------------------
      // Friction / rolling reference force
      //
      // Signed-Coetzee form:
      // Fref = FL - F0_phys + kc*delta
      //
      // Since F0_phys = -F0_input:
      //
      // Fref = FL + F0_input + kc*delta
      //
      // On adhesion branch:
      //
      // FL = -F0-kc*delta
      // -> Fref = 0
      //
      // Normal dashpot is NOT included.
      // --------------------------------------------------------

      double Fref =
          FL +
          F0_now +
          kc_now*deltan;

      if(Fref < 0.0)
          Fref = 0.0;


      // --------------------------------------------------------
      // Tangential stiffness
      //
      // Always based on k1.
      // Does NOT follow normal branch.
      //
      // kt = ktToKn*k1
      // --------------------------------------------------------
      const double kt = ktToKn*k1_now;

      // --------------------------------------------------------
      // Damping coefficient from coefficientRestitution
      //
      // coeffRestLog = ln(e)
      // --------------------------------------------------------
      const double coeffRestLogChosen =
          coeffRestLog[itype][jtype];

      const double coeffRestLogChosenSq =
          coeffRestLogChosen *
          coeffRestLogChosen;


      // --------------------------------------------------------
      // Normal damping:
      //
      // branch dependent
      //
      // loading   -> kn = k1
      // unloading -> kn = k2
      // adhesion  -> kn = kc
      // --------------------------------------------------------
      const double gamman =
          sqrt(
              4.0 *
              meff *
              kn_current *
              coeffRestLogChosenSq /
              (
                  coeffRestLogChosenSq +
                  M_PI*M_PI
              )
          );


      // --------------------------------------------------------
      // Tangential damping:
      //
      // always uses fixed tangential stiffness
      //
      // kt = ktToKn*k1
      // --------------------------------------------------------
      const double gammat =
          tangential_damping ?
          sqrt(
              4.0 *
              meff *
              kt *
              coeffRestLogChosenSq /
              (
                  coeffRestLogChosenSq +
                  M_PI*M_PI
              )
          ) :
          0.0;

      // --------------------------------------------------------
      // Actual normal force
      // --------------------------------------------------------

      const double Fn_damping =
          -gamman*cdata.vn;

      const double Fn =
          FL + Fn_damping;


      // --------------------------------------------------------
      // Pass parameters to tangential / rolling models
      //
      // IMPORTANT:
      //
      // cdata.Fn = Fref
      //
      // NOT actual applied normal force.
      //
      // Existing tangential model uses:
      //
      // Ft_max = mu*abs(cdata.Fn)
      //
      // so this automatically gives the adhesive Coulomb
      // reference force.
      // --------------------------------------------------------

      cdata.Fn = Fref;
      cdata.kn = kn_current;
      cdata.kt = kt;
      cdata.gamman = gamman;
      cdata.gammat = gammat;
      
////////////////////////////////////////////////////////////////////////////////
      // const double kn = k1_now;
      // const double kt = ktToKn*kn;
      // const double sqrtval = sqrt(reff);
      // const double k = (16./15.)*sqrtval*(Yeff[itype][jtype]);
      // double kn = k*pow(meff*charVel*charVel/k,0.2);
      // double kt = kn;
      // const double coeffRestLogChosen = coeffRestLog[itype][jtype];
      // const double coeffRestLogChosenSq = coeffRestLogChosen*coeffRestLogChosen;
      // const double gamman = sqrt(4.*meff*kn*coeffRestLogChosenSq/(coeffRestLogChosenSq+M_PI*M_PI));
      // const double gammat = tangential_damping ? gamman : 0.0;

      // convert Kn and Kt from pressure units to force/distance^2
      // kn /= force->nktv2p;
      // kt /= force->nktv2p;

      // coefficients
      /*NL*/ //if(screen) fprintf(screen,"For types %d %d\n",itype,jtype);
      /*NL*/ //if(screen) fprintf(screen,"kn2k2Max = %f and kn2kc = %f \n",kn2k2Max_[itype][jtype],kn2kc_[itype][jtype]);
      // const double k2Max = kn * kn2k2Max[itype][jtype]; //NP Test parameter
      // const double kc = kn * kn2kc[itype][jtype]; //NP Test parameter

      // get the history value -- maximal overlap
      // if(cdata.touch) *cdata.touch |= TOUCH_NORMAL_MODEL;
      // double * const history = &cdata.contact_history[history_offset];
      // double deltaMax; // the 4th value of the history array is deltaMax
      // if (deltan > history[0]) {
          // history[0] = deltan;
          // deltaMax = deltan;
      // } else
          // deltaMax = history[0];

      // k2 dependent on the maximum overlap
      // this accounts for an increasing stiffness with deformation
      // const double deltaMaxLim =(k2Max_now/(k2Max_now-kn))*phiF[itype][jtype]*2*reff;
      // double k2, fHys;
      // if (deltaMax >= deltaMaxLim) // big overlap ... no kn at all
      // {
      //     k2 = k2Max_now;
      //     const double fTmp = k2*(deltan-deltaMaxLim)+kn*deltaMaxLim;//k2*(deltan-delta0);
      //     if (fTmp >= -kc_now*deltan) { // un-/reloading part (k2)
      //         fHys = fTmp;
      //     } else { // cohesion part
      //         fHys = -kc_now*deltan;

      //         const double newDeltaMax = 0.5*(deltan+sqrt(deltan*deltan+4*((kn+kc_now)*deltan*deltaMaxLim/(k2Max_now-kn))));
      //         history[0] = newDeltaMax;
      //     }
      // } else {
      //     k2 = kn+(k2Max_now-kn)*deltaMax/deltaMaxLim;
      //     const double fTmp = k2*(deltan-deltaMax)+kn*deltaMax;//k2*(deltan-delta0);
      //     if (fTmp >= kn*deltan) { // loading part (kn)
      //         fHys = kn*deltan;
      //     } else { // un-/reloading part (k2)
      //         if (fTmp > -kc_now*deltan) {
      //             fHys = fTmp;
      //         } else { // cohesion part
      //             fHys = -kc_now*deltan;

      //             const double newDeltaMax = 0.5*(deltan+sqrt(deltan*deltan+4*((kn+kc_now)*deltan*deltaMaxLim/(k2Max_now-kn))));
      //             history[0] = newDeltaMax;
      //         }
      //     }
      // }

      // Three branch forces

      /*NL*/ //if(screen) fprintf(screen,"k1 = kn = %f; k2 = %f; kc = %f \n",kn,k2,kc);

      // const double Fn_damping = -gamman*cdata.vn;
      // const double Fn = fHys + Fn_damping;

      // cdata.Fn = Fn;
      // cdata.kn = kn;
      // cdata.kt = kt;
      // cdata.gamman = gamman;
      // cdata.gammat = gammat;

#ifdef NONSPHERICAL_ACTIVE_FLAG
      double Fn_i[3] = { Fn * cdata.en[0], Fn * cdata.en[1], Fn * cdata.en[2] };
      double torque_i[3] = { 0.0, 0.0, 0.0 };
      if(cdata.is_non_spherical) {
        double xci[3];
        vectorSubtract3D(cdata.contact_point, atom->x[cdata.i], xci);
        vectorCross3D(xci, Fn_i, torque_i);
      }
#endif
      // apply normal force
      if(cdata.is_wall) {
        const double Fn_ = Fn * cdata.area_ratio;
        i_forces.delta_F[0] = Fn_ * cdata.en[0];
        i_forces.delta_F[1] = Fn_ * cdata.en[1];
        i_forces.delta_F[2] = Fn_ * cdata.en[2];

#ifdef NONSPHERICAL_ACTIVE_FLAG
        if(cdata.is_non_spherical) {
          // for non-spherical particles normal force can produce torque!
          i_forces.delta_torque[0] += torque_i[0];
          i_forces.delta_torque[1] += torque_i[1];
          i_forces.delta_torque[2] += torque_i[2];
        }
#endif
      } else {
        // DO NOT use cdata.Fn here.
        // cdata.Fn is Fref.
        // Particle dynamics must use actual Fn.

        i_forces.delta_F[0] = Fn * cdata.en[0];
        i_forces.delta_F[1] = Fn * cdata.en[1];
        i_forces.delta_F[2] = Fn * cdata.en[2];

        j_forces.delta_F[0] = -i_forces.delta_F[0];
        j_forces.delta_F[1] = -i_forces.delta_F[1];
        j_forces.delta_F[2] = -i_forces.delta_F[2];

#ifdef NONSPHERICAL_ACTIVE_FLAG
        if(cdata.is_non_spherical) {
          // for non-spherical particles normal force can produce torque!
          double xcj[3], torque_j[3];
          double Fn_j[3] = { -Fn_i[0], -Fn_i[1], -Fn_i[2] };
          vectorSubtract3D(cdata.contact_point, atom->x[cdata.j], xcj);
          vectorCross3D(xcj, Fn_j, torque_j);

          i_forces.delta_torque[0] += torque_i[0];
          i_forces.delta_torque[1] += torque_i[1];
          i_forces.delta_torque[2] += torque_i[2];

          j_forces.delta_torque[0] += torque_j[0];
          j_forces.delta_torque[1] += torque_j[1];
          j_forces.delta_torque[2] += torque_j[2];
        }
#endif
      }
    }

    inline void noCollision(ContactData & cdata, ForceData&, ForceData&)
    {
      if(cdata.touch) *cdata.touch &= ~TOUCH_NORMAL_MODEL;
      // double * const history = &cdata.contact_history[history_offset];
      // history[0] = 0.0;
      cdata.contact_history[deltaMax_offset] = 0.0;
      cdata.contact_history[deltaP_offset] = 0.0;
    }

    void beginPass(CollisionData&, ForceData&, ForceData&){}
    void endPass(CollisionData&, ForceData&, ForceData&){}

  protected:
    double **k1;
    double **k2Max;
    double **kc;
    double **F0;
    double **phiF;

    double **coeffRestLog;

    double ktToKn;

    bool tangential_damping;
    bool fixedK2;

    int deltaMax_offset;
    int deltaP_offset;
  };
}
}
#endif // NORMAL_MODEL_LUDING_H_
#endif
