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
    static const int MASK = CM_REGISTER_SETTINGS | CM_CONNECT_TO_PROPERTIES |
                            CM_COLLISION | CM_NO_COLLISION;

    NormalModel(LAMMPS * lmp, IContactHistorySetup * hsetup) : Pointers(lmp),
        k1(NULL), k2Max(NULL), kc(NULL), F0(NULL), phiF(NULL),
        coeffRestLog(NULL),
        ktToKn(1.0), tangential_damping(false), fixedK2(false)
    {
      deltaMax_offset = hsetup->add_history_value("deltaMax", "0");
      deltaP_offset = hsetup->add_history_value("deltaP", "0");

      if(comm->me == 0 && screen) fprintf(screen, "LUDING loaded\n");
    }

    inline void registerSettings(Settings & settings)
    {
      settings.registerOnOff("tangential_damping", tangential_damping, true);
      settings.registerOnOff("fixedK2", fixedK2);
      settings.registerDoubleSetting("ktToKn", ktToKn, 1.0);
    }

    inline void connectToProperties(PropertyRegistry & registry)
    {
      // Pair properties: loading/unloading stiffness, adhesion slope and force offset.
      registry.registerProperty("k1", &MODEL_PARAMS::createLudingK1);
      registry.registerProperty("k2Max", &MODEL_PARAMS::createLudingK2Max);
      registry.registerProperty("kc", &MODEL_PARAMS::createLudingKc);
      registry.registerProperty("F0", &MODEL_PARAMS::createLudingF0);

      registry.connect("k1", k1, "model luding");
      registry.connect("k2Max", k2Max, "model luding");
      registry.connect("kc", kc, "model luding");
      registry.connect("F0", F0, "model luding");

      // Plastic-depth scale is only required for history-dependent hardening.
      if(!fixedK2) {
        registry.registerProperty("phiF", &MODEL_PARAMS::createLudingPhiF);
        registry.connect("phiF", phiF, "model luding");
      }

      registry.registerProperty("coeffRestLog", &MODEL_PARAMS::createCoeffRestLog);
      registry.connect("coeffRestLog", coeffRestLog, "model luding");
    }

    // Linear stress-strain exponent for heat-transfer area correction.
    inline double stressStrainExponent()
    {
      return 1.;
    }

    inline void collision(CollisionData & cdata, ForceData & i_forces, ForceData & j_forces)
    {
      // Contact geometry and material properties for this particle/type pair.
      // Effective radius is ri*rj/(ri+rj) for spheres and ri for a plane wall.
      const int itype = cdata.itype;
      const int jtype = cdata.jtype;
      const double deltan = cdata.deltan;
      double ri = cdata.radi;
      double rj = cdata.radj;
      double reff = cdata.is_wall ? cdata.radi : (ri * rj / (ri + rj));

#ifdef SUPERQUADRIC_ACTIVE_FLAG
      if(cdata.is_non_spherical && atom->superquadric_flag)
        reff = cdata.reff;
#endif

      double meff = cdata.meff;
      const double k1_now = k1[itype][jtype];
      const double k2Max_now = k2Max[itype][jtype];
      const double kc_now = kc[itype][jtype];
      const double F0_now = F0[itype][jtype];

      // Contact history: deltaMax never decreases during an active contact;
      // deltaP is the zero-force intercept of the unloading line before F0.
      if(cdata.touch) *cdata.touch |= TOUCH_NORMAL_MODEL;
      double & deltaMax = cdata.contact_history[deltaMax_offset];
      double & deltaP = cdata.contact_history[deltaP_offset];

      const bool newMaximum = (deltan > deltaMax);
      if(newMaximum)
        deltaMax = deltan;

      // Unloading stiffness: constant k2Max or linear hardening from k1 to k2Max.
      // phiF sets the overlap scale at which hardening saturates.
      double k2_now;
      if(fixedK2) {
        k2_now = k2Max_now;
      } else {
        const double phiF_now = phiF[itype][jtype];
        const double deltaMaxLim = k2Max_now / (k2Max_now - k1_now) *
                                   phiF_now * 2.0 * reff;

        if(deltaMax >= deltaMaxLim) {
          k2_now = k2Max_now;
        } else {
          k2_now = k1_now + (k2Max_now - k1_now) * deltaMax / deltaMaxLim;
        }
      }

      // Keep the unloading line through the new loading point: k2*(delta-deltaP)=k1*delta.
      if(newMaximum) {
        deltaP = (1.0 - k1_now / k2_now) * deltaMax;
      }

      // Three force branches before applying the common attractive F0 offset.
      const double fLoad = k1_now * deltan;
      double fUnload = k2_now * (deltan - deltaP);
      const double fAdhesion = -kc_now * deltan;

      // Reloading from adhesion: vn < 0 means increasing overlap. Move deltaP
      // so the new unloading/reloading line passes through -kc*delta continuously.
      bool reloadFromAdhesion = false;
      if(!newMaximum && cdata.vn < 0.0 && fUnload <= fAdhesion) {
        deltaP = deltan * (1.0 + kc_now / k2_now);
        fUnload = k2_now * (deltan - deltaP);
        reloadFromAdhesion = true;
      }

      // Select the loading envelope, unloading/reloading line, or adhesion bound.
      double fBase;
      double kn_current;
      if(newMaximum) {
        // Virgin loading beyond the historical maximum always follows k1.
        // Avoid round-off switching damping branches when fUnload == fLoad.
        fBase = fLoad;
        kn_current = k1_now;
      } else if(fUnload >= fLoad) {
        // Reloading has reached the loading envelope.
        fBase = fLoad;
        kn_current = k1_now;
      } else if(fUnload > fAdhesion || reloadFromAdhesion) {
        // Follow k2 between the loading envelope and adhesion bound.
        fBase = fUnload;
        kn_current = k2_now;
      } else {
        // Attractive branch: force decreases as overlap increases.
        fBase = fAdhesion;
        kn_current = kc_now;
      }

      // F0 is a positive attraction magnitude applied to every branch.
      const double FL = fBase - F0_now;

      // Friction/rolling uses Fref = max(FL + F0 + kc*delta, 0).
      // It excludes damping and vanishes on adhesion (FL = -kc*delta-F0).
      double Fref = FL + F0_now + kc_now * deltan;
      if(Fref < 0.0)
        Fref = 0.0;

      // Tangential stiffness remains ktToKn*k1, independent of the normal branch.
      const double kt = ktToKn * k1_now;

      // Dashpots use ln(e): normal damping follows k1/k2/kc of the active branch;
      // optional tangential damping uses kt. Setting e=1 removes both dashpots.
      const double coeffRestLogChosen = coeffRestLog[itype][jtype];
      const double coeffRestLogChosenSq = coeffRestLogChosen * coeffRestLogChosen;
      const double gamman =
          sqrt(4.0 * meff * kn_current * coeffRestLogChosenSq /
               (coeffRestLogChosenSq + M_PI * M_PI));
      const double gammat = tangential_damping ?
          sqrt(4.0 * meff * kt * coeffRestLogChosenSq /
               (coeffRestLogChosenSq + M_PI * M_PI)) : 0.0;

      // Actual normal force includes the attractive offset and velocity damping.
      const double Fn_damping = -gamman * cdata.vn;
      const double Fn = FL + Fn_damping;

      // cdata.Fn is the friction reference; dynamics below uses actual Fn.
      cdata.Fn = Fref;
      cdata.kn = kn_current;
      cdata.kt = kt;
      cdata.gamman = gamman;
      cdata.gammat = gammat;

#ifdef NONSPHERICAL_ACTIVE_FLAG
      // Off-center normal contact forces also produce torque on nonspherical particles.
      double Fn_i[3] = { Fn * cdata.en[0], Fn * cdata.en[1], Fn * cdata.en[2] };
      double torque_i[3] = { 0.0, 0.0, 0.0 };
      if(cdata.is_non_spherical) {
        double xci[3];
        vectorSubtract3D(cdata.contact_point, atom->x[cdata.i], xci);
        vectorCross3D(xci, Fn_i, torque_i);
      }
#endif

      // Apply the actual force, with wall area weighting or equal/opposite pair forces.
      if(cdata.is_wall) {
        const double Fn_ = Fn * cdata.area_ratio;
        i_forces.delta_F[0] = Fn_ * cdata.en[0];
        i_forces.delta_F[1] = Fn_ * cdata.en[1];
        i_forces.delta_F[2] = Fn_ * cdata.en[2];

#ifdef NONSPHERICAL_ACTIVE_FLAG
        if(cdata.is_non_spherical) {
          i_forces.delta_torque[0] += torque_i[0];
          i_forces.delta_torque[1] += torque_i[1];
          i_forces.delta_torque[2] += torque_i[2];
        }
#endif
      } else {
        i_forces.delta_F[0] = Fn * cdata.en[0];
        i_forces.delta_F[1] = Fn * cdata.en[1];
        i_forces.delta_F[2] = Fn * cdata.en[2];

        j_forces.delta_F[0] = -i_forces.delta_F[0];
        j_forces.delta_F[1] = -i_forces.delta_F[1];
        j_forces.delta_F[2] = -i_forces.delta_F[2];

#ifdef NONSPHERICAL_ACTIVE_FLAG
        if(cdata.is_non_spherical) {
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
      // Separation ends this contact; a future contact starts with fresh history.
      if(cdata.touch) *cdata.touch &= ~TOUCH_NORMAL_MODEL;
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
