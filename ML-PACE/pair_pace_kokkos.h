/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(pace/kk,PairPACEKokkosDevice<LMPDeviceType>);
PairStyle(pace/kk/device,PairPACEKokkosDevice<LMPDeviceType>);
#ifdef LMP_KOKKOS_GPU
PairStyle(pace/kk/host,PairPACEKokkosHost<LMPHostType>);
#else
PairStyle(pace/kk/host,PairPACEKokkosDevice<LMPHostType>);
#endif
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_PACE_KOKKOS_H
#define LMP_PAIR_PACE_KOKKOS_H

#include "pair_pace.h"
#include "kokkos_type.h"
#include "neigh_list_kokkos.h"
#include "pair_kokkos.h"

namespace LAMMPS_NS {

class PairPACEKokkos : public PairPACE {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairPACEKokkos(class LAMMPS *);
  ~PairPACEKokkos() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

 protected:
  struct ACEImpl *aceimpl;

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  typename AT::t_x_array_randomread x;
  typename AT::t_f_array f;
  typename AT::t_int_1d_randomread type;

  int need_dup;

  using KKDeviceType = typename KKDevice<DeviceType>::value;

  template<typename DataType, typename Layout>
  using DupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterDuplicated>;

  template<typename DataType, typename Layout>
  using NonDupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterNonDuplicated>;

  DupScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout> dup_f;
  DupScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout> dup_vatom;

  NonDupScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout> ndup_f;
  NonDupScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout> ndup_vatom;

  friend void pair_virial_fdotr_compute<PairSNAPKokkos>(PairSNAPKokkos*);

  virtual void allocate();

  double **scale;
  bool recursive;    // "recursive" option for ACERecursiveEvaluator
};
}    // namespace LAMMPS_NS

#endif
#endif
