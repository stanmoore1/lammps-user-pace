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

  struct TagPairPACEComputeNeigh{};
  struct TagPairPACEComputeRadial{};
  struct TagPairPACEComputeYlm{};
  struct TagPairPACEComputeAi{};
  struct TagPairPACEConjugateAi{};
  struct TagPairPACEComputeWeights{};
  struct TagPairPACEComputeRho{};
  struct TagPairPACEComputeFS;

  template<int NEIGHFLAG, int EVFLAG>
  struct TagPairPACEComputeForce{};

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

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPACEComputeNeigh,const typename Kokkos::TeamPolicy<DeviceType, TagPairPACEComputeNeigh>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPACEComputeRadial,const typename Kokkos::TeamPolicy<DeviceType, TagPairPACEComputeRadial>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPACEComputeYlm,const typename Kokkos::TeamPolicy<DeviceType, TagPairPACEComputeYlm>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPACEComputeAi,const typename Kokkos::TeamPolicy<DeviceType, TagPairPACEComputeAi>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPACEConjugateAi,const int& ii) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPACEComputeRho,const int& ii) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPACEComputeFS,const int& ii) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPACEComputeWeights,const int& ii) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPACEComputeForce<NEIGHFLAG,EVFLAG>,const int& ii) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPACEComputeForce<NEIGHFLAG,EVFLAG>,const int& ii, EV_FLOAT&) const;

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

  friend void pair_virial_fdotr_compute<PairPACEKokkos>(PairPACEKokkos*);

  virtual void allocate();

  typedef Kokkos::View<int*, DeviceType> t_ace_1i;
  typedef Kokkos::View<double*, DeviceType> t_ace_1d;
  typedef Kokkos::View<complex*, DeviceType> t_ace_1c;
  typedef Kokkos::View<double**, DeviceType> t_ace_2d;
  typedef Kokkos::View<double*[3], DeviceType> t_ace_2d3;
  typedef Kokkos::View<complex****, DeviceType> t_ace_4c;
  typedef Kokkos::View<complex***[3], DeviceType> t_ace_4c3;

  t_ace_2d A_rank1; ///< 2D-array for storing A's for rank=1, shape: A(mu_j,n)
  t_ace_4c A; ///< 4D array with (l,m) last indices  for storing A's for rank>1: A(mu_j, n, l, m)

  t_ace_1d rhos; ///< densities \f$ \rho^{(p)} \f$(ndensity), p  = 0 .. ndensity-1
  t_ace_1d dF_drho; ///< derivatives of cluster functional wrt. densities, index = 0 .. ndensity-1

 // Spherical Harmonics

  void pre_compute_harmonics();

  KOKKOS_INLINE_FUNCTION
  void compute_barplm(double rz, int lmaxi);

  KOKKOS_INLINE_FUNCTION
  void compute_ylm(double rx, double ry, double rz, int lmaxi);

  t_ace_1d alm;
  t_ace_1d blm;
  t_ace_1d cl;
  t_ace_1d dl;

  t_ace_3d plm;
  t_ace_3d dplm;

  t_ace_3c ylm;
  t_ace_3c3 dylm;

  class SplineInterpolatorKokkos {
   public:
    int ntot, nlut, num_of_functions;
    double cutoff, deltaSplineBins, invrscalelookup, rscalelookup;

    t_ace_1d values, derivatives, second_derivatives;

    typedef Kokkos::View<double**[4], DeviceType> t_ace_3d4;
    t_ace_3d4 lookupTable;

    void operator=(const SplineInterpolator &spline) {
      cutoff = spline.cutoff;
      deltaSplineBins = spline.deltaSplineBins;
      ntot = spline.ntot;
      nlut = spline.nlut;
      invrscalelookup = spline.invrscalelookup;
      rscalelookup = spline.rscalelookup;
      num_of_functions = spline.num_of_functions;

      values = t_ace_1d("values", num_of_functions);
      derivatives = t_ace_1d("derivatives", num_of_functions);
      second_derivatives = t_ace_1d("second_derivatives", num_of_functions);

      lookupTable = t_ace_3d4("lookupTable", ntot+1, num_of_functions);
      auto h_lookupTable = Kokkos::create_mirror_view(lookupTable);
      for (int i = 0; i < ntot+1; i++)
        for (int j = 0; j < num_of_functions; j++)
          for (int k = 0; k < 4; k++)
            h_lookupTable(i, j, k) = spline.lookupTable(i, j, k);
      Kokkos::deep_copy(lookupTable, h_lookupTable);
    }

    void deallocate() {
      values = t_ace_1d();
      derivative = t_ace_1d();
      second_derivatives = t_ace_1d();
      lookupTable = t_ace_3d4();
    }
  };

  Kokkos::DualView<SplineInterpolatorKokkos**, DeviceType> k_splines_gk_kk;
  Kokkos::DualView<SplineInterpolatorKokkos**, DeviceType> k_splines_rnl_kk;
  Kokkos::DuaolView<SplineInterpolatorKokkos**, DeviceType> k_splines_hc_kk;

};
}    // namespace LAMMPS_NS

#endif
#endif
