/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Stan Moore (SNL)
------------------------------------------------------------------------- */

#include "pair_pace_kokkos.h"

#include "atom_kokkos.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"

#include <cstring>
#include <exception>

#include "ace_c_basis.h"
#include "ace_evaluator.h"
#include "ace_recursive.h"
#include "ace_version.h"

namespace LAMMPS_NS {
struct ACEImpl {
  ACEImpl() : basis_set(nullptr), ace(nullptr) {}
  ~ACEImpl()
  {
    delete basis_set;
    delete ace;
  }
  ACECTildeBasisSet *basis_set;
  ACERecursiveEvaluator *ace;
};
}    // namespace LAMMPS_NS

using namespace LAMMPS_NS;
using namespace MathConst;

static char const *const elements_pace[] = {
    "X",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si",
    "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac",
    "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"};
static constexpr int elements_num_pace = sizeof(elements_pace) / sizeof(const char *);

static int AtomicNumberByName_pace(char *elname)
{
  for (int i = 1; i < elements_num_pace; i++)
    if (strcmp(elname, elements_pace[i]) == 0) return i;
  return -1;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType
PairPACEKokkos<DeviceType>::PairPACEKokkos(LAMMPS *lmp) : PairPACE(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;

  k_cutsq = tdual_fparams("PairPACEKokkos::cutsq",atom->ntypes+1,atom->ntypes+1);
  auto d_cutsq = k_cutsq.template view<DeviceType>();
  rnd_cutsq = d_cutsq;

  host_flag = (execution_space == Host);
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

template<class DeviceType
PairPACEKokkos<DeviceType>::~PairPACEKokkos()
{
  if (copymode) return;

  memoryKK->destroy_kokkos(k_eatom,eatom);
  memoryKK->destroy_kokkos(k_vatom,vatom);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType
void PairPACEKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  copymode = 1;
  int newton_pair = force->newton_pair;
  if (newton_pair == false)
    error->all(FLERR,"PairPACEKokkos requires 'newton on'");

  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK);
  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  k_cutsq.template sync<DeviceType>();

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;
  inum = list->inum;

  need_dup = lmp->kokkos->need_dup<DeviceType>();
  if (need_dup) {
    dup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(f);
    dup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_vatom);
  } else {
    ndup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(f);
    ndup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_vatom);
  }

  if (inum != nlocal) error->all(FLERR, "inum: {} nlocal: {} are different", inum, nlocal);

  max_neighs = 0;
  Kokkos::parallel_reduce("PairPACEKokkos::find_max_neighs",inum, FindMaxNumNeighs<DeviceType>(k_list), Kokkos::Max<int>(max_neighs));

  aceimpl->ace->resize_neighbours_cache(max_neighs);

  //ComputeNeigh
  {
    // team_size_compute_neigh is defined in `pair_snap_kokkos.h`
    int scratch_size = scratch_size_helper<int>(team_size_compute_neigh * max_neighs);

    SnapAoSoATeamPolicy<DeviceType, team_size_compute_neigh, TagPairPACEComputeNeigh> policy_neigh(chunk_size,team_size_compute_neigh,vector_length);
    policy_neigh = policy_neigh.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
    Kokkos::parallel_for("ComputeNeigh",policy_neigh,*this);
  }

  //loop over atoms

  for (ii = 0; ii < list->inum; ii++) {
    i = h_ilist[ii];
    const int itype = type[i];

    const double xtmp = x[i][0];
    const double ytmp = x[i][1];
    const double ztmp = x[i][2];

    jnum = h_numneigh[i];

    // checking if neighbours are actually within cutoff range is done inside compute_atom
    // mapping from LAMMPS atom types ('type' array) to ACE species is done inside compute_atom
    //      by using 'aceimpl->ace->element_type_mapping' array
    // x: [r0 ,r1, r2, ..., r100]
    // i = 0 ,1
    // jnum(0) = 50
    // jlist(neigh ind of 0-atom) = [1,2,10,7,99,25, .. 50 element in total]

    try {
      aceimpl->ace->compute_atom(i, x, type, jnum, jlist);
    } catch (exception &e) {
      error->one(FLERR, e.what());
    }

    h_enery(ii) = aceimpl->ace->e_atom;

    for (jj = 0; jj < jnum; jj++) {
      h_forces(ii,jj,0) = aceimpl->ace->neighbours_forces(jj,0);
      h_forces(ii,jj,1) = aceimpl->ace->neighbours_forces(jj,1);
      h_forces(ii,jj,2) = aceimpl->ace->neighbours_forces(jj,2);
    }
  }

  // temp copy of energy, forces

  Kokkos::deep_copy(d_energy,h_energy);
  Kokkos::deep_copy(d_forces,h_forces);

  //ComputeForce
  {
    if (evflag) {
      if (neighflag == HALF) {
        typename Kokkos::RangePolicy<DeviceType,TagPairPACEComputeForce<HALF,1> > policy_force(0,inum);
        Kokkos::parallel_reduce(policy_force, *this, ev_tmp);
      } else if (neighflag == HALFTHREAD) {
        typename Kokkos::RangePolicy<DeviceType,TagPairPACEComputeForce<HALFTHREAD,1> > policy_force(0,inum);
        Kokkos::parallel_reduce(policy_force, *this, ev_tmp);
      }
    } else {
      if (neighflag == HALF) {
        typename Kokkos::RangePolicy<DeviceType,TagPairPACEComputeForce<HALF,0> > policy_force(0,inum);
        Kokkos::parallel_for(policy_force, *this);
      } else if (neighflag == HALFTHREAD) {
        typename Kokkos::RangePolicy<DeviceType,TagPairPACEComputeForce<HALFTHREAD,0> > policy_force(0,inum);
        Kokkos::parallel_for(policy_force, *this);
      }
    }
  }
  ev += ev_tmp;

  if (need_dup)
    Kokkos::Experimental::contribute(f, dup_f);

  if (eflag_global) eng_vdwl += ev.evdwl;
  if (vflag_global) {
    virial[0] += ev.v[0];
    virial[1] += ev.v[1];
    virial[2] += ev.v[2];
    virial[3] += ev.v[3];
    virial[4] += ev.v[4];
    virial[5] += ev.v[5];
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    if (need_dup)
      Kokkos::Experimental::contribute(d_vatom, dup_vatom);
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  atomKK->modified(execution_space,F_MASK);

  copymode = 0;

  // free duplicated memory
  if (need_dup) {
    dup_f     = decltype(dup_f)();
    dup_vatom = decltype(dup_vatom)();
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename real_type, int vector_length>
KOKKOS_INLINE_FUNCTION
void PairPACEKokkos<DeviceType, real_type, vector_length>::operator() (TagPairPACEComputeNeigh,const typename Kokkos::TeamPolicy<DeviceType,TagPairPACEComputeNeigh>::member_type& team) const {

  SNAKokkos<DeviceType, real_type, vector_length> my_sna = snaKK;

  // extract atom number
  int ii = team.team_rank() + team.league_rank() * team.team_size();
  if (ii >= chunk_size) return;

  // get a pointer to scratch memory
  // This is used to cache whether or not an atom is within the cutoff.
  // If it is, type_cache is assigned to the atom type.
  // If it's not, it's assigned to -1.
  const int tile_size = max_neighs; // number of elements per thread
  const int team_rank = team.team_rank();
  const int scratch_shift = team_rank * tile_size; // offset into pointer for entire team
  int* type_cache = (int*)team.team_shmem().get_shmem(team.team_size() * tile_size * sizeof(int), 0) + scratch_shift;

  // Load various info about myself up front
  const int i = d_ilist[ii + chunk_offset];
  const F_FLOAT xtmp = x(i,0);
  const F_FLOAT ytmp = x(i,1);
  const F_FLOAT ztmp = x(i,2);
  const int itype = type[i];
  const int ielem = d_map[itype];
  const double radi = d_radelem[ielem];

  const int num_neighs = d_numneigh[i];

  // rij[][3] = displacements between atom I and those neighbors
  // inside = indices of neighbors of I within cutoff
  // wj = weights for neighbors of I within cutoff
  // rcutij = cutoffs for neighbors of I within cutoff
  // note Rij sign convention => dU/dRij = dU/dRj = -dU/dRi

  // Compute the number of neighbors, store rsq
  int ninside = 0;
  Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team,num_neighs),
    [&] (const int jj, int& count) {
    int j = d_neighbors(i,jj);
    const F_FLOAT dx = x(j,0) - xtmp;
    const F_FLOAT dy = x(j,1) - ytmp;
    const F_FLOAT dz = x(j,2) - ztmp;

    int jtype = type(j);
    const F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

    if (rsq >= rnd_cutsq(itype,jtype)) {
      jtype = -1; // use -1 to signal it's outside the radius
    }

    type_cache[jj] = jtype;

    if (jtype >= 0)
     count++;
  }, ninside);

  d_ninside(ii) = ninside;

  Kokkos::parallel_scan(Kokkos::ThreadVectorRange(team,num_neighs),
    [&] (const int jj, int& offset, bool final) {

    const int jtype = type_cache[jj];

    if (jtype >= 0) {
      if (final) {
        int j = d_neighbors(i,jj);
        const F_FLOAT dx = x(j,0) - xtmp;
        const F_FLOAT dy = x(j,1) - ytmp;
        const F_FLOAT dz = x(j,2) - ztmp;
        const int elem_j = d_map[jtype];
        my_sna.rij(ii,offset,0) = static_cast<real_type>(dx);
        my_sna.rij(ii,offset,1) = static_cast<real_type>(dy);
        my_sna.rij(ii,offset,2) = static_cast<real_type>(dz);
        my_sna.wj(ii,offset) = static_cast<real_type>(d_wjelem[elem_j]);
        my_sna.rcutij(ii,offset) = static_cast<real_type>((radi + d_radelem[elem_j])*rcutfac);
        my_sna.inside(ii,offset) = j;
        if (chemflag)
          my_sna.element(ii,offset) = elem_j;
        else
          my_sna.element(ii,offset) = 0;
      }
      offset++;
    }
  });
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename real_type, int vector_length>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairPACEKokkos<DeviceType, real_type, vector_length>::operator() (TagPairPACEComputeForce<NEIGHFLAG,EVFLAG>, const int& ii, EV_FLOAT& ev) const {

  // The f array is duplicated for OpenMP, atomic for CUDA, and neither for Serial
  auto v_f = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  auto a_f = v_f.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  const int i = d_ilist[ii + chunk_offset];

  SNAKokkos<DeviceType, real_type, vector_length> my_sna = snaKK;

  const int ninside = d_ninside(ii);

  for (int jj = 0; jj < ninside; jj++) {
    int j = my_sna.inside(ii,jj);

  for (ii = 0; ii < list->inum; ii++) {
    i = h_ilist[ii];
    const int itype = type[i];

    const double xtmp = x(i,0);
    const double ytmp = x(i,1);
    const double ztmp = x(i,2);

    jnum = d_numneigh[i];

    // 'compute_atom' will update the `aceimpl->ace->e_atom` and `aceimpl->ace->neighbours_forces(jj, alpha)` arrays

    for (jj = 0; jj < jnum; jj++) {
      j = d_neighbors(i,jj);
      j &= NEIGHMASK;
      delx = x(j,0) - xtmp;
      dely = x(j,1) - ytmp;
      delz = x(j,2) - ztmp;

      fij[0] = d_scale(itype,itype) * d_forces(jj,0);
      fij[1] = d_scale(itype,itype) * d_forces(jj,1);
      fij[2] = d_scale(itype,itype) * d_forces(jj,2);

      a_f(i,0) += fij[0];
      a_f(i,1) += fij[1];
      a_f(i,2) += fij[2];
      a_f(j,0) -= fij[0];
      a_f(j,1) -= fij[1];
      a_f(j,2) -= fij[2];

      // tally per-atom virial contribution
      if (vflag)
        ev_tally_xyz(i, j, nlocal, newton_pair, 0.0, 0.0, fij[0], fij[1], fij[2], -delx, -dely,
                     -delz);
    }

    // tally energy contribution
    if (eflag) {
      // evdwl = energy of atom I
      evdwl = scale[itype][itype]*aceimpl->ace->e_atom;
      ev_tally_full(i, 2.0 * evdwl, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
  }
}

template<class DeviceType, typename real_type, int vector_length>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairPACEKokkos<DeviceType, real_type, vector_length>::operator() (TagPairPACEComputeForce<NEIGHFLAG,EVFLAG>,const int& ii) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,EVFLAG>(TagPairPACEComputeForce<NEIGHFLAG,EVFLAG>(), ii, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename real_type, int vector_length>
template<int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION
void PairPACEKokkos<DeviceType, real_type, vector_length>::v_tally_xyz(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &fx, const F_FLOAT &fy, const F_FLOAT &fz,
      const F_FLOAT &delx, const F_FLOAT &dely, const F_FLOAT &delz) const
{
  // The vatom array is duplicated for OpenMP, atomic for CUDA, and neither for Serial

  auto v_vatom = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_vatom),decltype(ndup_vatom)>::get(dup_vatom,ndup_vatom);
  auto a_vatom = v_vatom.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  const E_FLOAT v0 = delx*fx;
  const E_FLOAT v1 = dely*fy;
  const E_FLOAT v2 = delz*fz;
  const E_FLOAT v3 = delx*fy;
  const E_FLOAT v4 = delx*fz;
  const E_FLOAT v5 = dely*fz;

  if (vflag_global) {
    ev.v[0] += v0;
    ev.v[1] += v1;
    ev.v[2] += v2;
    ev.v[3] += v3;
    ev.v[4] += v4;
    ev.v[5] += v5;
  }

  if (vflag_atom) {
    a_vatom(i,0) += 0.5*v0;
    a_vatom(i,1) += 0.5*v1;
    a_vatom(i,2) += 0.5*v2;
    a_vatom(i,3) += 0.5*v3;
    a_vatom(i,4) += 0.5*v4;
    a_vatom(i,5) += 0.5*v5;
    a_vatom(j,0) += 0.5*v0;
    a_vatom(j,1) += 0.5*v1;
    a_vatom(j,2) += 0.5*v2;
    a_vatom(j,3) += 0.5*v3;
    a_vatom(j,4) += 0.5*v4;
    a_vatom(j,5) += 0.5*v5;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType
void PairPACEKokkos<DeviceType>::allocate()
{
  PairPACE::allocate();

  int n = atom->ntypes + 1;
  d_scale = Kokkos::View<int*, DeviceType>("PairPACEKokkos::map",n);
  d_map = Kokkos::View<int*, DeviceType>("PairPACEKokkos::map",n);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

template<class DeviceType
void PairPACEKokkos<DeviceType>::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  map_element2type(narg - 3, arg + 3);

  auto potential_file_name = utils::get_potential_file_path(arg[2]);
  char **elemtypes = &arg[3];

  //load potential file
  delete aceimpl->basis_set;
  if (comm->me == 0) utils::logmesg(lmp, "Loading {}\n", potential_file_name);
  aceimpl->basis_set = new ACECTildeBasisSet(potential_file_name);

  if (comm->me == 0) {
    utils::logmesg(lmp, "Total number of basis functions\n");

    for (SPECIES_TYPE mu = 0; mu < aceimpl->basis_set->nelements; mu++) {
      int n_r1 = aceimpl->basis_set->total_basis_size_rank1[mu];
      int n = aceimpl->basis_set->total_basis_size[mu];
      utils::logmesg(lmp, "\t{}: {} (r=1) {} (r>1)\n", aceimpl->basis_set->elements_name[mu], n_r1,
                     n);
    }
  }

  // read args that map atom types to pACE elements
  // map[i] = which element the Ith atom type is, -1 if not mapped
  // map[0] is not used

  delete aceimpl->ace;
  aceimpl->ace = new ACERecursiveEvaluator();
  aceimpl->ace->set_recursive(recursive);
  aceimpl->ace->element_type_mapping.init(atom->ntypes + 1);

  const int n = atom->ntypes;
  for (int i = 1; i <= n; i++) {
    char *elemname = elemtypes[i - 1];
    int atomic_number = AtomicNumberByName_pace(elemname);
    if (atomic_number == -1) error->all(FLERR, "'{}' is not a valid element\n", elemname);

    SPECIES_TYPE mu = aceimpl->basis_set->get_species_index_by_name(elemname);
    if (mu != -1) {
      if (comm->me == 0)
        utils::logmesg(lmp, "Mapping LAMMPS atom type #{}({}) -> ACE species type #{}\n", i,
                       elemname, mu);
      map[i] = mu;
      // set up LAMMPS atom type to ACE species  mapping for ace evaluator
      aceimpl->ace->element_type_mapping(i) = mu;
    } else {
      error->all(FLERR, "Element {} is not supported by ACE-potential from file {}", elemname,
                 potential_file_name);
    }
  }

  // initialize scale factor
  for (int i = 1; i <= n; i++) {
    for (int j = i; j <= n; j++) { scale[i][j] = 1.0; }
  }

  aceimpl->ace->set_basis(*aceimpl->basis_set, 1);


  PairPACE::coeff(narg,arg);

  // Set up element lists

  d_radelem = Kokkos::View<real_type*, DeviceType>("pair:radelem",nelements);
  d_wjelem = Kokkos::View<real_type*, DeviceType>("pair:wjelem",nelements);
  d_coeffelem = Kokkos::View<real_type**, Kokkos::LayoutRight, DeviceType>("pair:coeffelem",nelements,ncoeffall);

  auto h_radelem = Kokkos::create_mirror_view(d_radelem);
  auto h_wjelem = Kokkos::create_mirror_view(d_wjelem);
  auto h_coeffelem = Kokkos::create_mirror_view(d_coeffelem);
  auto h_map = Kokkos::create_mirror_view(d_map);

  for (int ielem = 0; ielem < nelements; ielem++) {
    h_radelem(ielem) = radelem[ielem];
    h_wjelem(ielem) = wjelem[ielem];
    for (int jcoeff = 0; jcoeff < ncoeffall; jcoeff++) {
      h_coeffelem(ielem,jcoeff) = coeffelem[ielem][jcoeff];
    }
  }

  for (int i = 1; i <= atom->ntypes; i++) {
    h_map(i) = map[i];
  }

  Kokkos::deep_copy(d_radelem,h_radelem);
  Kokkos::deep_copy(d_wjelem,h_wjelem);
  Kokkos::deep_copy(d_coeffelem,h_coeffelem);
  Kokkos::deep_copy(d_map,h_map);

  snaKK = SNAKokkos<DeviceType, real_type, vector_length>(rfac0,twojmax,
                  rmin0,switchflag,bzeroflag,chemflag,bnormflag,wselfallflag,nelements);
  snaKK.grow_rij(0,0);
  snaKK.init();
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType
void PairPACEKokkos<DeviceType>::init_style()
{
  if (atom->tag_enable == 0) error->all(FLERR, "Pair style pACE requires atom IDs");
  if (force->newton_pair == 0) error->all(FLERR, "Pair style pACE requires newton pair on");

  // neighbor list request for KOKKOS

  neighflag = lmp->kokkos->neighflag;

  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_host(std::is_same<DeviceType,LMPHostType>::value &&
                           !std::is_same<DeviceType,LMPDeviceType>::value);
  request->set_kokkos_device(std::is_same<DeviceType,LMPDeviceType>::value);
  if (neighflag == FULL)
    error->all(FLERR,"Must use half neighbor list style with pair snap/kk");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template<class DeviceType
double PairPACEKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairPACE::init_one(i,j);
  k_cutsq.h_view(i,j) = k_cutsq.h_view(j,i) = cutone*cutone;
  k_cutsq.template modify<LMPHostType>();

  return cutone;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct FindMaxNumNeighs {
  typedef DeviceType device_type;
  NeighListKokkos<DeviceType> k_list;

  FindMaxNumNeighs(NeighListKokkos<DeviceType>* nl): k_list(*nl) {}
  ~FindMaxNumNeighs() {k_list.copymode = 1;}

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& ii, int& max_neighs) const {
    const int i = k_list.d_ilist[ii];
    const int num_neighs = k_list.d_numneigh[i];
    if (max_neighs<num_neighs) max_neighs = num_neighs;
  }
};

