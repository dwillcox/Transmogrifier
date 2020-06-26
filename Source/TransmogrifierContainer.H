#ifndef TRANSMOGRIFIER_CONTAINER_H_
#define TRANSMOGRIFIER_CONTAINER_H_

#include <AMReX_Particles.H>
#include <AMReX_ParticleMesh.H>
#include <AMReX_Geometry.H>

// Indexes into particle real struct data for the A & B positions
struct PIdx
{
    enum {x_A = 0, y_A, z_A,
          x_B, y_B, z_B,
          NCoord};
};

// Ncomp is the number of components from the grid we want to interpolate
// We add 6 components (the first 6 Reals in struct data) to store:
// x_A, y_A, z_A, x_B, y_B, z_B for transmogrifying A->B
template<int Ncomp>
class TransmogrifierContainer
    : public amrex::ParticleContainer<PIdx::NCoord+Ncomp, 0, 0, 0>
{

    // Typedef the base particle container, particle iterator and the particle type
    typedef amrex::ParticleContainer<PIdx::NCoord+Ncomp, 0, 0, 0> BasePC;
    typedef amrex::ParIter<BasePC::NStructReal, 0, 0, 0> TParIter;
    typedef amrex::Particle<BasePC::NStructReal, 0> TParticleType;

    // Store the grid metadata for A & B domains
    amrex::ParGDB gdb_A, gdb_B;

public:

    TransmogrifierContainer(const amrex::Geometry& geom_A,
                            const amrex::DistributionMapping& dmap_A,
                            const amrex::BoxArray& ba_A,
                            const amrex::Geometry& geom_B,
                            const amrex::DistributionMapping& dmap_B,
                            const amrex::BoxArray& ba_B) :
                            amrex::ParticleContainer<PIdx::NCoord+Ncomp, 0, 0, 0>(geom_B, dmap_B, ba_B)
    {
        // Save the A & B domain metadata in our gdb objects.
        // We don't assume anything about which of A or B was
        // passed to the constructor.
        gdb_A = amrex::ParGDB(geom_A, dmap_A, ba_A);
        gdb_B = amrex::ParGDB(geom_B, dmap_B, ba_B);

        // Do some input sanity checking while we're at it
        const auto gdata_A = geom_A.data();
        const auto gdata_B = geom_B.data();

        // Require that A be defined in the 2D R-Z coordinate system
        AMREX_ASSERT(gdata_A.Coord() == amrex::CoordSys::RZ);

        // Require that B be defined in the 3D Cartesian coordinate system
        AMREX_ASSERT(gdata_B.Coord() == amrex::CoordSys::cartesian);
        AMREX_ASSERT(AMREX_SPACEDIM == 3);

        // Require that cell size dz = dz between R-Z and Cartesian geometries
        AMREX_ASSERT(gdata_A.CellSize(1) == gdata_B.CellSize(2));

        // Initialize particles in domain B
        InitParticles();
    }


    void Interpolate(const amrex::MultiFab& state_A, amrex::MultiFab& state_B)
    {
        // Given A and B grid data, use the particles created in B cells to
        // interpolate data from A to B.

        // Configure metadata & particle positions for A & Redistribute
        amrex::Print() << "Moving particles to A\n";
        SetA();

        // Interpolate mesh to particles from A
        amrex::Print() << "Interpolating mesh to particles from A\n";
        InterpolateFromMesh(state_A, gdb_A.Geom(0));

        // Configure metadata & particle positions for B & Redistribute
        amrex::Print() << "Moving particles to B\n";
        SetB();

        // Deposit particle data onto mesh in B
        amrex::Print() << "Depositing particle data onto mesh in B\n";
        DepositToMesh(state_B, gdb_B.Geom(0));
    }


    void InitParticles()
    {
        // Create 1 particle centered in each cell in B,
        // and initialize its position in both B and A.
        amrex::Print() << "Initializing particles in B\n";
        SetB();

        // Construct a ParticleInitData containing only zeros
        amrex::ParticleInitType<BasePC::NStructReal, 0, 0, 0> pdata {};

        amrex::Real x_offset = 0;
        amrex::Real y_offset = 0;
        amrex::Real z_offset = 0;

        // Do not offset in extra dimensions
        AMREX_D_TERM( x_offset = 0.5; ,
                      y_offset = 0.5; ,
                      z_offset = 0.5; )

        BasePC::InitOnePerCell(x_offset, y_offset, z_offset, pdata);

        ComputePositionAB();
    }


    void SetA()
    {
        // Updates the geometry, distribution mapping and box array
        // for the particle container using the metadata for A.
        BasePC::m_gdb_object = gdb_A; 

        // Set particle positions to their positions in A
#ifdef _OPENMP
#pragma omp parallel
#endif
        for (TParIter pti(*this, 0); pti.isValid(); ++pti)
        {
            const int np  = pti.numParticles();

            TParticleType* aos_p = &(pti.GetArrayOfStructs()[0]);

            amrex::ParallelFor ( np, [=] AMREX_GPU_DEVICE (int i) {
                TParticleType& p = aos_p[i];

                // Fill (x, y, z) for p.pos from A
                AMREX_D_TERM( p.pos(0) = p.rdata(PIdx::x_A); ,
                              p.pos(1) = p.rdata(PIdx::y_A); ,
                              p.pos(2) = p.rdata(PIdx::z_A); )
            });
        }

        BasePC::Redistribute();
    }


    void SetB()
    {
        // Updates the geometry, distribution mapping and box array
        // for the particle container using the metadata for B.
        BasePC::m_gdb_object = gdb_B; 

        // Set particle positions to their positions in B
#ifdef _OPENMP
#pragma omp parallel
#endif
        for (TParIter pti(*this, 0); pti.isValid(); ++pti)
        {
            const int np  = pti.numParticles();

            TParticleType* aos_p = &(pti.GetArrayOfStructs()[0]);

            amrex::ParallelFor ( np, [=] AMREX_GPU_DEVICE (int i) {
                TParticleType& p = aos_p[i];

                // Fill (x, y, z) for p.pos from B
                AMREX_D_TERM( p.pos(0) = p.rdata(PIdx::x_B); ,
                              p.pos(1) = p.rdata(PIdx::y_B); ,
                              p.pos(2) = p.rdata(PIdx::z_B); )
            });
        }

        BasePC::Redistribute();
    }


    void ComputePositionAB()
    {
        // This function sets x_A, y_A, z_A, x_B, y_B, z_B.
        // assuming A is 2D R-Z geometry, and B is 3D Cartesian geometry.
        //
        // Here we assume the Z-axis is the same between B and A
        // and that the X, Y plane in B is centered on the r=0 origin in A.
        auto A_from_B_map = [] AMREX_GPU_DEVICE (amrex::Real& x_A, amrex::Real& y_A, amrex::Real& z_A,
                                                 const amrex::Real& x_B, const amrex::Real& y_B, const amrex::Real& z_B)
        {
            // A: 2D cylindrical (r,z) with rlo = 0.
            // B: 3D cartesian (x, y, z) with (x=0, y=0) at r=0.
            z_A = 0.0; // (A is 2D)
            y_A = z_B; // (since A is 2D, y_A = cartesian z)
            x_A = std::sqrt(x_B * x_B + y_B * y_B); // r = sqrt(x^2 + y^2)
        };

#ifdef _OPENMP
#pragma omp parallel
#endif
        for (TParIter pti(*this, 0); pti.isValid(); ++pti)
        {
            const int np  = pti.numParticles();

            TParticleType* aos_p = &(pti.GetArrayOfStructs()[0]);

            amrex::ParallelFor ( np, [=] AMREX_GPU_DEVICE (int i) {
                TParticleType& p = aos_p[i];

                // Zero out positions before we start
                for (int j = 0; j < PIdx::NCoord; ++j) p.rdata(j) = 0.0;

                // Fill (x, y, z) for B from p.pos
                AMREX_D_TERM( p.rdata(PIdx::x_B) = p.pos(0); ,
                              p.rdata(PIdx::y_B) = p.pos(1); ,
                              p.rdata(PIdx::z_B) = p.pos(2); )

                // Call the user function to set the position in A from B
                A_from_B_map(p.rdata(PIdx::x_A),
                             p.rdata(PIdx::y_A),
                             p.rdata(PIdx::z_A),
                             p.rdata(PIdx::x_B),
                             p.rdata(PIdx::y_B),
                             p.rdata(PIdx::z_B));
            });
        }
    }


    void DepositToMesh(amrex::MultiFab& state, const amrex::Geometry& geom)
    {
        const auto plo = geom.ProbLoArray();
        const auto dxi = geom.InvCellSizeArray();

        const auto dbox = geom.Domain();
        const auto dlo  = amrex::lbound(dbox);

        amrex::ParticleToMesh(*this, state, 0,
        [=] AMREX_GPU_DEVICE (const TParticleType& p,
                              amrex::Array4<amrex::Real> const& sarr)
        {
            // Get particle offset from physical domain lo
            amrex::Real lx = (p.pos(0) - plo[0]) * dxi[0];
            amrex::Real ly = (p.pos(1) - plo[1]) * dxi[1];
            amrex::Real lz = (p.pos(2) - plo[2]) * dxi[2];

            // Get the cell index where the particle is located
            int i = amrex::Math::floor(lx) + dlo.x;
            int j = amrex::Math::floor(ly) + dlo.y;
            int k = amrex::Math::floor(lz) + dlo.z;

            // Because we originally created 1 particle per cell at
            // the center of each cell in B, all we have to do
            // is directly inject the particle quantities into the cell here.
            for (int n = 0; n < Ncomp; ++n) {
                sarr(i, j, k, n) = p.rdata(PIdx::NCoord + n);
            }
        });
    }


    void InterpolateFromMesh(const amrex::MultiFab& state, const amrex::Geometry& geom)
    {
        const auto dbox = geom.Domain();
        const auto dlo  = amrex::lbound(dbox);
        const auto dhi  = amrex::ubound(dbox);

        const auto plo = geom.ProbLoArray();
        const auto dxi = geom.InvCellSizeArray();
        const auto dxc = geom.CellSizeArray();

        amrex::MeshToParticle(*this, state, 0,
        [=] AMREX_GPU_DEVICE (TParticleType& p,
                              amrex::Array4<const amrex::Real> const& sarr)
        {
            // Get particle offset from physical domain lo
            amrex::Real lx = (p.pos(0) - plo[0]) * dxi[0];
            amrex::Real ly = (p.pos(1) - plo[1]) * dxi[1];
            amrex::Real lz = (p.pos(2) - plo[2]) * dxi[2];

            // Get the cell index where the particle is located
            int i = amrex::Math::floor(lx) + dlo.x;
            int j = amrex::Math::floor(ly) + dlo.y;
            int k = amrex::Math::floor(lz) + dlo.z;

            // Get the index bounds of the Array4
            const auto slo = amrex::lbound(sarr);
            const auto shi = amrex::ubound(sarr);

            // 1D Quadratic interpolation
            auto quad_interpolate_1D = [&](const amrex::Real dx, const amrex::Real dxm, const amrex::Real dxp,
                                           const amrex::Real Fi, const amrex::Real Fim, const amrex::Real Fip) -> amrex::Real
            {
                return ((Fim*(dx - dxm) - dxm*(Fi - Fim))*((dx - dxm)*(-std::pow(dxm, 2) + std::pow(dxp, 2)) - (std::pow(dx, 2) - std::pow(dxm, 2))*(-dxm + dxp)) - (std::pow(dxm, 2)*(dx - dxm) - dxm*(std::pow(dx, 2) - std::pow(dxm, 2)))*(-(Fi - Fim)*(-dxm + dxp) + (-Fim + Fip)*(dx - dxm)))/((dx - dxm)*((dx - dxm)*(-std::pow(dxm, 2) + std::pow(dxp, 2)) - (std::pow(dx, 2) - std::pow(dxm, 2))*(-dxm + dxp)));
            };

            // Modify the center of the interpolation grid depending on how close we are to the edge of the sarr.
            int ic = i;
            int jc = j;
            int kc = k;

            // amrex::Print() << "kc = " << kc << ", p.pos(2) = " << p.pos(2) << ", plo[2] = " << plo[2] << ", dxi[2] = " << dxi[2] << ", dlo.z " << dlo.z << "\n";

            AMREX_D_TERM( if (i == slo.x) ic++;
                          if (i == shi.x) ic--; ,

                          if (j == slo.y) jc++;
                          if (j == shi.y) jc--; ,

                          if (k == slo.z) kc++;
                          if (k == shi.z) kc--; )

            // Location of the interpolation center cell's center
            amrex::Real cell_x = (ic - dlo.x + 0.5) * dxc[0] + plo[0];
            amrex::Real cell_y = (jc - dlo.y + 0.5) * dxc[1] + plo[1];
            amrex::Real cell_z = (kc - dlo.z + 0.5) * dxc[2] + plo[2];

            // Do 2D Quadratic interpolation from surrounding cells, first in R, then Z
            amrex::Real Fi, Fim, Fip;
            amrex::Real Fj, Fjm, Fjp;
            amrex::Real dx, dxm, dxp;

            for (int n = 0; n < Ncomp; ++n) {
                dx = cell_x - p.pos(0);
                dxm = dx - dxc[0];
                dxp = dx + dxc[0];

                Fi  = sarr(ic  , jc, kc, n);
                Fim = sarr(ic-1, jc, kc, n);
                Fip = sarr(ic+1, jc, kc, n);
                Fj = quad_interpolate_1D(dx, dxm, dxp, Fi, Fim, Fip);

                Fi  = sarr(ic  , jc-1, kc, n);
                Fim = sarr(ic-1, jc-1, kc, n);
                Fip = sarr(ic+1, jc-1, kc, n);
                Fjm = quad_interpolate_1D(dx, dxm, dxp, Fi, Fim, Fip);

                Fi  = sarr(ic  , jc+1, kc, n);
                Fim = sarr(ic-1, jc+1, kc, n);
                Fip = sarr(ic+1, jc+1, kc, n);
                Fjp = quad_interpolate_1D(dx, dxm, dxp, Fi, Fim, Fip);

                dx = cell_y - p.pos(1);
                dxm = dx - dxc[1];
                dxp = dx + dxc[1];

                p.rdata(PIdx::NCoord+n) = quad_interpolate_1D(dx, dxm, dxp, Fj, Fjm, Fjp);
            }
        });
    }

};

#endif
