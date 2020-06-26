#include <iostream>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_BCRec.H>
#include <AMReX_BCUtil.H>
#include <AMReX_PlotFileUtil.H>

#include "TransmogrifierContainer.H"
#include "Parameters.H"

using namespace amrex;

void transmogrify(const TransmogrifyParams& parms)
{
    /* Common to both MultiFabs */

    // We don't need boundary conditions so periodicity won't matter
    Vector<int> is_periodic(AMREX_SPACEDIM, 0);

    // We don't need ghost cells (grids are "grown" by ngrow ghost cells in each direction)
    const int ngrow = 0;
    constexpr int ncomp = 1;

    /* Read input "A" data */
    PlotFileData input_pfdata(parms.input_plotfile);

    const int lev = 0;

    // Get BoxArray and DistributionMapping
    const BoxArray ba_A = input_pfdata.boxArray(lev);
    const DistributionMapping dm_A = input_pfdata.DistributionMap(lev);

    // Get the Geometry
    const Box domain_A = input_pfdata.probDomain(lev);
    const auto problem_lo_A = input_pfdata.probLo();
    const auto problem_hi_A = input_pfdata.probHi();

    // Assumes A data is from 2D domain
    const RealBox real_box_A({AMREX_D_DECL(problem_lo_A[0], problem_lo_A[1], 0.0)},
                             {AMREX_D_DECL(problem_hi_A[0], problem_hi_A[1], 1.0)});

    const auto coord_A = input_pfdata.coordSys();

    const Geometry geom_A(domain_A, &real_box_A, coord_A, is_periodic.data());
    const Geometry geom_A_cart(domain_A, &real_box_A, CoordSys::cartesian, is_periodic.data());

    MultiFab state_A = input_pfdata.get(lev);

    // To check that we read it correctly, write it out as a 3D cartesian plotfile
    amrex::WriteSingleLevelPlotfile("data_A", state_A, {"phi"}, geom_A_cart, 0.0, 0);

    /* Create the "B" MultiFab we are filling */

    // Define the index space of the domain
    const IntVect domain_lo_B(AMREX_D_DECL(0, 0, 0));
    const IntVect domain_hi_B(AMREX_D_DECL(parms.ncell_B[0]-1,parms.ncell_B[1]-1,parms.ncell_B[2]-1));
    const Box domain_B(domain_lo_B, domain_hi_B);

    // Initialize the boxarray "ba" from the single box "domain"
    BoxArray ba_B(domain_B);

    // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
    ba_B.maxSize(parms.max_grid_size_B);

    // This defines the physical box, [0,1] in each dimension
    RealBox real_box_B({AMREX_D_DECL(parms.xlo_B, parms.ylo_B, parms.zlo_B)},
                       {AMREX_D_DECL(parms.xhi_B, parms.yhi_B, parms.zhi_B)});

    // This defines the domain Geometry as Cartesian coordinates
    Geometry geom_B(domain_B, &real_box_B, CoordSys::cartesian, is_periodic.data());

    // Create the DistributionMapping from the BoxArray
    DistributionMapping dm_B(ba_B);

    // Create a MultiFab to hold our grid state data and initialize to 0.0
    MultiFab state_B(ba_B, dm_B, ncomp, ngrow);

    /* Create TransmogrifierContainer */

    // Pass grid metadata for A & B
    TransmogrifierContainer<ncomp> tpc(geom_A, dm_A, ba_A,
                                       geom_B, dm_B, ba_B);

    // Interpolate from A to B
    tpc.Interpolate(state_A, state_B);

    /* Write B data */
    amrex::WriteSingleLevelPlotfile("data_B", state_B, {"phi"}, geom_B, 0.0, 0);

    if (parms.write_particles)
    {
        tpc.Checkpoint("data_B", "transmogrifier", true, {"phi"});
    }
}

void generate_test_data(const TransmogrifyParams& parms)
{
    // We don't need boundary conditions so periodicity won't matter
    Vector<int> is_periodic(AMREX_SPACEDIM, 0);

    // We don't need ghost cells (grids are "grown" by ngrow ghost cells in each direction)
    const int ngrow = 0;
    constexpr int ncomp = 1;

    /* Create the "A" MultiFab we are filling from */

    // Define the index space of the domain
    const IntVect domain_lo_A(AMREX_D_DECL(0, 0, 0));
    const IntVect domain_hi_A(AMREX_D_DECL(parms.ncell_A[0]-1,parms.ncell_A[1]-1,parms.ncell_A[2]-1));
    const Box domain_A(domain_lo_A, domain_hi_A);

    // Initialize the boxarray "ba" from the single box "domain"
    BoxArray ba_A(domain_A);

    // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
    ba_A.maxSize(parms.max_grid_size_A);

    // This defines the physical box, [0,1] in each dimension
    RealBox real_box_A({AMREX_D_DECL(parms.xlo_A, parms.ylo_A, parms.zlo_A)},
                       {AMREX_D_DECL(parms.xhi_A, parms.yhi_A, parms.zhi_A)});

    // This defines the domain Geometry as R-Z cylindrical coordinates
    Geometry geom_A(domain_A, &real_box_A, CoordSys::RZ, is_periodic.data());

    // Create the DistributionMapping from the BoxArray
    DistributionMapping dm_A(ba_A);

    // Create a MultiFab to hold our grid state data and initialize to 0.0
    MultiFab state_A(ba_A, dm_A, ncomp, ngrow);

    /* Initialize MultiFab A with disc test data. */

    for (MFIter mfi(state_A, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        auto sarr = state_A.array(mfi);

        const auto plo  = geom_A.ProbLoArray();
        const auto phi  = geom_A.ProbHiArray();

        const auto dbox = geom_A.Domain();
        const auto dlo  = amrex::lbound(dbox);
        const auto dxc  = geom_A.CellSizeArray();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            Real x = (i - dlo.x + 0.5) * dxc[0] + plo[0];

            // initialize 1 if r < 0.5 * (Rlo + Rhi) and -1 otherwise
            Real rhalf = 0.5 * (plo[0] + phi[0]);
            sarr(i, j, k) = (x < rhalf) ? 1.0 : -1.0;
        });
    }

    /* Write A data */
    amrex::WriteSingleLevelPlotfile(parms.input_plotfile, state_A, {"phi"}, geom_A, 0.0, 0);
}

void main_main()
{
    // Read the input parameters
    TransmogrifyParams parms;

    if (parms.write_cylindrical_test) {
        // Assert we compiled in 2D
        AMREX_ASSERT(AMREX_SPACEDIM == 2);

        // Make 2D cylindrical test dataset
        generate_test_data(parms);
    } else {
        // Read input dataset and transmogrify it to 3D
        transmogrify(parms);
    }
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    {
        main_main();
    }

    amrex::Finalize();
}
