#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <AMReX_REAL.H>
#include <AMReX_IntVect.H>

using namespace amrex;

struct TransmogrifyParams
{
    IntVect ncell_A, ncell_B;
    Real xlo_A, xlo_B, xhi_A, xhi_B;
    Real ylo_A, ylo_B, yhi_A, yhi_B;
    Real zlo_A, zlo_B, zhi_A, zhi_B;
    int max_grid_size_A, max_grid_size_B;
    int write_particles;
    int write_cylindrical_test;
    std::string input_plotfile;

    TransmogrifyParams()
    {
        ParmParse pp;

        write_particles = 0;
        pp.query("write_particles", write_particles);

        write_cylindrical_test = 0;
        pp.query("write_cylindrical_test", write_cylindrical_test);

        input_plotfile = "data_A";
        pp.query("input_plotfile", input_plotfile);

        pp.get("ncell_A", ncell_A);
        pp.get("xlo_A", xlo_A);
        pp.get("ylo_A", ylo_A);
        pp.get("zlo_A", zlo_A);
        pp.get("xhi_A", xhi_A);
        pp.get("yhi_A", yhi_A);
        pp.get("zhi_A", zhi_A);
        pp.get("max_grid_size_A", max_grid_size_A);

        pp.get("ncell_B", ncell_B);
        pp.get("xlo_B", xlo_B);
        pp.get("ylo_B", ylo_B);
        pp.get("zlo_B", zlo_B);
        pp.get("xhi_B", xhi_B);
        pp.get("yhi_B", yhi_B);
        pp.get("zhi_B", zhi_B);
        pp.get("max_grid_size_B", max_grid_size_B);
    }
};

#endif
