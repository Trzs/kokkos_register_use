#include <Kokkos_Core.hpp>

#define CUDAREAL double

#include "kokkos_types.h"
#include "kernel_math.h"
using vec3 = kokkostbx::vector3<CUDAREAL>;

int main () {

    Kokkos::initialize();
    {
        CUDAREAL phi = 3.6;
        const auto spindle_vector = vector_cudareal_t("spindle", 4);
        Kokkos::copy(spindle_vector, [0, 1, 0, 0]);
        const auto a0 = vector_cudareal_t("a0", 4);
        Kokkos::copy(a0, [0, 1, 2, 3]);
        const auto b0 = vector_cudareal_t("b0", 4);
        Kokkos::copy(b0, [0, 0, 1, 1]);
        const auto c0 = vector_cudareal_t("c0", 4);
        Kokkos::copy(c0, [0, 1, 0, -2]);

        vec3 a0_tmp {a0(1), a0(2), a0(3)};
        vec3 b0_tmp {b0(1), b0(2), b0(3)};
        vec3 c0_tmp {c0(1), c0(2), c0(3)};

        Kokkos::parallel_for("Testkernel1", 1, KOKKOS_LAMBDA(const int idx) {
            vec3 ap_tmp, bp_tmp, cp_tmp;// = a0_tmp.rotate_around_axis(spindle_vector_tmp, phi);
            rotate_axis2(a0_tmp, ap_tmp, spindle_vector_tmp, phi);
            // rotate_axis(b0_tmp, bp_tmp, spindle_vector_tmp, phi);
            // rotate_axis(c0_tmp, cp_tmp, spindle_vector_tmp, phi);
            // vec3 bp_tmp = b0_tmp.rotate_around_axis(spindle_vector_tmp, phi);
            // vec3 cp_tmp = c0_tmp.rotate_around_axis(spindle_vector_tmp, phi);  
            CUDAREAL ap[4] = {0.0, ap_tmp[0], ap_tmp[1], ap_tmp[2]};
            CUDAREAL bp[4];// = {0.0, bp_tmp[0], bp_tmp[1], bp_tmp[2]};;
            CUDAREAL cp[4];// = {0.0, cp_tmp[0], cp_tmp[1], cp_tmp[2]};;
            // rotate_axis(a0, ap, spindle_vector, phi);
            rotate_axis1(b0, bp, spindle_vector, phi);
            rotate_axis1(c0, cp, spindle_vector, phi);
        });
    }
    Kokkos::finalize();

}