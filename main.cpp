#include <Kokkos_Core.hpp>

#define CUDAREAL double

#include "kokkos_types.h"
#include "kernel_math.h"
using vec3 = kokkostbx::vector3<CUDAREAL>;

void init(vector_cudareal_t& spindle,
          vector_cudareal_t& A, 
          vector_cudareal_t& B, 
          vector_cudareal_t& C) {
    Kokkos::parallel_for("InitA", 1, KOKKOS_LAMBDA(const int idx) {
        spindle(1) = 1;
        spindle(2) = 0;
        spindle(3) = 0;

        A(1) = 1;
        A(2) = 2;
        A(3) = 3;

        B(1) = 1;
        B(2) = 0;
        B(3) = 1;

        C(1) = 0;
        C(2) = 1;
        C(3) = -2;
    });
}

void test_kernel1(const vector_cudareal_t& spindle,
                  const vector_cudareal_t& A, 
                  const vector_cudareal_t& B, 
                  const vector_cudareal_t& C,
                  const CUDAREAL phi) {
    
    Kokkos::parallel_for("Testkernel1", 1, KOKKOS_LAMBDA(const int idx) {
        CUDAREAL ap[4];
        CUDAREAL bp[4];
        CUDAREAL cp[4];
        rotate_axis1(A, ap, spindle, phi);
        rotate_axis1(B, bp, spindle, phi);
        rotate_axis1(C, cp, spindle, phi);
    });    
}

void test_kernel2(const vec3& spindle,
                  const vec3& A, 
                  const vec3& B, 
                  const vec3& C,
                  const CUDAREAL phi) {
    
    Kokkos::parallel_for("Testkernel2", 1, KOKKOS_LAMBDA(const int idx) {
        vec3 ap_tmp, bp_tmp, cp_tmp;
        rotate_axis2(A, ap_tmp, spindle, phi);
        rotate_axis2(B, bp_tmp, spindle, phi);
        rotate_axis2(C, cp_tmp, spindle, phi);
        CUDAREAL ap[4] = {0.0, ap_tmp[0], ap_tmp[1], ap_tmp[2]};
        CUDAREAL bp[4] = {0.0, bp_tmp[0], bp_tmp[1], bp_tmp[2]};;
        CUDAREAL cp[4] = {0.0, cp_tmp[0], cp_tmp[1], cp_tmp[2]};;
    });    
}

int main () {

    Kokkos::initialize();
    {
        CUDAREAL phi = 3.6;
        const auto spindle_vector = vector_cudareal_t("spindle", 4);
        const auto a0 = vector_cudareal_t("a0", 4);
        const auto b0 = vector_cudareal_t("b0", 4);
        const auto c0 = vector_cudareal_t("c0", 4);
        init(spindle_vector, a0, b0, c0);

        vec3 spindle_vector_tmp {spindle_vector(1), spindle_vector(2), spindle_vector(3)};
        vec3 a0_tmp {a0(1), a0(2), a0(3)};
        vec3 b0_tmp {b0(1), b0(2), b0(3)};
        vec3 c0_tmp {c0(1), c0(2), c0(3)};

        test_kernel1(spindle_vector, a0, b0, c0, phi);
        test_kernel2(spindle_vector_tmp, a0_tmp, b0_tmp, c0_tmp, phi);
    }
    Kokkos::finalize();

}