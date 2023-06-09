#include <cstdio>
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
    
    const CUDAREAL sinphi = sin(phi);
    const CUDAREAL cosphi = cos(phi);
    Kokkos::parallel_for("Testkernel1", 1, KOKKOS_LAMBDA(const int idx) {
        CUDAREAL ap[4];
        CUDAREAL bp[4];
        CUDAREAL cp[4];
        rotate_axis1(A, ap, spindle, sinphi, cosphi);
        rotate_axis1(B, bp, spindle, sinphi, cosphi);
        rotate_axis1(C, cp, spindle, sinphi, cosphi);
        std::printf("----- KERNEL 1 -----\n");
        std::printf("Ap = [%f, %f, %f]\n", ap[1], ap[2], ap[3]);
        std::printf("Bp = [%f, %f, %f]\n", bp[1], bp[2], bp[3]);
        std::printf("Cp = [%f, %f, %f]\n", cp[1], cp[2], cp[3]);
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
        std::printf("----- KERNEL 2 -----\n");
        std::printf("Ap = [%f, %f, %f]\n", ap_tmp[0], ap_tmp[1], ap_tmp[2]);
        std::printf("Bp = [%f, %f, %f]\n", bp_tmp[0], bp_tmp[1], bp_tmp[2]);
        std::printf("Cp = [%f, %f, %f]\n", cp_tmp[0], cp_tmp[1], cp_tmp[2]);
    });    
}

void test_kernel3(const vec3& spindle,
                  const vec3& A, 
                  const vec3& B, 
                  const vec3& C,
                  const CUDAREAL phi) {

    const CUDAREAL sinphi = sin(phi);
    const CUDAREAL cosphi = cos(phi);    
    Kokkos::parallel_for("Testkernel3", 1, KOKKOS_LAMBDA(const int idx) {
        vec3 ap_tmp, bp_tmp, cp_tmp;
        rotate_axis3(A, ap_tmp, spindle, sinphi, cosphi);
        rotate_axis3(B, bp_tmp, spindle, sinphi, cosphi);
        rotate_axis3(C, cp_tmp, spindle, sinphi, cosphi);
        const CUDAREAL ap[4] = {0.0, ap_tmp[0], ap_tmp[1], ap_tmp[2]};
        const CUDAREAL bp[4] = {0.0, bp_tmp[0], bp_tmp[1], bp_tmp[2]};
        const CUDAREAL cp[4] = {0.0, cp_tmp[0], cp_tmp[1], cp_tmp[2]};
        std::printf("----- KERNEL 3 -----\n");
        std::printf("Ap = [%f, %f, %f]\n", ap[1], ap[2], ap[3]);
        std::printf("Bp = [%f, %f, %f]\n", bp[1], bp[2], bp[3]);
        std::printf("Cp = [%f, %f, %f]\n", cp[1], cp[2], cp[3]);

    });    
}

int main () {

    Kokkos::initialize();
    {
        CUDAREAL phi = 3.6;
        auto spindle_vector = vector_cudareal_t("spindle", 4);
        auto a0 = vector_cudareal_t("a0", 4);
        auto b0 = vector_cudareal_t("b0", 4);
        auto c0 = vector_cudareal_t("c0", 4);
        init(spindle_vector, a0, b0, c0);

        vec3 spindle_vector_tmp {spindle_vector(1), spindle_vector(2), spindle_vector(3)};
        vec3 a0_tmp {a0(1), a0(2), a0(3)};
        vec3 b0_tmp {b0(1), b0(2), b0(3)};
        vec3 c0_tmp {c0(1), c0(2), c0(3)};

        test_kernel1(spindle_vector, a0, b0, c0, phi);
        test_kernel2(spindle_vector_tmp, a0_tmp, b0_tmp, c0_tmp, phi);
        test_kernel3(spindle_vector_tmp, a0_tmp, b0_tmp, c0_tmp, phi);
    }
    Kokkos::finalize();

}