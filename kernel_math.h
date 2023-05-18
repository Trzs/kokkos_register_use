#include "kokkos_vector3.h"

#ifndef CUDAREAL
#define CUDAREAL float
#endif

using kokkostbx::vector3;

KOKKOS_INLINE_FUNCTION static CUDAREAL *rotate_axis1(const vector_cudareal_t v, CUDAREAL * newv, const vector_cudareal_t axis, const CUDAREAL phi);
KOKKOS_INLINE_FUNCTION static vector3<CUDAREAL> rotate_axis2(const vector3<CUDAREAL>& v, vector3<CUDAREAL>& newv, const vector3<CUDAREAL>& axis, const CUDAREAL phi);
KOKKOS_INLINE_FUNCTION static vector3<CUDAREAL> rotate_axis3(const vector3<CUDAREAL>& v, vector3<CUDAREAL>& newv, const vector3<CUDAREAL>& axis, const CUDAREAL phi);


/* rotate a point about a unit vector axis */
KOKKOS_INLINE_FUNCTION CUDAREAL *rotate_axis1(const vector_cudareal_t v, CUDAREAL * newv, const vector_cudareal_t axis, const CUDAREAL phi) {

        const CUDAREAL sinphi = sin(phi);
        const CUDAREAL cosphi = cos(phi);
        const CUDAREAL a1 = axis(1);
        const CUDAREAL a2 = axis(2);
        const CUDAREAL a3 = axis(3);
        const CUDAREAL v1 = v(1);
        const CUDAREAL v2 = v(2);
        const CUDAREAL v3 = v(3);
        const CUDAREAL dot = (a1 * v1 + a2 * v2 + a3 * v3) * (1.0 - cosphi);

        newv[1] = a1 * dot + v1 * cosphi + (-a3 * v2 + a2 * v3) * sinphi;
        newv[2] = a2 * dot + v2 * cosphi + (+a3 * v1 - a1 * v3) * sinphi;
        newv[3] = a3 * dot + v3 * cosphi + (-a2 * v1 + a1 * v2) * sinphi;

        return newv;
}

/* rotate a point about a unit vector axis */
KOKKOS_INLINE_FUNCTION vector3<CUDAREAL> rotate_axis2(const vector3<CUDAREAL>& v, vector3<CUDAREAL>& newv, const vector3<CUDAREAL>& axis, const CUDAREAL phi) {

        const CUDAREAL sinphi = sin(phi);
        const CUDAREAL cosphi = cos(phi);
        const CUDAREAL a1 = axis[0];
        const CUDAREAL a2 = axis[1];
        const CUDAREAL a3 = axis[2];
        const CUDAREAL v1 = v[0];
        const CUDAREAL v2 = v[1];
        const CUDAREAL v3 = v[2];
        const CUDAREAL dot = (a1 * v1 + a2 * v2 + a3 * v3) * (1.0 - cosphi);

        newv[0] = a1 * dot + v1 * cosphi + (-a3 * v2 + a2 * v3) * sinphi;
        newv[1] = a2 * dot + v2 * cosphi + (+a3 * v1 - a1 * v3) * sinphi;
        newv[2] = a3 * dot + v3 * cosphi + (-a2 * v1 + a1 * v2) * sinphi;

        return newv;
}

/* rotate a point about a unit vector axis */
KOKKOS_INLINE_FUNCTION vector3<CUDAREAL> rotate_axis3(const vector3<CUDAREAL>& v, vector3<CUDAREAL>& newv, const vector3<CUDAREAL>& axis, const CUDAREAL phi) {

        const CUDAREAL sinphi = sin(phi);
        const CUDAREAL cosphi = cos(phi);
        const CUDAREAL dot = axis.dot(v) * (1.0 - cosphi);

        newv = axis * dot + v * cosphi + axis.cross(v) * sinphi;
        return newv;
}