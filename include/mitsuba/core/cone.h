#pragma once

#include <mitsuba/core/vector.h>
#include <mitsuba/core/transform.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float_, typename Point_> struct Cone {
    using Float = Float_;
    MTS_IMPORT_CORE_TYPES()

    static constexpr size_t Dimension = array_size_v<Point_>;
    using Point = Point_;
    using Value = value_t<Point>;
    using Scalar = scalar_t<Point>;
    using Vector = typename Point::Vector;

    Cone() { reset(); }
    Cone(const Vector &v): axis(v), normal_angle(0), emission_angle(0) { }
    Cone(const Vector &v, const Scalar &n_angle, const Scalar &e_angle):
        axis(v), normal_angle(n_angle), emission_angle(e_angle) { }

    Scalar surface_area() {
        Scalar total_angle = min(normal_angle + emission_angle, M_PIf32);
        Scalar cos_n_angle = cos(normal_angle);
        Scalar sin_n_angle = sin(normal_angle);

        return 2 * M_PIf32 * (1.0f - cos_n_angle)
                + M_PI_2f32 * (2 * total_angle * sin_n_angle
                            - cos(normal_angle - 2 * total_angle)
                            - 2 * normal_angle * sin_n_angle
                            + cos_n_angle);
    }

    static Cone merge(const Cone &c1, const Cone &c2) {
        Cone c1_tmp = { c1.axis, c1.normal_angle, c1.emission_angle };
        Cone c2_tmp = { c2.axis, c2.normal_angle, c2.emission_angle };

        if (c2_tmp.normal_angle > c1_tmp.normal_angle) {
            std::swap(c1_tmp, c2_tmp);
        }

        Scalar diff_angle = acos(dot(c1_tmp.axis, c2_tmp.axis));
        Scalar e_angle = max(c1_tmp.emission_angle, c2_tmp.emission_angle);

        if (min(diff_angle + c2_tmp.emission_angle, M_PI)) {
            return { c1_tmp.axis, c1_tmp.normal_angle, e_angle }; // Bounds of c1 already covers c2
        }

        Scalar n_angle = (c1_tmp.normal_angle + diff_angle + c2_tmp.normal_angle) / 2.0f;
        if (M_PIf32 <= n_angle) {
            return { c1_tmp.normal_angle, M_PI, e_angle }; // Cone covers the sphere
        }

        Scalar n_diff_angle = n_angle - c1_tmp.normal_angle;
        Vector rotation_axis = cross(c1_tmp.axis, c2_tmp.axis);
        Vector new_axis = ScalarTransform4f::rotate(rotation_axis, n_diff_angle) * c1_tmp.axis; // TODO: Verify this works as expected
        return { new_axis, n_angle, e_angle };
    }

    void reset() {
        axis = Vector();
        normal_angle = emission_angle = 0;
    }

    Vector axis;
    Scalar normal_angle;
    Scalar emission_angle;
};

NAMESPACE_END(mitsuba)
