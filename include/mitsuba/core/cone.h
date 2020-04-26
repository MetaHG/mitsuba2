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
    Cone(const Cone &c): axis(c.axis), normal_angle(c.normal_angle), emission_angle(c.emission_angle) { }

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
        if (!c1.valid() || !c2.valid()) {
            if (!c1.valid()) {
                return Cone(c2);
            }

            return Cone(c1);
        }

        if (c2.normal_angle > c1.normal_angle) {
            return merge(c2, c1);
        }

        Scalar diff_angle = enoki::unit_angle(c1.axis, c2.axis);
        Scalar e_angle = max(c1.emission_angle, c2.emission_angle);

        if (min(diff_angle + c2.normal_angle, math::Pi<Float>) <= c1.normal_angle + std::numeric_limits<float>::epsilon()) {
            // std::cout << "Cone::Merge: Bounds of c1 already covers c2." << std::endl;
            return { c1.axis, c1.normal_angle, e_angle }; // Bounds of c1 already covers c2
        }

        Scalar n_angle = (c1.normal_angle + diff_angle + c2.normal_angle) / 2.0f;

        if (M_PIf32 <= n_angle) {
            // std::cout << "Cone::Merge: Cone covers the sphere." << std::endl;
            return { c1.axis, M_PI, e_angle }; // Cone covers the sphere
        }

        Scalar n_diff_angle = n_angle - c1.normal_angle;
        Vector rotation_axis = cross(c1.axis, c2.axis);

        Vector new_axis;
        if (all(rotation_axis < std::numeric_limits<float>::epsilon())) {
            // Cone vectors were anti-parallel. Pick a random vector in the disk perpendicular to the vectors as rotation axis
            Vector a, b;
            std::tie(a, b) = coordinate_system(c1.axis);
            rotation_axis = a;
        }
        rotation_axis = normalize(rotation_axis);

        new_axis = ScalarTransform4f::rotate(rotation_axis, rad_to_deg(n_diff_angle)) * c1.axis;
        return { new_axis, n_angle, e_angle };
    }

    bool valid() const {
        return any(axis != 0);
    }

    void reset() {
        axis = Vector(0);
        normal_angle = emission_angle = 0;
    }

    Vector axis;
    Scalar normal_angle;
    Scalar emission_angle;
};

template <typename Float, typename Point>
std::ostream &operator<<(std::ostream &os, const Cone<Float, Point> &cone) {
    os << "Cone" << type_suffix<Point>();
    os << "[" << std::endl
       << "  axis = " << cone.axis << "," << std::endl
       << "  normal_angle = " << cone.normal_angle << "," << std::endl
       << "  emission_angle = " << cone.emission_angle << std::endl
       << "]";
    return os;
}

NAMESPACE_END(mitsuba)
