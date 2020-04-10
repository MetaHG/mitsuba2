#pragma once

#include <mitsuba/core/bbox.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/vector.h>

#include <mitsuba/render/emitter.h>

#include <filesystem>
#include <fstream>

NAMESPACE_BEGIN(mitsuba)

enum class SplitMethod { SAH, SAOH, Middle, EqualCounts };

MTS_VARIANT class BVH : public Object {
public:
    MTS_IMPORT_TYPES(Emitter)

    BVH(const host_vector<ref<Emitter>, Float> &p, int max_prims_in_node, SplitMethod split_method, bool visualize_volumes = false):
        m_max_prims_in_node(std::min(255, max_prims_in_node)), m_split_method(split_method), m_primitives(p), m_visualize_volumes(visualize_volumes) {

        if (m_primitives.size() == 0) {
            return;
        }

        // std::vector<BVHPrimInfo> prim_info(m_primitives.size());
        std::vector<BVHPrimInfo> prim_info;
        for (size_t i = 0; i < m_primitives.size(); i++) {
            switch (m_split_method) {
            case SplitMethod::SAOH: {
                prim_info.push_back({ i, m_primitives[i]->bbox(), m_primitives[i]->get_total_radiance(), m_primitives[i]->cone() });
                break;
            }
            default: {
                prim_info.push_back({ i, m_primitives[i]->bbox() });
                break;
            }
            }
//            prim_info[i] = {i, m_primitives[i]->bbox() };
        }

        int total_nodes = 0;
        host_vector<ref<Emitter>, Float> ordered_prims;
        BVHNode *root;

        root = recursive_build(prim_info, 0, m_primitives.size(), &total_nodes, ordered_prims, "_");
        m_primitives.swap(ordered_prims);

        m_nodes = new LinearBVHNode[total_nodes]; // TODO: Allocate memory differently?
        m_total_nodes = total_nodes;

        int offset = 0;
        flatten_bvh_tree(root, &offset);
    }

    ~BVH() {
        delete m_nodes;
    }

    std::pair<DirectionSample3f, Spectrum> sample_emitter(const Float &tree_sample, const SurfaceInteraction3f &ref, const Point2f &emitter_sample, const Mask active) {
        float pdf = 1.0f;
        Emitter* emitter = sample_tree(ref, pdf, tree_sample);

        DirectionSample3f ds;
        Spectrum spec;
        std::tie(ds, spec) = emitter->sample_direction(ref, emitter_sample, active);

        ds.pdf *= pdf;
        return std::pair(ds, spec / pdf);
    }

    void to_obj() {
        std::string dir_name = "lighttree_bboxes";

        std::filesystem::path dir(dir_name);
        if (!std::filesystem::exists(dir)) {
            std::filesystem::create_directory(dir);
        }


        for (int i = 0; i < m_total_nodes; i++) {
            std::ostringstream oss;
            oss << dir_name << "/" << i << ".obj";
            std::ofstream ofs(oss.str(), std::ofstream::out);

            ofs << "# Vertices" << std::endl;
            for (size_t j = 0; j < 8; j++) { // Magic number here: TODO DEFINE: 8 = number of corners in bounding box
                Point p = m_nodes[i].bbox.corner(j);
                ofs << "v " << p.x() << " " << p.y() << " " << p.z() << std::endl;
            }

            ofs << std::endl << "# Faces" << std::endl;
            ofs << "f 1 2 4 3" << std::endl;
            ofs << "f 1 5 6 2" << std::endl;
            ofs << "f 5 6 8 7" << std::endl;
            ofs << "f 3 7 8 4" << std::endl;
            ofs << "f 1 5 7 3" << std::endl;
            ofs << "f 6 2 4 8" << std::endl;

            ofs.close();
        }
    }

protected:
    struct BVHPrimInfo {
        BVHPrimInfo(size_t prim_number, const ScalarBoundingBox3f &bbox, Spectrum intensity = 0.f, ScalarCone3f cone = ScalarCone3f()):
            prim_number(prim_number), bbox(bbox), centroid(bbox.center()), intensity(intensity), cone(cone) { }

        size_t prim_number;
        ScalarBoundingBox3f bbox;
        Point3f centroid;
        Spectrum intensity;
        ScalarCone3f cone;
    };

    struct BVHNode {
        void init_leaf(int first, int n, const ScalarBoundingBox3f &box, Spectrum e_intensity, ScalarCone3f cone) {
            first_prim_offset = first;
            prim_count = n;
            bbox = box;
            children[0] = children[1] = nullptr;
            intensity = e_intensity;
            bcone = cone;
        }

        void init_inner(int axis, BVHNode *c0, BVHNode *c1) {
            children[0] = c0;
            children[1] = c1;
            bbox = ScalarBoundingBox3f::merge(c0->bbox, c1->bbox);
            split_axis = axis;
            prim_count = 0;
            intensity = c0->intensity + c1->intensity;
            bcone = ScalarCone3f::merge(c0->bcone, c1->bcone);
        }

        ScalarBoundingBox3f bbox;
        BVHNode *children[2];
        int split_axis;
        int first_prim_offset;
        int prim_count;
        // Emitter related fields
        Spectrum intensity;
        ScalarCone3f bcone;
    };

    struct LinearBVHNode {

        bool is_leaf() {
            return prim_count > 0;
        }

        ScalarBoundingBox3f bbox;
        union {
            int primitives_offset; // leaf
            int second_child_offset; // inner
        };
        uint16_t prim_count; // 0 -> inner node
        uint8_t axis;
        Spectrum intensity;
        ScalarCone3f bcone;
        uint8_t pad[1]; // Padding for memory/cache alignment
    };

protected:
    Emitter* sample_tree(const SurfaceInteraction3f &si, float &importance_ratio, const Float &sample_) {
        importance_ratio = 1.0;
        Float sample(sample_);

        int offset = 0;
        while (!m_nodes[offset].is_leaf()) {
            float w_left, w_right;
            std::tie(w_left, w_right) = compute_children_weights(offset, si);

            float p_left = 0.5f;
            if (w_left + w_right >= std::numeric_limits<float>::epsilon()) {
                p_left = w_left / (w_left + w_right);
            }

            if (sample <= p_left) {
                offset += 1;
                sample = p_left == 0 ? 0 : sample / p_left;
                importance_ratio *= p_left;
            } else {
                offset = m_nodes[offset].second_child_offset;
                float p_right = (1 - p_left);
                sample = (sample - p_left) / p_right;
                importance_ratio *= p_right;
            }
        }

        int leaf_offset = m_nodes[offset].primitives_offset;
        int leaf_prim_count = m_nodes[offset].prim_count;

        int prim_offset = leaf_prim_count * sample;
        if (prim_offset == leaf_prim_count) {
            prim_offset -= 1;
        }

        prim_offset += leaf_offset;
        importance_ratio /= leaf_prim_count;

        return m_primitives[prim_offset].get();
    }

    std::pair<float, float> compute_children_weights(int offset, const SurfaceInteraction3f &ref) {
        LinearBVHNode ln = m_nodes[offset + 1];
        LinearBVHNode rn = m_nodes[m_nodes[offset].second_child_offset];

        float l_weight = compute_cone_weight(&ln, ref);
        float r_weight = compute_cone_weight(&rn, ref);

        l_weight *= compute_luminance(ln.intensity);
        r_weight *= compute_luminance(rn.intensity);

        float left_d = ln.bbox.distance(ref.p);
        float right_d = rn.bbox.distance(ref.p);

        float distance_ratio = 1.0f; //TODO: DEFINE IT
        if (left_d <= distance_ratio * norm(ln.bbox.extents())
            || right_d <= distance_ratio * norm(rn.bbox.extents())) {
            return std::pair(l_weight, r_weight);
        }

        return std::pair(l_weight / (left_d * left_d), r_weight / (right_d * right_d));
    }

    BVHNode* recursive_build(std::vector<BVHPrimInfo> &primitive_info,
                             int start,
                             int end,
                             int *total_nodes,
                             host_vector<ref<Emitter>, Float> &ordered_prims,
                             std::string node_name = "") {
        BVHNode* node;
        (*total_nodes)++;

        ScalarBoundingBox3f node_bbox = ScalarBoundingBox3f();
        ScalarBoundingBox3f centroid_bbox = ScalarBoundingBox3f();
        Spectrum node_intensity = 0.f;
        ScalarCone3f node_cone = ScalarCone3f();

        for (int i = start; i < end; i++) {
            node_bbox.expand(primitive_info[i].bbox);
            centroid_bbox.expand(primitive_info[i].centroid);
            node_intensity += primitive_info[i].intensity;
            node_cone = ScalarCone3f::merge(node_cone, primitive_info[i].cone);
        }

        if (m_visualize_volumes) {
            save_to_obj(node_name, node_bbox, node_cone);
        }

        int nb_prim = end - start;
        if (nb_prim == 1) {
            node = create_leaf(primitive_info, start, end, ordered_prims, node_bbox, node_intensity, node_cone);
        } else {
            if (all(eq(centroid_bbox.min, centroid_bbox.max))) {
                node = create_leaf(primitive_info, start, end, ordered_prims, node_bbox, node_intensity, node_cone);
            } else {
                int mid = (start + end) / 2;
                int dim = centroid_bbox.major_axis();

                switch (m_split_method) {
                case SplitMethod::Middle: {
                    Float p_mid = centroid_bbox.center()[dim];
                    BVHPrimInfo *mid_ptr = std::partition(&primitive_info[start],
                                                         &primitive_info[end - 1] + 1,
                                                         [dim, p_mid](const BVHPrimInfo &pi) {
                                                                return pi.centroid[dim] < p_mid;
                                                         });
                    mid = mid_ptr - &primitive_info[0];
                    if (mid != start && mid != end) {
                        break;
                    }
                }

                case SplitMethod::EqualCounts: {
                    mid = (start + end) / 2;
                    std::nth_element(&primitive_info[start], &primitive_info[mid], &primitive_info[end-1] + 1,
                            [dim](const BVHPrimInfo &a, const BVHPrimInfo &b) {
                        return a.centroid[dim] < b.centroid[dim];
                    });
                    break;
                }

                case SplitMethod::SAH:
                case SplitMethod::SAOH:
                default: {
                    if (m_split_method != SplitMethod::SAOH && nb_prim <= 4) { //TODO: Define this threshold
                        mid = (start + end) / 2;
                        std::nth_element(&primitive_info[start], &primitive_info[mid], &primitive_info[end-1] + 1,
                                [dim](const BVHPrimInfo &a, const BVHPrimInfo &b) {
                            return a.centroid[dim] < b.centroid[dim];
                        });

                        break;
                    }

                    constexpr int nb_buckets = 12;
                    int min_cost_split_bucket;
                    Float min_cost;
                    find_split(primitive_info, start, end, centroid_bbox, node_bbox, node_cone,
                                    nb_buckets, dim, min_cost_split_bucket, min_cost);

                    Float leaf_cost;
                    if (m_split_method == SplitMethod::SAOH) {
                        leaf_cost = compute_luminance(node_intensity);
                    } else {
                        leaf_cost = nb_prim;
                    }

                    std::cout << "Min cost: " << min_cost << ", Leaf cost: " << leaf_cost << std::endl;
                    if (min_cost < leaf_cost || (m_split_method != SplitMethod::SAOH && nb_prim > m_max_prims_in_node)) {
                        BVHPrimInfo *p_mid = std::partition(&primitive_info[start], &primitive_info[end-1] + 1,
                                [=](const BVHPrimInfo &pi) {
                            int b = nb_buckets * centroid_bbox.offset(pi.centroid)[dim];
                            if (b == nb_buckets) {
                                b = nb_buckets - 1;
                            }

                            return b <= min_cost_split_bucket;
                        });

                        mid = p_mid - &primitive_info[0];
                    } else {
                        node = create_leaf(primitive_info, start, end, ordered_prims, node_bbox, node_intensity, node_cone);
                        return node;
                    }

                    break;
                }
                }

                node = new BVHNode();
                node->init_inner(dim,
                                 recursive_build(primitive_info, start, mid, total_nodes, ordered_prims, node_name + "l"),
                                 recursive_build(primitive_info, mid, end, total_nodes, ordered_prims, node_name + "r"));
            }
        }

        return node;
    }

    int flatten_bvh_tree(BVHNode *node, int *offset) {
        LinearBVHNode *linear_node = &m_nodes[*offset];
        linear_node->bbox = node->bbox;
        linear_node->bcone = node->bcone;
        linear_node->intensity = node->intensity;
        int my_offset = (*offset)++;

        if (node->prim_count > 0) {
            linear_node->primitives_offset = node->first_prim_offset;
            linear_node->prim_count = node->prim_count;
        } else {
            linear_node->axis = node->split_axis;
            linear_node->prim_count = 0;
            flatten_bvh_tree(node->children[0], offset);
            linear_node->second_child_offset = flatten_bvh_tree(node->children[1], offset);
        }

        return my_offset;
    }

private:
    void find_split(std::vector<BVHPrimInfo> &primitive_info, int start, int end,
                    ScalarBoundingBox3f &centroid_bbox, ScalarBoundingBox3f &node_bbox, ScalarCone3f &node_cone,
                    int nb_buckets, int &split_dim, int &split_bucket, Float &min_cost) {

        split_dim = centroid_bbox.major_axis();
        split_bucket = 0;
        min_cost = std::numeric_limits<Float>::max();

        for (int dim = 0; dim < 3; dim++) {
            struct BucketInfo {
                BucketInfo() {
                    count = 0;
                    bbox = ScalarBoundingBox3f();
                    intensity = 0.f;
                    cone = ScalarCone3f();
                }

                std::string to_string() const {
                    std::ostringstream oss;
                    oss << "BucketInfo[" << std::endl
                        << "  count = " << count << "," << std::endl
                        << "  bbox = " << bbox << "," << std::endl
                        << "  intensity = " << intensity << "," << std::endl
                        << "  cone = " << cone << std::endl
                        << "]";
                    return oss.str();
                }

                int count;
                ScalarBoundingBox3f bbox;
                Spectrum intensity;
                ScalarCone3f cone;
            };

            BucketInfo buckets[nb_buckets];

            for (int i = start; i < end; i++) {
                int b = nb_buckets * centroid_bbox.offset(primitive_info[i].centroid)[dim];
                if (b == nb_buckets) {
                    b = nb_buckets - 1;
                }

                buckets[b].count++;
                buckets[b].bbox.expand(primitive_info[i].bbox);

                if (m_split_method == SplitMethod::SAOH) {
                    buckets[b].intensity += primitive_info[i].intensity;
                    buckets[b].cone = ScalarCone3f::merge(buckets[b].cone, primitive_info[i].cone);
                }
            }


            Float cost[nb_buckets - 1];
            for (int i = 0; i < nb_buckets - 1; i++) {
                ScalarBoundingBox3f b0 = ScalarBoundingBox3f(), b1 = ScalarBoundingBox3f();
                int count0 = 0, count1 = 0;
                Spectrum i0 = 0.f, i1 = 0.f;
                ScalarCone3f c0 = ScalarCone3f(), c1 = ScalarCone3f();

                for (int j = 0; j <= i; j++) {
                    b0.expand(buckets[j].bbox);
                    count0 += buckets[j].count;

                    if (m_split_method == SplitMethod::SAOH) {
                        i0 += buckets[j].intensity;
                        c0 = ScalarCone3f::merge(c0, buckets[j].cone);
                    }
                }

                for (int j = i+1; j < nb_buckets; j++) {
                    b1.expand(buckets[j].bbox);
                    count1 += buckets[j].count;

                    if (m_split_method == SplitMethod::SAOH) {
                        i1 += buckets[j].intensity;
                        c1 = ScalarCone3f::merge(c1, buckets[j].cone);
                    }
                }

                if (m_split_method == SplitMethod::SAOH) {
                    if (node_bbox.surface_area() < std::numeric_limits<float>::epsilon()) {
                        //cost[i] = squared_norm(node_bbox.extents()) * (compute_luminance(i0) * c0.surface_area() + compute_luminance(i1) * c1.surface_area()) / node_cone.surface_area();
                        cost[i] = (compute_luminance(i0) * c0.surface_area() + compute_luminance(i1) * c1.surface_area()) / node_cone.surface_area();
                    } else {
                        cost[i] = (compute_luminance(i0) * b0.surface_area() * c0.surface_area() // TODO: Need to add regularizer from paper?
                                   + compute_luminance(i1) * b1.surface_area() * c1.surface_area())
                                / (node_bbox.surface_area() * node_cone.surface_area());

//                            cost[i] = (compute_luminance(i0) * squared_norm(b0.extents()) * c0.surface_area()
//                                       + compute_luminance(i1) * squared_norm(b1.extents() * c1.surface_area()));
                    }
                } else {
                    // m_split_method == SplitMethod::SAH
                    cost[i] = 0.125f + (count0 * b0.surface_area() + // TODO: Define this cost
                                        count1 * b1.surface_area()) / node_bbox.surface_area();
                }

            }

            for (int i = 1; i < nb_buckets - 1; i++) {
                if (cost[i] < min_cost) {
                    min_cost = cost[i];
                    split_bucket = i;
                    split_dim = dim;
                }
            }
        }
    }

    float compute_cone_weight(LinearBVHNode *node, const SurfaceInteraction3f &si){
        ScalarVector3f p_to_box_center = node->bbox.center() - si.p;

        float in_angle = acos(dot(normalize(p_to_box_center), si.n));

        float bangle = node->bbox.solid_angle(si.p);

        float min_in_angle = max(in_angle - bangle, 0);

        float caxis_p_angle = acos(dot(normalize(node->bcone.axis), normalize(-p_to_box_center)));

        float min_e_angle = max(caxis_p_angle - node->bcone.normal_angle - bangle, 0);

        float cone_weight = 0;

        if (min_e_angle < node->bcone.emission_angle) {
            cone_weight = abs(cos(min_in_angle)) * cos(min_e_angle);
        }

        return cone_weight;
    }

    BVHNode* create_leaf(std::vector<BVHPrimInfo> &primitive_info,
                         int start,
                         int end,
                         host_vector<ref<Emitter>, Float> &ordered_prims,
                         ScalarBoundingBox3f &prims_bbox,
                         Spectrum intensity = 0.f,
                         ScalarCone3f cone = ScalarCone3f()) {
        BVHNode *leaf = new BVHNode();

        int nb_prim = end - start;
        int first_prim_offset = ordered_prims.size();
        for (int i = start; i < end; i++) {
            int prim_num = primitive_info[i].prim_number;
            ordered_prims.push_back(m_primitives[prim_num]);
        }

        leaf->init_leaf(first_prim_offset, nb_prim, prims_bbox, intensity, cone);

        return leaf;
    }

    static float compute_luminance(Spectrum intensity) {
        return hmean(intensity);
    }

    void save_to_obj(std::string node_name, ScalarBoundingBox3f bbox, ScalarCone3f cone) {
        std::string dir_name = "lighttree_bboxes";

        std::filesystem::path dir(dir_name);
        if (!std::filesystem::exists(dir)) {
            std::filesystem::create_directory(dir);
        }

        std::ostringstream oss;
        oss << dir_name << "/" << node_name << ".obj";
        std::ofstream ofs(oss.str(), std::ofstream::out);

        ofs << "# Vertices" << std::endl;
        for (size_t j = 0; j < 8; j++) { // Magic number here: TODO DEFINE: 8 = number of corners in bounding box
            Point p = bbox.corner(j);
            ofs << "v " << p.x() << " " << p.y() << " " << p.z() << std::endl;
        }

        ofs << std::endl << "# Faces" << std::endl;
        ofs << "f 1 2 4 3" << std::endl;
        ofs << "f 1 5 6 2" << std::endl;
        ofs << "f 5 6 8 7" << std::endl;
        ofs << "f 3 7 8 4" << std::endl;
        ofs << "f 1 5 7 3" << std::endl;
        ofs << "f 6 2 4 8" << std::endl;

        ofs.close();

        std::ostringstream oss_cone;
        oss_cone << dir_name << "/cone" << node_name << ".obj";
        cone_to_obj(oss_cone.str(), bbox.center(), cone);
    }

    void cone_to_obj(std::string filename, ScalarPoint3f center, ScalarCone3f cone) {
        std::ofstream ofs(filename, std::ofstream::out);

        ofs << "# Vertices" << std::endl;
        // Center
        ofs << obj_vertex(center);

        // Circle
        ScalarVector3f a, b;
        std::tie(a, b) = coordinate_system(cone.axis);
        Point circle_center = center;
        Float scale_factor = tan(cone.normal_angle);

        float cone_scale_factor = 0.5f; // TODO: DEFINE
        if (cone.normal_angle < M_PI_2f32 - std::numeric_limits<float>::epsilon()) {
            float cos_scale_factor = cos(cone.normal_angle) * cone_scale_factor;
            a *= scale_factor * cos_scale_factor;
            circle_center += cone.axis * cos_scale_factor;
        }

        if (scale_factor < std::numeric_limits<float>::epsilon()) {
            a = b * cone_scale_factor;
        }

        if (cone.normal_angle == 0) {
            ofs << obj_vertex(circle_center);
            ofs << "l 1 2" << std::endl;
            return;
        }

        int nb_section = 12;
        float section = 2 * M_PI / nb_section;
        for (int i = 0; i < nb_section; i++) {
            Point p = circle_center + ScalarTransform4f::rotate(cone.axis, rad_to_deg(i * section)) * a;
            ofs << obj_vertex(p);
        }

        ofs << std::endl << "# Faces" << std::endl;

        for (int i = 2; i < nb_section + 1; i++) {
            ofs << "f 1 " << i << " " << i+1 << std::endl;
        }

        ofs << "f 1 " << nb_section + 1 << " 2" << std::endl;

        ofs.close();
    }

    std::string obj_vertex(ScalarPoint3f p) {
        std::ostringstream oss;
        oss << "v " << p.x() << " " << p.y() << " " << p.z() << std::endl;
        return oss.str();
    }

private:
    const int m_max_prims_in_node;
    const SplitMethod m_split_method;
    host_vector<ref<Emitter>, Float> m_primitives;
    LinearBVHNode *m_nodes = nullptr;
    int m_total_nodes;
    bool m_visualize_volumes;
};

NAMESPACE_END(mitsuba)
