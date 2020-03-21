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

    BVH(const host_vector<ref<Emitter>, Float> &p, int max_prims_in_node, SplitMethod split_method):
        m_max_prims_in_node(std::min(255, max_prims_in_node)), m_split_method(split_method), m_primitives(p) {

        if (m_primitives.size() == 0) {
            return;
        }

        // std::vector<BVHPrimInfo> prim_info(m_primitives.size());
        std::vector<BVHPrimInfo> prim_info;
        for (size_t i = 0; i < m_primitives.size(); i++) {
            prim_info.push_back({ i, m_primitives[i]->bbox() });
//            prim_info[i] = {i, m_primitives[i]->bbox() };
        }

        int total_nodes = 0;
        host_vector<ref<Emitter>, Float> ordered_prims;
        BVHNode *root;

        root = recursive_build(prim_info, 0, m_primitives.size(), &total_nodes, ordered_prims);
        m_primitives.swap(ordered_prims);

        m_nodes = new LinearBVHNode[total_nodes]; // TODO: Allocate memory differently?
        m_total_nodes = total_nodes;

        int offset = 0;
        flatten_bvh_tree(root, &offset);
    }

    ~BVH() {
        delete m_nodes;
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
        BVHPrimInfo(size_t prim_number, const ScalarBoundingBox3f &bbox):
            prim_number(prim_number), bbox(bbox), centroid(bbox.center()) { }

        size_t prim_number;
        ScalarBoundingBox3f bbox;
        Point3f centroid;
    };

    struct BVHNode {
        void init_leaf(int first, int n, const ScalarBoundingBox3f &box) {
            first_prim_offset = first;
            prim_count = n;
            bbox = box;
            children[0] = children[1] = nullptr;
        }

        void init_inner(int axis, BVHNode *c0, BVHNode *c1) {
            children[0] = c0;
            children[1] = c1;
            bbox = ScalarBoundingBox3f::merge(c0->bbox, c1->bbox);
            split_axis = axis;
            prim_count = 0;
        }

        ScalarBoundingBox3f bbox;
        BVHNode *children[2];
        int split_axis;
        int first_prim_offset;
        int prim_count;
    };

    struct LinearBVHNode {
        ScalarBoundingBox3f bbox;
        union {
            int primitives_offset; // leaf
            int second_child_offset; // inner
        };
        uint16_t prim_count; // 0 -> inner node
        uint8_t axis;
        uint8_t pad[1];
    };

protected:
    BVHNode* recursive_build(std::vector<BVHPrimInfo> &primitive_info,
                             int start,
                             int end,
                             int *total_nodes,
                             host_vector<ref<Emitter>, Float> &ordered_prims) {
        BVHNode* node;
        (*total_nodes)++;

        ScalarBoundingBox3f node_bbox;
        ScalarBoundingBox3f centroid_bbox;
        for (int i = start; i < end; i++) {
            node_bbox.expand(primitive_info[i].bbox);
            centroid_bbox.expand(primitive_info[i].centroid); // TODO: Verify this works as box isn't initalized at the beginning
        }

        int nb_prim = end - start;
        if (nb_prim == 1) {
            node = create_leaf(primitive_info, start, end, ordered_prims, node_bbox);
        } else {
            int dim = centroid_bbox.major_axis(); // TODO: Wrong type

            int mid = (start + end) / 2;
            if (centroid_bbox.max[dim] == centroid_bbox.min[dim]) { // TODO: Check if this equality is problematic
                node = create_leaf(primitive_info, start, end, ordered_prims, node_bbox);
            } else {
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
                default: {
                    if (nb_prim <= 4) { //TODO: Define this threshold
                        mid = (start + end) / 2;
                        std::nth_element(&primitive_info[start], &primitive_info[mid], &primitive_info[end-1] + 1,
                                [dim](const BVHPrimInfo &a, const BVHPrimInfo &b) {
                            return a.centroid[dim] < b.centroid[dim];
                        });
                    } else {
                        constexpr int nb_buckets = 12; // TODO: define this value

                        struct BucketInfo {
                            int count = 0;
                            ScalarBoundingBox3f bbox;
                        };

                        BucketInfo buckets[nb_buckets];

                        for (int i = start; i < end; i++) {
                            int b = nb_buckets * centroid_bbox.offset(primitive_info[i].centroid)[dim];
                            if (b == nb_buckets) {
                                b = nb_buckets - 1;
                            }

                            buckets[b].count++;
                            buckets[b].bbox.expand(primitive_info[i].bbox);
                        }

                        Float cost[nb_buckets - 1];
                        for (int i = 0; i < nb_buckets - 1; i++) {
                            ScalarBoundingBox3f b0, b1;
                            int count0 = 0, count1 = 0;

                            for (int j = 0; j <= i; j++) {
                                b0.expand(buckets[j].bbox);
                                count0 += buckets[j].count;
                            }

                            for (int j = i+1; j < nb_buckets; j++) {
                                b1.expand(buckets[j].bbox);
                                count1 += buckets[j].count;
                            }

                            cost[i] = 0.125f + (count0 * b0.surface_area() + // TODO: Define this cost
                                                count1 * b1.surface_area()) / node_bbox.surface_area();
                        }

                        Float min_cost = cost[0];
                        int min_cost_split_bucket = 0;
                        for (int i = 1; i < nb_buckets - 1; i++) {
                            if (cost[i] < min_cost) {
                                min_cost = cost[i];
                                min_cost_split_bucket = i;
                            }
                        }

                        Float leaf_cost = nb_prim;
                        if (nb_prim > m_max_prims_in_node || min_cost < leaf_cost) {
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
                            node = create_leaf(primitive_info, start, end, ordered_prims, node_bbox);
                            return node;
                        }
                    }
                    break;
                }
                }

                node = new BVHNode();
                node->init_inner(dim,
                                 recursive_build(primitive_info, start, mid, total_nodes, ordered_prims),
                                 recursive_build(primitive_info, mid, end, total_nodes, ordered_prims));
            }
        }

        return node;
    }

    int flatten_bvh_tree(BVHNode *node, int *offset) {
        LinearBVHNode *linear_node = &m_nodes[*offset];
        linear_node->bbox = node->bbox;
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
    BVHNode* create_leaf(std::vector<BVHPrimInfo> &primitive_info,
                         int start,
                         int end,
                         host_vector<ref<Emitter>, Float> &ordered_prims,
                         ScalarBoundingBox3f &prims_bbox) {
        BVHNode *leaf = new BVHNode();

        int nb_prim = end - start;
        int first_prim_offset = ordered_prims.size();
        for (int i = start; i < end; i++) {
            int prim_num = primitive_info[i].prim_number;
            ordered_prims.push_back(m_primitives[prim_num]);
        }

        leaf->init_leaf(first_prim_offset, nb_prim, prims_bbox);

        return leaf;
    }

private:
    const int m_max_prims_in_node;
    const SplitMethod m_split_method;
    host_vector<ref<Emitter>, Float> m_primitives;
    LinearBVHNode *m_nodes = nullptr;
    int m_total_nodes;
};

NAMESPACE_END(mitsuba)
