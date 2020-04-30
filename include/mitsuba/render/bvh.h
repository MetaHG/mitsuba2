#pragma once

#include <mitsuba/core/bbox.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/vector.h>

#include <mitsuba/render/emitter.h>

#include <filesystem>
#include <fstream>
#include <unordered_map>

NAMESPACE_BEGIN(mitsuba)

enum class SplitMethod { SAH, SAOH, Middle, EqualCounts };

MTS_VARIANT class MTS_EXPORT_RENDER BVH : public Object {
public:
    MTS_IMPORT_TYPES(Emitter, Shape, Mesh)

    // Use 32 bit indices to keep track of indices to conserve memory
    using ScalarIndex = uint32_t; // TODO: See how to import this from shape
    using EmitterPtr = replace_scalar_t<Float, const Emitter *>;

    BVH(host_vector<ref<Emitter>, Float> p, int max_prims_in_node, SplitMethod split_method, bool visualize_volumes = false);

    ~BVH();

    std::pair<DirectionSample3f, Spectrum> sample_emitter(const Float &tree_sample, const SurfaceInteraction3f &ref, const Point2f &emitter_sample, const Mask active);

    Float pdf_emitter_direction(const SurfaceInteraction3f &ref,
                                const DirectionSample3f &ds,
                                Mask active);

    std::pair<DirectionSample3f, Spectrum> sample_emitter_pure(const Float &tree_sample, const SurfaceInteraction3f &ref, const Point2f &emitter_sample, const Mask active);

    Float pdf_emitter_direction_pure(const SurfaceInteraction3f &ref,
                                const DirectionSample3f &ds,
                                Mask active);

    void to_obj();

    MTS_DECLARE_CLASS()

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

    // TODO: Refactor to use union to save some space ?
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
        bool is_root() {
            return parent_offset < 0;
        }

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
        int parent_offset;
        uint8_t pad[1]; // Padding for memory/cache alignment
    };

    struct BVHPrimitive {
        BVHPrimitive(Emitter *emitter) : emitter(emitter), leaf_offset(-1), is_triangle(false), face_id(0) {
            prim_bbox = ScalarBoundingBox3f();
            prim_cone = ScalarCone3f();
        }

        BVHPrimitive(Emitter *emitter, ScalarIndex face_id, ScalarBoundingBox3f bbox, ScalarCone3f cone) : BVHPrimitive(emitter) {
            is_triangle = true;
            this->face_id = face_id;
            prim_bbox = bbox;
            prim_cone = cone;
        }

        Emitter *emitter;
        int leaf_offset;

        bool is_triangle;
        ScalarIndex face_id;
        ScalarBoundingBox3f prim_bbox;
        ScalarCone3f prim_cone;


        inline ScalarBoundingBox3f bbox() {
            if (is_triangle) {
                return prim_bbox;
            }

            return emitter->bbox();
        }

        inline ScalarCone3f cone() {
            if (is_triangle) {
                return prim_cone;
            }

            return emitter->cone();
        }

        inline Spectrum get_total_radiance() {
            if (is_triangle) {
                const Shape *shape = emitter->shape();
                const Mesh *mesh = static_cast<const Mesh*>(shape);
                Float tri_area = mesh->face_area(face_id);
                return emitter->get_radiance() * tri_area;
            }

            return emitter->get_total_radiance();
        }

        inline std::pair<DirectionSample3f, Spectrum> sample_direction(const SurfaceInteraction3f &ref, const Point2f &emitter_sample, const Mask active) {
            if (is_triangle) {
                return emitter->sample_face_direction(face_id, ref, emitter_sample, active);
            }

            return emitter->sample_direction(ref, emitter_sample, active);
        }

        inline bool operator==(const BVHPrimitive &prim){
            return emitter == prim.emitter &&
                   face_id == prim.face_id;
        }
    };

protected:
    BVHPrimitive* sample_tree(const SurfaceInteraction3f &si, float &importance_ratio, const Float &sample_);

    // face_idx could be the node index
    Float pdf_tree(const SurfaceInteraction3f &si, const Emitter *emitter, const ScalarIndex face_idx);

    std::pair<Float, Float> compute_children_weights(int offset, const SurfaceInteraction3f &ref);

    BVHNode* recursive_build(std::vector<BVHPrimInfo> &primitive_info,
                             int start,
                             int end,
                             int *total_nodes,
                             std::vector<BVHPrimitive*> &ordered_prims,
                             std::string node_name = "");

    int flatten_bvh_tree(BVHNode *node, int *offset, int parent_offset);

private:

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

    void find_split(std::vector<BVHPrimInfo> &primitive_info, int start, int end,
                    ScalarBoundingBox3f &centroid_bbox, ScalarBoundingBox3f &node_bbox, ScalarCone3f &node_cone,
                    int nb_buckets, int &split_dim, int &split_bucket, Float &min_cost);

    Float compute_cone_weight(const LinearBVHNode &node, const SurfaceInteraction3f &si);

    BVHNode* create_leaf(std::vector<BVHPrimInfo> &primitive_info,
                         int start,
                         int end,
                         std::vector<BVHPrimitive*> &ordered_prims,
                         ScalarBoundingBox3f &prims_bbox,
                         Spectrum intensity = 0.f,
                         ScalarCone3f cone = ScalarCone3f());

    static MTS_INLINE float compute_luminance(Spectrum intensity) {
        return hmean(intensity);
    }

    void save_to_obj(std::string node_name, ScalarBoundingBox3f bbox, ScalarCone3f cone);

    void cone_to_obj(std::string filename, ScalarPoint3f center, ScalarCone3f cone);

    MTS_INLINE std::string obj_vertex(ScalarPoint3f p) {
        std::ostringstream oss;
        oss << "v " << p.x() << " " << p.y() << " " << p.z() << std::endl;
        return oss.str();
    }

    struct HashPair {
        template<class T1, class T2>
        size_t operator() (const std::pair<T1, T2> &p) const {
            auto hash1 = std::hash<T1>{}(p.first);
            auto hash2 = std::hash<T2>{}(p.second);
            return hash1 ^ (hash2 << 1);
        }
    };

private:
    const int m_max_prims_in_node;
    const SplitMethod m_split_method;
    std::vector<BVHPrimitive*> m_primitives;
    std::unordered_map<std::pair<std::string, ScalarIndex>, ScalarIndex, HashPair> m_prim_index_map;
    LinearBVHNode *m_nodes = nullptr;
    int m_total_nodes;
    int m_leaf_count;
    bool m_visualize_volumes;
    std::vector<int> m_emitter_stats;
};

NAMESPACE_END(mitsuba)
