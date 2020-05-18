#pragma once

#include <mitsuba/core/bbox.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/core/properties.h>

#include <mitsuba/render/emitter.h>

#include <filesystem>
#include <fstream>
#include <unordered_map>

NAMESPACE_BEGIN(mitsuba)

enum class SplitMethod { SAH, SAOH, Middle, EqualCounts };
enum class ClusterImportanceMethod { POWER, BASE_ESTEVEZ_PAPER, ORIENTATION_ESTEVEZ_PAPER, BASE_STOCHASTIC_YUKSEL_PAPER, ORIENTATION_STOCHASTIC_YUKSEL_PAPER };

MTS_VARIANT class MTS_EXPORT_RENDER BVH : public Object {
public:
    MTS_IMPORT_TYPES(Emitter, Shape, Mesh)

    // Use 32 bit indices to keep track of indices to conserve memory
    using ScalarIndex = uint32_t; // TODO: See how to import this from shape
    using EmitterPtr = replace_scalar_t<Float, const Emitter *>;

    BVH(const Properties &props);

    BVH(host_vector<ref<Emitter>, Float> p, int max_prims_in_node, SplitMethod split_method,
        ClusterImportanceMethod cluster_importance_method, bool visualize_volumes = false);

    ~BVH();

    void build();

    void set_primitives(host_vector<ref<Emitter>, Float> emitters);

    std::pair<DirectionSample3f, Spectrum> sample_emitter(const Float &tree_sample, const SurfaceInteraction3f &ref,
                                                          const Point2f &emitter_sample, const Mask active);

    Float pdf_emitter_direction(const SurfaceInteraction3f &ref,
                                const DirectionSample3f &ds,
                                Mask active);

    std::pair<DirectionSample3f, Spectrum> sample_emitter_pure(const Float &tree_sample, const SurfaceInteraction3f &ref,
                                                               const Point2f &emitter_sample, const Mask active);

    Float pdf_emitter_direction_pure(const SurfaceInteraction3f &ref,
                                const DirectionSample3f &ds,
                                Mask active);

    void to_obj();

    MTS_DECLARE_CLASS()

protected:
    struct BVHPrimInfo {
        BVHPrimInfo() {
               prim_number = 0;
               bbox = ScalarBoundingBox3f();
               centroid = Point3f(0.0f);
               intensity = Spectrum(0.0f);
               cone = ScalarCone3f();
        }

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

    struct IBVHEmitter {
        virtual ~IBVHEmitter() { }
        virtual Spectrum intensity() const = 0;
        virtual ScalarBoundingBox3f bbox() const = 0;
        virtual ScalarCone3f cone() const = 0;
    };

    struct LinearBVHNode : IBVHEmitter {
        // Destructor
        virtual ~LinearBVHNode() { }

        // Methods
        bool is_root() {
            return parent_offset < 0;
        }

        bool is_leaf() {
            return prim_count > 0;
        }

        inline Spectrum intensity() const override {
            return node_intensity;
        }

        inline ScalarBoundingBox3f bbox() const override {
            return node_bbox;
        }

        inline ScalarCone3f cone() const override {
            return node_cone;
        }
        // --------------------------------------------------------

        // Fields
        ScalarBoundingBox3f node_bbox;
        union {
            int primitives_offset; // leaf
            int second_child_offset; // inner
        };
        uint16_t prim_count; // 0 -> inner node
        uint8_t axis;
        Spectrum node_intensity;
        ScalarCone3f node_cone;
        int parent_offset;
        uint8_t pad[1]; // Padding for memory/cache alignment
    };

    struct BVHPrimitive : IBVHEmitter {

        // Constructors

        BVHPrimitive() {
            emitter = nullptr;
            leaf_offset = 0;
            is_triangle = false;
            face_id = 0;
            prim_bbox = ScalarBoundingBox3f();
            prim_cone = ScalarCone3f();
        }

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

        // Destructor
        virtual ~BVHPrimitive() { }

        //-----------------------------------------------

        // Methods
        inline ScalarBoundingBox3f bbox() const override {
            if (is_triangle) {
                return prim_bbox;
            }

            return emitter->bbox();
        }

        inline ScalarCone3f cone() const override {
            if (is_triangle) {
                return prim_cone;
            }

            return emitter->cone();
        }

        inline Spectrum intensity() const override {
            if (is_triangle) {
                const Shape *shape = emitter->shape();
                const Mesh *mesh = static_cast<const Mesh*>(shape);
                Float tri_area = mesh->face_area(face_id);
                return emitter->get_radiance() * tri_area;
            }

            return emitter->get_total_radiance();
        }

        inline std::pair<DirectionSample3f, Spectrum> sample_direction(const SurfaceInteraction3f &ref,
                                                                       const Point2f &emitter_sample, const Mask active) {
            if (is_triangle) {
                return emitter->sample_face_direction(face_id, ref, emitter_sample, active);
            }

            return emitter->sample_direction(ref, emitter_sample, active);
        }

        inline bool operator==(const BVHPrimitive &prim) {
            return emitter == prim.emitter &&
                   face_id == prim.face_id;
        }

        inline void operator=(const BVHPrimitive &prim) {
            emitter = prim.emitter;
            leaf_offset = prim.leaf_offset;
            is_triangle = prim.is_triangle;
            face_id = prim.face_id;
            prim_bbox = ScalarBoundingBox3f(prim.prim_bbox.min, prim.prim_bbox.max);
            prim_cone = ScalarCone3f(prim.cone());
        }

        // --------------------------------------------------------------------------------
        // Fields
        Emitter *emitter;
        int leaf_offset;

        bool is_triangle;
        ScalarIndex face_id;
        ScalarBoundingBox3f prim_bbox;
        ScalarCone3f prim_cone;
    };

protected:
    BVHPrimitive* sample_tree(const SurfaceInteraction3f &si, float &importance_ratio, const Float &sample_);

    BVHPrimitive* sample_leaf(const SurfaceInteraction3f &si, float &importance_ratio, const Float &sample_, const LinearBVHNode &leaf) const;

    Float pdf_tree(const SurfaceInteraction3f &si, const Emitter *emitter, const ScalarIndex face_idx);

    std::pair<Float, Float> compute_children_weights(int offset, const SurfaceInteraction3f &ref);

    ScalarFloat* compute_bvh_emitters_weights(const std::vector<IBVHEmitter*> &emitters, size_t size, const SurfaceInteraction3f &ref) const;

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

    Float compute_cone_weight(const ScalarBoundingBox3f &bbox, const ScalarCone3f &cone, const SurfaceInteraction3f &si) const;

    Float compute_cone_weight_test(const ScalarBoundingBox3f &bbox, const ScalarCone3f &cone, const SurfaceInteraction3f &si) const;

    Float compute_cone_weight_custom(const ScalarBoundingBox3f &bbox, const ScalarCone3f &cone, const SurfaceInteraction3f &si) const;

    MTS_INLINE Float sine(Float x) const {
        const float B = 4/math::Pi<Float>;
        const float C = -4/(math::Pi<Float>*math::Pi<Float>);

        float y = B * x + C * x * abs(x);

        //  const float Q = 0.775;
        const float P = 0.225;

        y = P * (y * abs(y) - y) + y;   // Q * y + P * y * abs(y)


        return y;
    }

    MTS_INLINE Float cosine(Float x) const {
        return sine(x + (math::Pi<Float>/ 2));
    }

    MTS_INLINE Float arccosine(Float x) const {
       return (-0.69813170079773212 * x * x - 0.87266462599716477) * x + 1.5707963267948966;
    }



    BVHNode* create_leaf(std::vector<BVHPrimInfo> &primitive_info,
                         int start,
                         int end,
                         std::vector<BVHPrimitive*> &ordered_prims,
                         ScalarBoundingBox3f &prims_bbox,
                         Spectrum intensity = 0.f,
                         ScalarCone3f cone = ScalarCone3f());

    static MTS_INLINE Float compute_luminance(Spectrum intensity) {
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
    int m_max_prims_in_node;
    SplitMethod m_split_method;
    ClusterImportanceMethod m_cluster_importance_method;
    bool m_split_mesh;
    std::vector<BVHPrimitive*> m_primitives;
    std::unordered_map<std::pair<std::string, ScalarIndex>, ScalarIndex, HashPair> m_prim_index_map;
    LinearBVHNode *m_nodes = nullptr;
    int m_total_nodes;
    int m_leaf_count;
    bool m_visualize_volumes;
    std::vector<int> m_emitter_stats;
};

MTS_EXTERN_CLASS_RENDER(BVH)
NAMESPACE_END(mitsuba)
