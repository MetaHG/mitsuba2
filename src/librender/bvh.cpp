#include <mitsuba/render/bvh.h>
#include <mitsuba/render/mesh.h>
#include <mitsuba/core/distr_1d.h>

NAMESPACE_BEGIN(mitsuba)

#define CONE_SCALE_FACTOR 0.5
#define YUKSEL_DISTANCE_RATIO 1.0f
#define NB_CHILDREN_PER_NODE 2
#define MAX_PRIMS_IN_NODE 255

MTS_VARIANT BVH<Float,Spectrum>::BVH(const Properties &props) {
    m_split_mesh = props.bool_("split_mesh", true);
    m_uniform_leaf_sampling = props.bool_("uniform_leaf_sampling", false);

    const std::string split_metric = props.string("split_metric", "saoh");
    if (split_metric == "equal_counts") {
        m_split_method = SplitMethod::EqualCounts;
    } else if (split_metric == "middle") {
        m_split_method = SplitMethod::Middle;
    } else if (split_metric == "sah") {
        m_split_method = SplitMethod::SAH;
    } else {
        m_split_method = SplitMethod::SAOH;
    }

    const std::string cluster_importance = props.string("cluster_importance", "orientation_estevez");
    if (cluster_importance == "power") {
        m_cluster_importance_method = ClusterImportanceMethod::POWER;
    } else if (cluster_importance == "base_yuksel") {
        m_cluster_importance_method = ClusterImportanceMethod::BASE_STOCHASTIC_YUKSEL_PAPER;
    } else if (cluster_importance == "orientation_yuksel") {
        m_cluster_importance_method = ClusterImportanceMethod::ORIENTATION_STOCHASTIC_YUKSEL_PAPER;
    } else if (cluster_importance == "base_estevez") {
        m_cluster_importance_method = ClusterImportanceMethod::BASE_ESTEVEZ_PAPER;
    } else {
        m_cluster_importance_method = ClusterImportanceMethod::ORIENTATION_ESTEVEZ_PAPER;
    }

    m_visualize_volumes = props.bool_("visualization", false);
    m_max_prims_in_node = min(MAX_PRIMS_IN_NODE, props.int_("max_prims_in_node", 1));
}

MTS_VARIANT BVH<Float, Spectrum>::BVH(host_vector<ref<Emitter>, Float> p, int max_prims_in_node,
                                      SplitMethod split_method, ClusterImportanceMethod cluster_importance_method,
                                      bool split_mesh, bool uniform_leaf_sampling, bool visualize_volumes):
    m_max_prims_in_node(std::min(255, max_prims_in_node)), m_split_method(split_method),
    m_cluster_importance_method(cluster_importance_method), m_split_mesh(split_mesh),
    m_uniform_leaf_sampling(uniform_leaf_sampling), m_visualize_volumes(visualize_volumes) {

    Log(Info, "Building a SAOH BVH Light Hierarchy");

    set_primitives(p);
    build();
}

MTS_VARIANT BVH<Float, Spectrum>::~BVH() {
    // TODO: Clean this
    auto b = m_emitter_stats.begin();
    auto e = m_emitter_stats.end();
    auto q = m_emitter_stats.begin();
    std::advance(q, (int) (0.5 * m_emitter_stats.size()));
    std::nth_element(b, q, e);
    std::cout << "QUANTILE: " << *q << std::endl;

    std::cout << "TREE NODE: " << m_nodes->cone() << std::endl;

    for (BVHPrimitive* prim: m_primitives) {
        delete prim;
    }

    delete[] m_nodes;
}

MTS_VARIANT void BVH<Float,Spectrum>::build() {
    Log(Info, "Building a SAOH BVH Light Hierarchy");

    if (m_primitives.size() == 0) {
        return;
    }

    std::vector<BVHPrimInfo> prim_info(m_primitives.size());
    for (size_t i = 0; i < m_primitives.size(); i++) {
        switch (m_split_method) {
            case SplitMethod::SAOH: {
                prim_info[i] = { i, m_primitives[i]->bbox(), m_primitives[i]->intensity(), m_primitives[i]->cone() };
                break;
            }
            default: {
                prim_info[i] = { i, m_primitives[i]->bbox() };
                break;
            }
        }
    }

    m_leaf_count = 0;
    int total_nodes = 0;
    std::vector<BVHPrimitive*> ordered_prims;
    BVHNode *root;

    root = recursive_build(prim_info, 0, m_primitives.size(), &total_nodes, ordered_prims, "_");
    m_primitives.swap(ordered_prims);

    for (ScalarIndex i = 0; i < m_primitives.size(); i++) {
        m_prim_index_map[m_primitives[i]->emitter][m_primitives[i]->face_id] = i;
    }

    m_nodes = new LinearBVHNode[total_nodes];
    m_total_nodes = total_nodes;

    int offset = 0;
    flatten_bvh_tree(root, &offset, -1);

    Log(Info, "Finished.");
    Log(Info, "BVH Light Hierarchy statistics:\n"
              "  Primitive count: %s,\n"
              "  Leaf count: %s,\n"
              "  Total nodes: %s", m_primitives.size(), m_leaf_count, m_total_nodes);
}

MTS_VARIANT void BVH<Float, Spectrum>::set_primitives(host_vector<ref<Emitter>, Float> emitters) {
    m_emitter_stats = std::vector<int>();
    m_primitives = std::vector<BVHPrimitive*>();
    m_prim_index_map = std::unordered_map<const Emitter*, std::vector<ScalarIndex>>();
    for (size_t i = 0; i < emitters.size(); i++) {
        Shape *shape = emitters[i].get()->shape();
        if (shape && shape->is_mesh() && m_split_mesh) {
            Mesh *mesh = static_cast<Mesh*>(shape);

            ScalarIndex skipped_face_count = 0;
            for (ScalarIndex j = 0; j < mesh->face_count(); j++) {
                if (mesh->face_area(j) > 0) { // Don't add degenerate triangle with surface area of 0
                    m_primitives.push_back(new BVHPrimitive(emitters[i], j, mesh->face_bbox(j), mesh->face_cone(j)));
                    m_emitter_stats.push_back(0);
                } else {
                    skipped_face_count += 1;
                }
            }

            m_prim_index_map[emitters[i]] = std::vector<ScalarIndex>(mesh->face_count(), -1);

            if (skipped_face_count > 0) {
                Log(Warn, "BVH Light Hierarchy: Skipped %s faces (area is zero) of mesh with id %s", skipped_face_count, mesh->id());
            }
        } else {
            m_emitter_stats.push_back(0);
            m_primitives.push_back(new BVHPrimitive(emitters[i]));
            m_prim_index_map[emitters[i]] = std::vector<ScalarIndex>(1, -1);
        }
    }
}

MTS_VARIANT std::pair<typename BVH<Float, Spectrum>::DirectionSample3f, Spectrum>
BVH<Float,Spectrum>::sample_emitter(const Float &tree_sample, const SurfaceInteraction3f &ref, const Point2f &emitter_sample, const Mask active) {
    float pdf = 1.0f;

    BVHPrimitive *prim = sample_tree(ref, pdf, tree_sample);

    DirectionSample3f ds;
    Spectrum spec;
    std::tie(ds, spec) = prim->sample_direction(ref, emitter_sample, active);

    ds.pdf *= pdf;
    return std::pair(ds, spec / pdf);
}

MTS_VARIANT Float BVH<Float, Spectrum>::pdf_emitter_direction(const SurfaceInteraction3f &ref,
                            const DirectionSample3f &ds,
                            Mask active) {
    const Emitter *emitter = reinterpret_array<EmitterPtr>(ds.object);
    const Shape *shape = emitter->shape();

    ScalarIndex face_idx = m_split_mesh ? ds.prim_index : 0;

    Float emitter_pdf = 1.0f;
    if (shape->is_mesh() && m_split_mesh) {
        const Mesh *mesh = static_cast<const Mesh*>(shape);
        if (mesh->face_area(face_idx) == 0) { // Handle degenerate triangles
            return 0;
        }
        emitter_pdf = emitter->pdf_face_direction(face_idx, ref, ds, active);
    } else {
        emitter_pdf = emitter->pdf_direction(ref, ds, active);
    }

    return emitter_pdf * pdf_tree(ref, emitter, face_idx);
}

// TODO: REFACTOR THIS
MTS_VARIANT std::pair<typename BVH<Float, Spectrum>::DirectionSample3f, Spectrum>
BVH<Float,Spectrum>::sample_emitter_pure(const Float &tree_sample, const SurfaceInteraction3f &ref, const Point2f &emitter_sample, const Mask active) {
    float pdf = 1.0f;

    BVHPrimitive *prim = sample_tree(ref, pdf, tree_sample);

    DirectionSample3f ds;
    Spectrum spec;
    std::tie(ds, spec) = prim->sample_direction(ref, emitter_sample, active);

    ds.pdf *= pdf;
    ds.pdf = 1.0f;

    return std::pair(ds, prim->emitter->get_radiance());
}

// TODO: REFACTOR THIS
MTS_VARIANT Float BVH<Float, Spectrum>::pdf_emitter_direction_pure(const SurfaceInteraction3f &ref,
                            const DirectionSample3f &ds,
                            Mask active) {
    const Emitter *emitter = reinterpret_array<EmitterPtr>(ds.object);
    const Shape *shape = emitter->shape();

    ScalarIndex face_idx = ds.prim_index;

    Float emitter_pdf = 1.0f;
    if (shape->is_mesh()) {
        const Mesh *mesh = static_cast<const Mesh*>(shape);
        if (mesh->face_area(face_idx) == 0) { // Handle degenerate triangles
            return 0;
        }
        emitter_pdf = emitter->pdf_face_direction(face_idx, ref, ds, active);
    } else {
        emitter_pdf = emitter->pdf_direction(ref, ds, active);
    }

    return 1.0f; //emitter_pdf * pdf_tree(ref, emitter, face_idx);
}

// TODO: Refactor this
MTS_VARIANT void BVH<Float, Spectrum>::to_obj() {
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
            Point p = m_nodes[i].node_bbox.corner(j);
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

MTS_VARIANT typename BVH<Float, Spectrum>::BVHPrimitive*
BVH<Float, Spectrum>::sample_tree(const SurfaceInteraction3f &si, float &importance_ratio, const Float &sample_) {
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

    const LinearBVHNode leaf = m_nodes[offset];
    BVHPrimitive* prim;

    if (leaf.prim_count == 1) {
        // Fast path is there is only one primitive in the leaf.
        prim = m_primitives[leaf.primitives_offset];
    } else {
        prim = sample_leaf(si, importance_ratio, sample, m_nodes[offset]);;
    }

    return prim;
}

MTS_VARIANT Float BVH<Float, Spectrum>::pdf_tree(const SurfaceInteraction3f &si, const Emitter *emitter, const ScalarIndex face_idx) {
    Float pdf = 1.0;

    ScalarIndex prim_idx = m_prim_index_map[emitter][face_idx];
    BVHPrimitive* prim = m_primitives[prim_idx];

    int prev_offset;
    int current_offset = prim->leaf_offset;
    LinearBVHNode *current_node;

    // While root is not reached
    while(current_offset != -1) {
        current_node = &m_nodes[current_offset];

        if (current_node->is_leaf()) {
            // Skip leaves that contain only one primitive (as pdf = 1)
            if (current_node->prim_count != 1) {
                pdf *= pdf_leaf(si, current_node, prim_idx - current_node->primitives_offset);
            }
        } else {
            Float w_left, w_right;
            std::tie(w_left, w_right) = compute_children_weights(current_offset, si);

            Float p_left = 0.5f;
            if (w_left + w_right >= std::numeric_limits<Float>::epsilon()) {
                p_left = w_left / (w_left + w_right);
            }

            int first_child_offest = current_offset + 1;
            if (prev_offset == first_child_offest) {
                pdf *= p_left;
            } else {
                pdf *= (1.f - p_left);
            }
        }

        prev_offset = current_offset;
        current_offset = current_node->parent_offset;
    }

    return pdf;
}

MTS_VARIANT typename BVH<Float, Spectrum>::BVHPrimitive*
BVH<Float, Spectrum>::sample_leaf(const SurfaceInteraction3f &si, float &importance_ratio, const Float &sample_, const LinearBVHNode &leaf) {
    int prim_offset = 0;
    if (m_uniform_leaf_sampling) {
        int leaf_prim_count = leaf.prim_count;

        prim_offset = leaf_prim_count * sample_;
        if (prim_offset == leaf_prim_count) {
            prim_offset -= 1;
        }

        prim_offset += leaf.primitives_offset;;
        importance_ratio /= leaf_prim_count;

    } else {
        ScalarFloat* prim_weights = compute_bvh_emitters_weights(m_primitives, leaf.primitives_offset, leaf.prim_count, si);
        DiscreteDistribution<Float> leaf_distrib(prim_weights, leaf.prim_count);

        ScalarIndex offset;
        Float leaf_prim_pdf;
        std::tie(offset, leaf_prim_pdf) = leaf_distrib.sample_pmf(sample_);
        importance_ratio *= leaf_prim_pdf;

        prim_offset = leaf.primitives_offset + offset;
    }

    m_emitter_stats[prim_offset] += 1;
    return m_primitives[prim_offset];
}

MTS_VARIANT Float BVH<Float, Spectrum>::pdf_leaf(const SurfaceInteraction3f &si, const LinearBVHNode *leaf, ScalarIndex prim_idx) const {
    Float pdf;
    if (m_uniform_leaf_sampling) {
        pdf = 1.0f / leaf->prim_count;
    } else {
        ScalarFloat* prim_weights = compute_bvh_emitters_weights(m_primitives, leaf->primitives_offset, leaf->prim_count, si);
        DiscreteDistribution<Float> leaf_distrib(prim_weights, leaf->prim_count);
        pdf = leaf_distrib.eval_pmf_normalized(prim_idx);
    }

    return pdf;
}

MTS_VARIANT typename BVH<Float, Spectrum>::ScalarFloat*
BVH<Float, Spectrum>::compute_bvh_emitters_weights(const std::vector<BVHPrimitive*> &emitters, size_t offset, size_t size, const SurfaceInteraction3f &ref) const {
    ScalarFloat* weights = new ScalarFloat[size];
    ScalarFloat* distances = new ScalarFloat[size];

    for (size_t i = 0; i < size; i++) {
        weights[i] = compute_luminance(emitters[offset + i]->intensity());
        distances[i] = ScalarFloat(1.0f);
    }

    switch (m_cluster_importance_method) {
        case ClusterImportanceMethod::POWER: {
            return weights;
        }
        break;

        case ClusterImportanceMethod::ORIENTATION_STOCHASTIC_YUKSEL_PAPER: {
            for (size_t i = 0; i < size; i++) {
                BVHPrimitive* emitter = emitters[offset + i];
                weights[i] *= compute_cone_weight(emitter->prim_bbox, emitter->prim_cone_cosine, ref);
//                weights[i] *= compute_cone_weight_old(emitter->bbox(), emitter->cone(), ref);
            }
        }
        case ClusterImportanceMethod::BASE_STOCHASTIC_YUKSEL_PAPER: {
            for (size_t i = 0; i < size; i++) {
                distances[i] = emitters[offset + i]->bbox().squared_distance(ref.p);

                if (distances[i] <= YUKSEL_DISTANCE_RATIO * squared_norm(emitters[offset + i]->bbox().extents())) {
                    return weights;
                }
            }
        }
        break;

        case ClusterImportanceMethod::ORIENTATION_ESTEVEZ_PAPER: {
            for (size_t i = 0; i < size; i++) {
                BVHPrimitive* emitter = emitters[offset + i];
                weights[i] *= compute_cone_weight(emitter->prim_bbox, emitter->prim_cone_cosine, ref);
//                weights[i] *= compute_cone_weight_old(emitter->bbox(), emitter->cone(), ref);
            }
        }
        case ClusterImportanceMethod::BASE_ESTEVEZ_PAPER: {
            for (size_t i = 0; i < size; i++) {
                distances[i] = max(max(squared_norm(emitters[offset + i]->bbox().extents()) / 4.0f, std::numeric_limits<Float>::epsilon()),
                                   squared_norm(emitters[offset + i]->bbox().center() - ref.p));
            }
        }
        break;
    }

    for (size_t i = 0; i < size; i++) {
//        weights[i] /= distances[i];

        // Epsilon is added as area_distribution_1d does not handle full zero weights.
        weights[i] = (weights[i] + std::numeric_limits<Float>::epsilon()) / distances[i];
    }

    return weights;
}

MTS_VARIANT std::pair<Float, Float> BVH<Float, Spectrum>::compute_children_weights(int offset, const SurfaceInteraction3f &ref) {
    const LinearBVHNode &ln = m_nodes[offset + 1];
    const LinearBVHNode &rn = m_nodes[m_nodes[offset].second_child_offset];

    Float l_weight = compute_luminance(ln.node_intensity);
    Float r_weight = compute_luminance(rn.node_intensity);

    Float left_d = 1.0f;
    Float right_d = 1.0f;

    switch (m_cluster_importance_method) {
        case ClusterImportanceMethod::POWER: {
            return std::pair(l_weight, r_weight);
        }
        break;

        case ClusterImportanceMethod::ORIENTATION_STOCHASTIC_YUKSEL_PAPER: {
            l_weight *= compute_cone_weight_old(ln.node_bbox, ln.node_cone_cosine, ref);
            r_weight *= compute_cone_weight_old(rn.node_bbox, rn.node_cone_cosine, ref);
        }

        case ClusterImportanceMethod::BASE_STOCHASTIC_YUKSEL_PAPER: {
            left_d *= ln.node_bbox.squared_distance(ref.p);
            right_d *= rn.node_bbox.squared_distance(ref.p);

            if (left_d <= YUKSEL_DISTANCE_RATIO * squared_norm(ln.node_bbox.extents())
                || right_d <= YUKSEL_DISTANCE_RATIO * squared_norm(rn.node_bbox.extents())) {
                return std::pair(l_weight, r_weight);
            }
        }
        break;

        case ClusterImportanceMethod::ORIENTATION_ESTEVEZ_PAPER:
        default: {
            l_weight *= compute_cone_weight(ln.node_bbox, ln.node_cone_cosine, ref);
            r_weight *= compute_cone_weight(rn.node_bbox, rn.node_cone_cosine, ref);

//            Float l_cone_weight = compute_cone_weight_old(ln.node_bbox, ScalarCone3f(ln.node_cone_cosine.axis, acos(ln.node_cone_cosine.normal_angle), acos(ln.node_cone_cosine.emission_angle)), ref);
//            Float l_cone_weight_test = compute_cone_weight(ln.node_bbox, ln.node_cone_cosine, ref);
//            Float left_abs_error = abs(l_cone_weight_test - l_cone_weight);
//            Float left_error = left_abs_error / l_cone_weight;
//            if (left_error >= 0.01f && left_abs_error > 0.001f) {
//                Log(Warn, "Left error: %s, correct: %s, test: %s", left_error, l_cone_weight, l_cone_weight_test);
//            }

//            Float r_cone_weight = compute_cone_weight_old(rn.node_bbox, ScalarCone3f(rn.node_cone_cosine.axis, acos(rn.node_cone_cosine.normal_angle), acos(rn.node_cone_cosine.emission_angle)), ref);
//            Float r_cone_weight_test = compute_cone_weight(rn.node_bbox, rn.node_cone_cosine, ref);
//            Float right_abs_error = abs(r_cone_weight_test - r_cone_weight);
//            Float right_error = right_abs_error / r_cone_weight;
//            if (right_error >= 0.01f && right_abs_error > 0.001f) {
//                Log(Warn, "Right error: %s, correct: %s, test: %s", right_error, r_cone_weight, r_cone_weight_test);
//            }



//            Log(Info, "Left error: %s", abs(l_cone_weight_test - l_cone_weight) / l_cone_weight);
//            Log(Info, "Right error: %s", abs(r_cone_weight_test - r_cone_weight) / r_cone_weight);
//            Log(Warn, "Left: correct: %s, test: %s", compute_cone_weight(ln.node_bbox, ln.node_cone, ref), compute_cone_weight_test(ln.node_bbox, ln.node_cone, ref));
//            Log(Warn, "Right: correct: %s, test: %s", compute_cone_weight(rn.node_bbox, rn.node_cone, ref), compute_cone_weight_test(rn.node_bbox, rn.node_cone, ref));
        }

        case ClusterImportanceMethod::BASE_ESTEVEZ_PAPER: {
            left_d = max(max(squared_norm(ln.node_bbox.extents()) / 4.0f, std::numeric_limits<Float>::epsilon()), squared_norm(ln.node_bbox.center() - ref.p));
            right_d = max(max(squared_norm(rn.node_bbox.extents()) / 4.0f, std::numeric_limits<Float>::epsilon()), squared_norm(rn.node_bbox.center() - ref.p));
        }
        break;

    }

    return std::pair(l_weight / left_d, r_weight / right_d);
}

MTS_VARIANT MTS_INLINE Float BVH<Float,Spectrum>::compute_cone_weight(const ScalarBoundingBox3f &bbox, const ScalarCone3f &cone_cosine, const SurfaceInteraction3f &si) const {
    ScalarVector3f p_to_box_center = normalize(bbox.center() - si.p);

    Float cos_incident_angle = dot(p_to_box_center, si.n);
    Float cos_bounding_angle = 1.0f;

    if (bbox.contains(si.p)) {
        return 1.0f;
    }

    for (size_t i = 0; i < 8; i++) {
        ScalarPoint3f bbox_corner;

        switch (i) {
            case 0: bbox_corner = bbox.min; break;
            case 1: bbox_corner = ScalarPoint3f(bbox.max[0], bbox.min[1], bbox.min[2]); break;
            case 2: bbox_corner = ScalarPoint3f(bbox.min[0], bbox.max[1], bbox.min[2]); break;
            case 3: bbox_corner = ScalarPoint3f(bbox.max[0], bbox.max[1], bbox.min[2]); break;
            case 4: bbox_corner = ScalarPoint3f(bbox.min[0], bbox.min[1], bbox.max[2]); break;
            case 5: bbox_corner = ScalarPoint3f(bbox.max[0], bbox.min[1], bbox.max[2]); break;
            case 6: bbox_corner = ScalarPoint3f(bbox.min[0], bbox.max[1], bbox.max[2]); break;
            case 7: bbox_corner = bbox.max; break;
        }

        ScalarVector3f p_corner = normalize(bbox_corner - si.p);
        cos_bounding_angle = enoki::min(cos_bounding_angle, dot(p_to_box_center, p_corner));
    }

    Float sin_bounding_angle = safe_sqrt(1.0f - cos_bounding_angle * cos_bounding_angle);

    Float cos_cone_axis_and_box_to_p = dot(cone_cosine.axis, -p_to_box_center);
    Float sin_cone_axis_and_box_to_p = safe_sqrt(1.0f - cos_cone_axis_and_box_to_p * cos_cone_axis_and_box_to_p);

    Float cos_cone_normal_angle = cone_cosine.normal_angle;
    Float sin_cone_normal_angle = safe_sqrt(1.0f - cos_cone_normal_angle * cos_cone_normal_angle);

    Float cos_min_incident_angle = 1.0f;
    Float cos_min_emission_angle = 1.0f;

//    Log(Info, "Test: in_angle %s, cos_in_angle %s, bangle %s, cos_bangle %s", acos(cos_incident_angle), cos_incident_angle, acos(cos_bounding_angle), cos_bounding_angle);
    bool box_visible = cos_incident_angle < cos_bounding_angle;

    bool potential_illumination =
            (cos_cone_axis_and_box_to_p * cos_bounding_angle + sin_cone_axis_and_box_to_p * sin_bounding_angle < cos_cone_normal_angle) &&
            (cos_cone_axis_and_box_to_p * cos_cone_normal_angle - sin_cone_axis_and_box_to_p * sin_cone_normal_angle < cos_bounding_angle);

    if (box_visible) {
        cos_min_incident_angle = cos_incident_angle * cos_bounding_angle +
                safe_sqrt(1.0f - cos_incident_angle * cos_incident_angle) * sin_bounding_angle;
    }

    if (potential_illumination) {
//        Log(Info, "Test: Has potential illumination, caxis_p_angle: %s, cos_caxis_p_angle: %s, cone_normal_angle: %s, cos_cone_normal_angle %s", acos(cos_cone_axis_and_box_to_p), cos_cone_axis_and_box_to_p, cone.normal_angle, cos_cone_normal_angle);

        cos_min_emission_angle = cos_cone_axis_and_box_to_p * cos_cone_normal_angle * cos_bounding_angle
                + sin_cone_axis_and_box_to_p * sin_cone_normal_angle * cos_bounding_angle
                + sin_cone_axis_and_box_to_p * cos_cone_normal_angle * sin_bounding_angle
                - cos_cone_axis_and_box_to_p * sin_cone_normal_angle * sin_bounding_angle;
    }

    Float cos_cone_emission_angle = cone_cosine.emission_angle;
    Float cone_weight = 0;
    if (cos_min_emission_angle > cos_cone_emission_angle) {
//        Log(Info, "Test: cos_min_emission_angle: %s", cos_min_emission_angle);
        cone_weight = max(cos_min_incident_angle, 0) * cos_min_emission_angle; // cos_min_incident_angle is not 1 when it should (when cone_weight should be 1)
    }

    return cone_weight;
}

MTS_VARIANT MTS_INLINE Float BVH<Float,Spectrum>::compute_cone_weight_old(const ScalarBoundingBox3f &bbox, const ScalarCone3f &cone, const SurfaceInteraction3f &si) const {
    ScalarVector3f p_to_box_center = normalize(bbox.center() - si.p);

//  MATH_PI
//    if (!node->bbox.contains(si.p) && node->bcone.normal_angle + node->bcone.emission_angle <= M_PI_2f32 && dot(node->bcone.axis, -p_to_box_center) < 0) {
//        return 0;
//    }

    Float in_angle = acos(dot(p_to_box_center, si.n));
//    Log(Info, "p_to_box_center: %s, si.n: %s", p_to_box_center, si.n);

    Float bangle = bbox.solid_angle(si.p);

//    Log(Info, "Normal: in_angle %s, cos_in_angle %s, bangle %s, cos_bangle %s", in_angle, cos(in_angle), bangle, cos(bangle));
    Float min_in_angle = max(in_angle - bangle, 0);
//    Log(Info, "In_angle: %s, bangle: %s, min_in_angle: %s", rad_to_deg(in_angle), rad_to_deg(bangle), rad_to_deg(min_in_angle));

    Float caxis_p_angle = acos(dot(cone.axis, -p_to_box_center));

//    if (caxis_p_angle - cone.normal_angle - bangle > 0) {
//        Log(Info, "Normal: Has potential illumination, caxis_p_angle: %s, cos_caxis_p_angle: %s, cone_normal_angle: %s, cos_cone_normal_angle %s", caxis_p_angle, cos(caxis_p_angle), cone.normal_angle, cos(cone.normal_angle));
//    }

    Float min_e_angle = max(caxis_p_angle - cone.normal_angle - bangle, 0);
//    Log(Info, "caxis_p_angle: %s, min_e_angle: %s", rad_to_deg(caxis_p_angle), rad_to_deg(min_e_angle));

//    Log(Info, "Cluster cone: %s", node->cone());

    Float cone_weight = 0;
    if (min_e_angle < cone.emission_angle) {
//        Log(Info, "Normal: cos_min_emission_angle: %s", cos(min_e_angle));
        cone_weight = max(cos(min_in_angle), 0) * cos(min_e_angle);
    }

    return cone_weight;
}


// MAYBE REFACTOR THIS WITH SMALLER METHODS
MTS_VARIANT typename BVH<Float,Spectrum>::BVHNode*
BVH<Float,Spectrum>::recursive_build(std::vector<BVHPrimInfo> &primitive_info,
                                     int start,
                                     int end,
                                     int *total_nodes,
                                     std::vector<BVHPrimitive*> &ordered_prims,
                                     std::string node_name) {
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

                if (min_cost + std::numeric_limits<Float>::epsilon() < leaf_cost || nb_prim > m_max_prims_in_node) {
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

// Tree post-construction processing
MTS_VARIANT int BVH<Float, Spectrum>::flatten_bvh_tree(BVHNode *node, int *offset, int parent_offset) {
    LinearBVHNode *linear_node = &m_nodes[*offset];
    linear_node->node_bbox = node->bbox;
    // Store cosinus of angles for future computation optimizations (see "compute_cone_weight" function)
    linear_node->node_cone_cosine = ScalarCone3f(node->bcone.axis, cos(node->bcone.normal_angle), cos(node->bcone.emission_angle));
    linear_node->node_intensity = node->intensity;
    linear_node->parent_offset = parent_offset;
    int my_offset = (*offset)++;

    if (node->prim_count > 0) {
        linear_node->primitives_offset = node->first_prim_offset;
        linear_node->prim_count = node->prim_count;

        for (int i = 0; i < linear_node->prim_count; i++) {
            BVHPrimitive* prim_ptr = m_primitives[linear_node->primitives_offset + i];
            prim_ptr->leaf_offset = my_offset;
            // Convert to cone with cosinus angles for future computation optimizations (see "compute_cone_weight" function)
            prim_ptr->prim_cone_cosine = ScalarCone3f(prim_ptr->prim_cone.axis, cos(prim_ptr->prim_cone.normal_angle), cos(prim_ptr->prim_cone.emission_angle));
        }
    } else {
        linear_node->axis = node->split_axis;
        linear_node->prim_count = 0;
        flatten_bvh_tree(node->children[0], offset, my_offset);
        linear_node->second_child_offset = flatten_bvh_tree(node->children[1], offset, my_offset);
    }

    return my_offset;
}

MTS_VARIANT void BVH<Float, Spectrum>::find_split(std::vector<BVHPrimInfo> &primitive_info, int start, int end,
                ScalarBoundingBox3f &centroid_bbox, ScalarBoundingBox3f &node_bbox, ScalarCone3f &node_cone,
                int nb_buckets, int &split_dim, int &split_bucket, Float &min_cost) {

    split_dim = centroid_bbox.major_axis();
    split_bucket = 0;
    min_cost = std::numeric_limits<Float>::max();

    for (int dim = 0; dim < 3; dim++) {
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

            // TODO: CLEAN THIS METHOD
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

        for (int i = 0; i < nb_buckets - 1; i++) {
            if (cost[i] < min_cost) {
                min_cost = cost[i];
                split_bucket = i;
                split_dim = dim;
            }
        }
    }
}

MTS_VARIANT typename BVH<Float, Spectrum>::BVHNode*
BVH<Float, Spectrum>::create_leaf(std::vector<BVHPrimInfo> &primitive_info,
                                 int start,
                                 int end,
                                 std::vector<BVHPrimitive*> &ordered_prims,
                                 ScalarBoundingBox3f &prims_bbox,
                                 Spectrum intensity,
                                 ScalarCone3f cone) {
    BVHNode *leaf = new BVHNode();

    int nb_prim = end - start;
    int first_prim_offset = ordered_prims.size();
    for (int i = start; i < end; i++) {
        int prim_num = primitive_info[i].prim_number;
        ordered_prims.push_back(m_primitives[prim_num]);
    }

    leaf->init_leaf(first_prim_offset, nb_prim, prims_bbox, intensity, cone);
    m_leaf_count++;
    return leaf;
}

MTS_VARIANT void BVH<Float, Spectrum>::save_to_obj(std::string node_name, ScalarBoundingBox3f bbox, ScalarCone3f cone) {
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

MTS_VARIANT void BVH<Float, Spectrum>::cone_to_obj(std::string filename, ScalarPoint3f center, ScalarCone3f cone) {
    std::ofstream ofs(filename, std::ofstream::out);

    ofs << "# Vertices" << std::endl;

    // Center
    ofs << obj_vertex(center);

    // Circle
    ScalarVector3f a, b;
    std::tie(a, b) = coordinate_system(cone.axis);
    Point circle_center = center;
    Float scale_factor = tan(cone.normal_angle);

    float cone_scale_factor = CONE_SCALE_FACTOR;
    if (cone.normal_angle < M_PI_2f32 - std::numeric_limits<float>::epsilon()) {
        float cos_scale_factor = cos(cone.normal_angle) * cone_scale_factor;
        a *= scale_factor * cos_scale_factor;
        circle_center += cone.axis * cos_scale_factor;
    }

    if (scale_factor < std::numeric_limits<float>::epsilon()) {
        a = b * cone_scale_factor;
    }

    if (cone.normal_angle < std::numeric_limits<Float>::epsilon()) {
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


MTS_IMPLEMENT_CLASS_VARIANT(BVH, Object, "bvh")
MTS_INSTANTIATE_CLASS(BVH)
NAMESPACE_END(mitsuba)
