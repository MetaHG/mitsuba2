#include <mitsuba/render/bvh.h>
#include <mitsuba/render/mesh.h>

NAMESPACE_BEGIN(mitsuba)

MTS_VARIANT BVH<Float, Spectrum>::BVH(host_vector<ref<Emitter>, Float> p, int max_prims_in_node, SplitMethod split_method, bool visualize_volumes):
    m_max_prims_in_node(std::min(255, max_prims_in_node)), m_split_method(split_method), m_visualize_volumes(visualize_volumes) {

    m_emitter_stats = std::vector<int>();
    m_primitives = std::vector<BVHPrimitive*>();
    for (size_t i = 0; i < p.size(); i++) {
        Shape *shape = p[i].get()->shape();
        if (shape && shape->is_mesh()) {
            Mesh *mesh = static_cast<Mesh*>(shape);
            for (ScalarIndex j = 0; j < mesh->face_count(); j++) {
                if (mesh->face_area(j) > 0) { // Don't add degenerate triangle with surface area of 0
                    m_primitives.push_back(new BVHPrimitive(p[i], j, mesh->face_bbox(j), mesh->face_cone(j)));
                    m_emitter_stats.push_back(0);
                } else {
//                    std::cout << "Face area is zero: skip face " << j << std::endl;
                }
            }
        } else {
            m_primitives.push_back(new BVHPrimitive(p[i]));
        }
    }

//        Spectrum total_radiance(0.0f);
//        for (BVHPrimitive* b: m_primitives) {
//            std::cout << b->get_total_radiance() << std::endl;
//            total_radiance += b->get_total_radiance();
//        }

//        std::cout << "Total radiance: " << total_radiance << std::endl;


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
    m_leaf_count = 0;
    int total_nodes = 0;
    std::vector<BVHPrimitive*> ordered_prims;
    BVHNode *root;

    root = recursive_build(prim_info, 0, m_primitives.size(), &total_nodes, ordered_prims, "_");
    m_primitives.swap(ordered_prims);

    m_prim_index_map = std::unordered_map<std::pair<std::string, ScalarIndex>, ScalarIndex, HashPair>();
    for (ScalarIndex i = 0; i < m_primitives.size(); i++) {
        m_prim_index_map.insert({std::pair(m_primitives[i]->emitter->id(), m_primitives[i]->face_id), i});
    }

    m_nodes = new LinearBVHNode[total_nodes]; // TODO: Allocate memory differently?
    m_total_nodes = total_nodes;

    int offset = 0;
    flatten_bvh_tree(root, &offset, -1);

    std::cout << "Prim count: " << m_primitives.size() << ", Leaf count: " << m_leaf_count << ", Total nodes: " << m_total_nodes << std::endl;
}

MTS_VARIANT BVH<Float, Spectrum>::~BVH() {
//    for (int count : m_emitter_stats) {
//        std::cout << count << ", ";
//    }
//    std::cout << std::endl;

    auto b = m_emitter_stats.begin();
    auto e = m_emitter_stats.end();
    auto q = m_emitter_stats.begin();
    std::advance(q, (int) (0.5 * m_emitter_stats.size()));
    std::nth_element(b, q, e);
    std::cout << "QUANTILE: " << *q << std::endl;

    delete m_nodes;
}

MTS_VARIANT std::pair<typename BVH<Float, Spectrum>::DirectionSample3f, Spectrum>
BVH<Float,Spectrum>::sample_emitter(const Float &tree_sample, const SurfaceInteraction3f &ref, const Point2f &emitter_sample, const Mask active) {
    float pdf = 1.0f;

    BVHPrimitive *prim = sample_tree(ref, pdf, tree_sample);
//    BVHPrimitive *prim = m_primitives[(int) tree_sample * m_primitives.size()];
//    pdf /= m_primitives.size();

    DirectionSample3f ds;
    Spectrum spec;
    std::tie(ds, spec) = prim->sample_direction(ref, emitter_sample, active); // TODO: ds.pdf is sometimes INF

    ds.pdf *= pdf;
    return std::pair(ds, spec / pdf);
}

MTS_VARIANT Float BVH<Float, Spectrum>::pdf_emitter_direction(const SurfaceInteraction3f &ref,
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

    return emitter_pdf * pdf_tree(ref, emitter, face_idx);
}

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

    int leaf_offset = m_nodes[offset].primitives_offset;
    int leaf_prim_count = m_nodes[offset].prim_count;

    int prim_offset = leaf_prim_count * sample;
    if (prim_offset == leaf_prim_count) {
        prim_offset -= 1;
    }

    prim_offset += leaf_offset;
    importance_ratio /= leaf_prim_count;

    m_emitter_stats[prim_offset] += 1;
    return m_primitives[prim_offset];
}

MTS_VARIANT Float BVH<Float, Spectrum>::pdf_tree(const SurfaceInteraction3f &si, const Emitter *emitter, const ScalarIndex face_idx) {
    Float pdf = 1.0;

    //TODO: Change this. HashMap like data structure. Unordered Map
//    typename std::vector<BVHPrimitive*>::iterator it = std::find_if(m_primitives.begin(), m_primitives.end(),
//                                                                    [emitter, face_idx](BVHPrimitive* p) {return p->emitter == emitter && p->face_id == face_idx; });
//    BVHPrimitive* prim = *it;
    ScalarIndex prim_idx = m_prim_index_map[std::pair(emitter->id(), face_idx)];
    BVHPrimitive* prim = m_primitives[prim_idx];

    int prev_offset;
    int current_offset = prim->leaf_offset;
    LinearBVHNode *current_node;


    while(current_offset != -1) {
        current_node = &m_nodes[current_offset];

        if (current_node->is_leaf()) {
            pdf /= current_node->prim_count;
        } else {
            float w_left, w_right;
            std::tie(w_left, w_right) = compute_children_weights(current_offset, si);

            float p_left = 0.5f;
            if (w_left + w_right >= std::numeric_limits<float>::epsilon()) {
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

MTS_VARIANT std::pair<Float, Float> BVH<Float, Spectrum>::compute_children_weights(int offset, const SurfaceInteraction3f &ref) {
    const LinearBVHNode &ln = m_nodes[offset + 1];
    const LinearBVHNode &rn = m_nodes[m_nodes[offset].second_child_offset];

    Float l_weight = compute_cone_weight(ln, ref);
    Float r_weight = compute_cone_weight(rn, ref);

    l_weight *= compute_luminance(ln.intensity);
    r_weight *= compute_luminance(rn.intensity);

//    Log(Info, "Intensity L weight: %s, Intensity R weight: %s\n", l_weight, r_weight);

//    Float left_d = ln.bbox.squared_distance(ref.p);
//    Float right_d = rn.bbox.squared_distance(ref.p);

//    Float distance_ratio = 1.0f; //TODO: DEFINE IT
//    if (left_d <= distance_ratio * squared_norm(ln.bbox.extents())
//        || right_d <= distance_ratio * squared_norm(rn.bbox.extents())) {
//        return std::pair(l_weight, r_weight);
//    }

    Float left_d = max(squared_norm(ln.bbox.extents()) / 4.0f, squared_norm(ln.bbox.center() - ref.p));
    Float right_d = max(squared_norm(rn.bbox.extents()) / 4.0f, squared_norm(rn.bbox.center() - ref.p));

//    Log(Info, "Left distance %s, right distance: %s", left_d, right_d);

    return std::pair(l_weight / left_d, r_weight / right_d);
}

MTS_VARIANT MTS_INLINE Float BVH<Float,Spectrum>::compute_cone_weight(const LinearBVHNode &node, const SurfaceInteraction3f &si){
    ScalarVector3f p_to_box_center = normalize(node.bbox.center() - si.p);

    Float in_angle = acos(dot(p_to_box_center, si.n));

    Float bangle = node.bbox.solid_angle(si.p);

    Float min_in_angle = max(in_angle - bangle, 0);

    Float caxis_p_angle = acos(dot(node.bcone.axis, -p_to_box_center));

    Float min_e_angle = max(caxis_p_angle - node.bcone.normal_angle - bangle, 0);

    Float cone_weight = 0;

    if (min_e_angle < node.bcone.emission_angle) {
//        cone_weight = abs(cos(min_in_angle)) * cos(min_e_angle); THIS WAS INCORRECT
        cone_weight = max(cos(min_in_angle), 0) * cos(min_e_angle);
    }

    return cone_weight;
}

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

                if (min_cost + std::numeric_limits<Float>::epsilon() < leaf_cost || (m_split_method != SplitMethod::SAOH && nb_prim > m_max_prims_in_node)) {
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

MTS_VARIANT int BVH<Float, Spectrum>::flatten_bvh_tree(BVHNode *node, int *offset, int parent_offset) {
    LinearBVHNode *linear_node = &m_nodes[*offset];
    linear_node->bbox = node->bbox;
    linear_node->bcone = node->bcone;
    linear_node->intensity = node->intensity;
    linear_node->parent_offset = parent_offset;
    int my_offset = (*offset)++;

    if (node->prim_count > 0) {
        linear_node->primitives_offset = node->first_prim_offset;
        linear_node->prim_count = node->prim_count;

        for (int i = 0; i < linear_node->prim_count; i++) {
            m_primitives[linear_node->primitives_offset + i]->leaf_offset = my_offset;
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


MTS_IMPLEMENT_CLASS_VARIANT(BVH, Object, "bvh")
MTS_INSTANTIATE_CLASS(BVH)
NAMESPACE_END(mitsuba)
