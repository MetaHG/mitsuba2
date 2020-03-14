#pragma once

#include <mitsuba/core/bbox.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/plugin.h>

#include <mitsuba/render/emitter.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>

#include <fstream>
#include <filesystem>
#include <queue>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum, typename BoundingBox>
class LightTree: public Object {
public:
    MTS_IMPORT_TYPES(Emitter, Mesh, Sampler) // IndependentSampler apparently does not exist

    LightTree(host_vector<ref<Emitter>, Float> emitters) {
        Properties sampler_props("independent");
        sampler_props.set_int("sample_count", 4);
        m_sampler = static_cast<Sampler *>(PluginManager::instance()->create_object<Sampler>(sampler_props));

        std::vector<LightNode*> leaves;
        for (Emitter* emitter: emitters) {
            leaves.push_back(new LightNode(emitter));
            std::cout << emitter->get_total_radiance() << std::endl;
        }

        m_tree = build_tree(leaves);
    }

    ~LightTree() {
        //TODO
    }

    std::pair<DirectionSample3f, Spectrum> sample_emitter(const Float &tree_sample, const Interaction3f &ref, const Point2f &emitter_sample, const Mask active) {
        float pdf = 1.0f;
        Emitter* emitter = m_tree->sample_emitter(ref, pdf, tree_sample);

        DirectionSample3f ds;
        Spectrum spec;
        std::tie(ds, spec) = emitter->sample_direction(ref, emitter_sample, active);

        ds.pdf *= pdf;
        return std::pair(ds, spec / pdf);
    }

    std::string to_string() const override {
        std::cout << "LighTree: Printing tree.." << std::endl;
        return m_tree->to_string();
    }

    void to_obj() {
        m_tree->to_obj("_");
    }

protected:
    class LightNode : public Object {
    public:
        /// Create light tree leaf
        LightNode(Emitter* emitter) {
            m_left = nullptr;
            m_right = nullptr;
            m_representative = emitter;
            m_intensity = emitter->get_total_radiance();
            m_bbox = emitter->bbox();
        }

        /// Create light tree node
        LightNode(LightNode* left, LightNode* right, Float sample) {
            m_left = left;
            m_right = right;
            m_intensity = left->get_intensity() + right->get_intensity();
            float left_representative_prob = compute_luminance(left->get_intensity()) / compute_luminance(m_intensity);
            m_representative = left_representative_prob < sample ? left->get_representative() : right->get_representative();
            m_bbox = BoundingBox::merge(left->m_bbox, right->m_bbox);
        }

        ~LightNode() {
            //TODO
        }

        Emitter* sample_emitter(const Interaction3f &ref, float &importance_ratio, const Float &sample_) {
            importance_ratio = 1.0;
            Float sample(sample_);

            LightNode* current = this;

            while (!current->is_leaf()) {
                float w_left, w_right;
                std::tie(w_left, w_right) = current->compute_weights(ref);

                float p_left = 0.5f;
                if (w_left + w_right >= std::numeric_limits<float>::epsilon()) {
                    p_left = w_left / (w_left + w_right);
                }

                if (sample <= p_left) {
                    current = current->m_left;
                    sample = sample / p_left;
                    importance_ratio *= p_left;
                } else {
                    current = current->m_right;
                    float p_right = (1 - p_left);
                    sample = (sample - p_left) / p_right;
                    importance_ratio *= p_right;
                }
            }

            return current->get_representative();
        }

        Emitter* get_representative() {
            return m_representative;
        }

        Spectrum get_intensity() {
            return m_intensity;
        }

        static float compute_luminance(Spectrum intensity) {
            return hmean(intensity);
        }

        std::pair<LightNode*, LightNode*> get_children() {
            return std::pair<LightNode*, LightNode*>(m_left, m_right);
        }

        static float compute_cluster_metric(LightNode* n1, LightNode* n2) {
            BoundingBox bbox = BoundingBox::merge(n1->m_bbox, n2->m_bbox);
            Spectrum intensity = n1->get_intensity() + n2->get_intensity();
            return hmean(intensity) * squared_norm(bbox.extents());
        }

        std::string to_string() const override {
            // With a high number of lights, this recusrive algorithm could provoke call stack overflows ?
            // Should probably use while loop and such instead to avoid this problem.
            std::ostringstream oss;
            if (m_left) {
                oss << "LEFT" << std::endl
                    << m_left->to_string();
            }

            oss << "LightNode:" << std::endl
                 << "   Intensity: " << m_intensity << std::endl;
                // << "   Representative Emitter: \n\t" << m_representative->to_string() << std::endl;

            if (m_right) {
                oss << "RIGHT" << std::endl
                    << m_right->to_string();
            }

            return oss.str();
        }

        void to_obj(std::string str) {
            if (m_left) {
                m_left->to_obj(str + "l");
            }

            save_to_obj(str);

            if (m_right) {
                m_right->to_obj(str + "r");
            }
        }

    private:
        bool is_leaf() {
            return !(m_right && m_left);
        }

        std::pair<float, float> compute_weights(const Interaction3f &ref) {
            float left_d = m_left->m_bbox.distance(ref.p);
            float right_d = m_right->m_bbox.distance(ref.p);

            float left_lumi = compute_luminance(m_left->m_intensity);
            float right_lumi = compute_luminance(m_right->m_intensity);

            float distance_ratio = 1.0f; //TODO: DEFINE IT
            if (left_d <= distance_ratio * norm(m_left->m_bbox.extents())
                || right_d <= distance_ratio * norm(m_right->m_bbox.extents())) {
                return std::pair(left_lumi, right_lumi);
            }

            return std::pair(left_lumi * (1.0f / (left_d * left_d)), right_lumi * (1.0f / (right_d * right_d)));
        }

        void save_to_obj(std::string str) {
            std::string dir_name = "lighttree_bboxes";

            std::filesystem::path dir(dir_name);
            if (!std::filesystem::exists(dir)) {
                std::filesystem::create_directory(dir);
            }

            std::ofstream ofs(dir_name + "/" + str + ".obj", std::ofstream::out);

            ofs << "# Vertices" << std::endl;
            for (size_t i = 0; i < 8; i++) { // Magic number here: TODO DEFINE: 8 = number of corners in bounding box
                Point p = m_bbox.corner(i);
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


    private:
        Spectrum m_intensity;
        Emitter* m_representative; // Reprensentative light
        LightNode* m_left;
        LightNode* m_right;
        BoundingBox m_bbox;
        // BoundingCone m_bcone;
    };

protected:
    LightNode* build_tree(std::vector<LightNode*> leaves) {
        std::cout << "LightTree: Building.." << std::endl;
        std::set<LightNode*> leaves_set(leaves.begin(), leaves.end());
        LightNode* root;

        while (leaves_set.size() > 1) {
            LightNode* best_n1;
            LightNode* best_n2;
            float best_metric = std::numeric_limits<float>::max();

            for (LightNode* n1: leaves_set) {
                for (LightNode* n2: leaves_set) {
                    if (n1 != n2) {
                        float metric = LightNode::compute_cluster_metric(n1, n2);
                        if (metric < best_metric) {
                            best_n1 = n1;
                            best_n2 = n2;
                            best_metric = metric;
                        }
                    }
                }
            }

            LightNode* newCluster = new LightNode(best_n1, best_n2, m_sampler->next_1d());

            leaves_set.erase(best_n1);
            leaves_set.erase(best_n2);
            leaves_set.insert(newCluster);

            root = newCluster;
        }

        std::cout << "LightTree: Build Finished" << std::endl;

        return root;
    }

protected:
    LightNode* m_tree;
    ref<Sampler> m_sampler;
};


NAMESPACE_END(mitsuba)
