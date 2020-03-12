#pragma once

#include <mitsuba/core/bbox.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/vector.h>

#include <mitsuba/render/emitter.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>

#include <fstream>
#include <filesystem>

NAMESPACE_BEGIN(mitsuba)


template <typename Float, typename Spectrum, typename BoundingBox>
class LightTree: public Object {
public:
    MTS_IMPORT_TYPES(Emitter, Mesh, Sampler) // IndependentSampler apparently does not exist

    LightTree(host_vector<ref<Emitter>, Float> emitters) {
        std::vector<LightNode*> leaves;

        for (Emitter* emitter: emitters) {
            leaves.push_back(new LightNode(emitter));
        }

        m_tree = build_tree(leaves);
    }

    ~LightTree() {
        //TODO
    }

    std::vector<std::pair<DirectionSample3f, Spectrum>> sample_emitters_directions(const BSDFContext &ctx, const SurfaceInteraction3f &si, const Point2f &sample_, const Mask active) const {
        struct PriorityQLightElem {
            PriorityQLightElem(LightNode *n, float p_subtree) {
                node = n;
                float importance_ratio;
                emitter = n->sample_emitter(importance_ratio, std::rand()); // Should use sampler instead of rand
                std::tie(ds, illumination_estimate) = emitter->sample_direction(si, sample_, active);
                illumination_estimate *= p_subtree / importance_ratio;
                error_bound = n->get_intensity() * emitter->get_geometry_factor() * si->bsdf()->eval(ctx, si, ds, active); // Check if has to take opposite direction of ds
            }

            LightNode* node;
            Emitter* emitter;
            DirectionSample3f ds;
            Spectrum illumination_estimate;
            Spectrum error_bound;
        };

        auto priority_q_light_elem_comp = [](const PriorityQLightElem& n1, const PriorityQLightElem& n2) { return n1.error_bound > n2.error_bound; };
        std::priority_queue<PriorityQLightElem, 
                            std::vector<PriorityQLightElem>,
                            decltype(priority_q_light_elem_comp)> queue(priority_q_light_elem_comp);
        
        // Push initial node
        PriorityQLightElem root_elem(m_tree, 1.0);
        queue.push(root_elem);
        Spectrum total_illumination_estimate = root_elem.illumination_estimate;

        while (queue.top().error_bound > total_illumination_estimate * 0.02) { // Error ratio. TODO: To define somewhere to avoid magic numbers
            auto parent = queue.pop();
            total_illumination_estimate -= parent.illumnation_estimate;
            
            LightNode* left_child, right_child;
            std::tie(left_child, right_child) = parent.node->get_children();
            
            PriorityQLightElem leftElem(parent.node->left_child, 0.0); // TODO: Need to compute the correct subtree probability 
                                                                    // (Do an additional pass from top to bottom to compute subtree probabilities for all probabilities in the build tree function)
            PriorityQLightElem rightElem(parent.node->right_child, 0.0); // TODO: Need to compute the correct subtree probability
            
            queue.push(leftElem);
            queue.push(rightElem);
            
            total_illumination_estimate += leftElem.illumination_estimate;
            total_illumination_estimate += rightElem.illumination_estimate;
        }
        
        // Should return for each emitter the illumination estimate of the corresponding emitter, as well as the direction sampled of this emitter.
        return NULL;
    }

    std::string to_string() const override {
        std::cout << "LighTree: Printing tree.." << std::endl;
        return m_tree->to_string();
    }

    void to_obj() {
        m_tree->to_obj(0, 0);
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
        LightNode(LightNode* left, LightNode* right, float sample) {
            m_left = left;
            m_right = right;
            m_intensity = left->get_intensity() + right->get_intensity();
            float left_representative_prob = left->get_intensity() / m_intensity;
            m_representative = left_representative_prob < sample ? left->get_representative() : right->get_representative();
            m_bbox = BoundingBox::merge(left->m_bbox, right->m_bbox);
        }

        ~LightNode() {
            //TODO
        }

        Emitter* sample_emitter(float &importance_ratio, float sample) {
            importance_ratio = 1.0;

            LightNode* current = this;
            while (!current->is_leaf()) {
                float w_left = current->m_left->compute_weight();
                float w_right = current->m_right->compute_weight();

                float p_left = w_left / (w_left + w_right);
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

        float get_intensity() {
            return m_intensity;
        }

        std::pair<LightNode*, LightNode*> get_children() {
            return std::pair<LightNode*, LightNode*>(m_left, m_right);
        }

        static float compute_cluster_metric(LightNode* n1, LightNode* n2) {
            BoundingBox bbox = BoundingBox::merge(n1->m_bbox, n2->m_bbox);
            return (n1->get_intensity() + n2->get_intensity()) * squared_norm(bbox.extents());
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

        void to_obj(int left_str, int right_str) {
            if (m_left) {
                m_left->to_obj(left_str + 1, right_str);
            }

            save_to_obj(left_str, right_str);

            if (m_right) {
                m_right->to_obj(left_str, right_str + 1);
            }
        }

    private:
        bool is_leaf() {
            return m_right && m_left;
        }

        float compute_weight() {
            return m_intensity;
        }

        void save_to_obj(int left_str, int right_str) {
            std::string dir_name = "lighttree_bboxes";
            std::ostringstream oss;
            oss << dir_name << "/l" << left_str << "_r" << right_str << ".obj";

            std::filesystem::path dir(dir_name);
            if (!std::filesystem::exists(dir)) {
                std::filesystem::create_directory(dir);
            }

            std::ofstream ofs(oss.str(), std::ofstream::out);

            for (size_t i = 0; i < 8; i++) { // Magic number here: TODO DEFINE: 8 = number of corners in bounding box
                Point p = m_bbox.corner(i);
                ofs << "v " << p.x() << " " << p.y() << " " << p.z() << std::endl;
            }

            // TODO: See if faces declaration are necessary

            ofs.close();
        }


    private:
        float m_intensity;
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

            LightNode* newCluster = new LightNode(best_n1, best_n2, std::rand()); // Wanted to use independent sampler here, but didn't manage to import it...

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
};


NAMESPACE_END(mitsuba)
