#include <random>
#include <enoki/stl.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class EmitterIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth)
    MTS_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)


    EmitterIntegrator(const Properties &props) : Base(props) { }

    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     Sampler * sampler,
                                     const RayDifferential3f &ray,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        Spectrum result(0.f);

        // ---------------------- First intersection ----------------------

        SurfaceInteraction3f si = scene->ray_intersect(ray, active);
        Mask valid_ray = si.is_valid();

         // --------------------- Emitter sampling ---------------------

        auto [ds, emitter_val] = scene->sample_emitter_direction_pure(
         si, sampler->next_2d(active), sampler->next_1d(active), true, active);
        Mask active_e = active && neq(ds.pdf, 0.f);

        result[active_e] += emitter_val;

        return { result, valid_ray };
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return std::string("EmitterIntegrator[]");
    }

    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_VARIANT(EmitterIntegrator, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(EmitterIntegrator, "Emitter Sampling integrator");
NAMESPACE_END(mitsuba)
