#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/endpoint.h>

NAMESPACE_BEGIN(mitsuba)

MTS_VARIANT Emitter<Float, Spectrum>::Emitter(const Properties &props) : Base(props) { }
MTS_VARIANT Spectrum Emitter<Float, Spectrum>::get_total_radiance() const { NotImplementedError("get_total_radiance"); }
MTS_VARIANT Spectrum Emitter<Float, Spectrum>::get_radiance() const { NotImplementedError("get_radiance"); }
MTS_VARIANT Emitter<Float, Spectrum>::~Emitter() { }

MTS_IMPLEMENT_CLASS_VARIANT(Emitter, Endpoint, "emitter")
MTS_INSTANTIATE_CLASS(Emitter)
NAMESPACE_END(mitsuba)
