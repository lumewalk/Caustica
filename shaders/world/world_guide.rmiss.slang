// Minimal miss record for reflection/refraction guide probes. These rays consume only hit-vs-miss
// state; fixed sky guide values are supplied by raygen, so running the atmosphere shader is wasted work.
import world_common;

[shader("miss")]
void main(inout Payload payload) {
    payload.hitT = -1.0;
    payload.normal = half3(0.0h, 0.0h, 0.0h);
    payload.flags = 0u;
}
