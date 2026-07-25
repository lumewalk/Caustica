package dev.comfyfluffy.caustica.rt.material;

import net.minecraft.resources.Identifier;

import java.util.Map;

/**
 * Built-in refractive indices and thin/volume classification for the dielectric blocks vanilla ships.
 *
 * <p>Two properties decide how {@code world.rgen}'s dielectric interface behaves, and both are physical
 * facts about the material rather than anything derivable from the render layer:
 *
 * <ul>
 *   <li><b>IOR</b> drives Snell refraction and the Fresnel split. Everything translucent used to share a
 *       single per-model constant (1.52 for glass, 1.333 for water), so ice refracted like window glass.
 *   <li><b>Volume vs thin.</b> A volume dielectric bends the ray and pushes a participating medium whose
 *       extinction attenuates the segment inside it. A thin one is a collapsed slab: its two interfaces
 *       cancel, so the ray passes straight through and the tint is applied once. This is a deliberate
 *       modelling choice, not a measurement — a glass block is geometrically a cube, but treating it as a
 *       solid refracting cube makes windows unreadable, so vanilla-style glass stays thin while ice
 *       (which reads as a solid block of frozen water) becomes a volume.
 * </ul>
 *
 * <p>Sprite-keyed rather than block-keyed because the material registry compiles per sprite; a resource
 * pack that renames textures simply falls back to the per-model default, and a
 * {@code caustica/materials/*.json} rule overrides either value explicitly.
 */
public final class RtDielectrics {
    private RtDielectrics() {}

    /** Soda-lime glass. The default for anything translucent that is not otherwise classified. */
    public static final float GLASS_IOR = 1.52f;
    /** Fresh water at room temperature; also the fluid singleton's index. */
    public static final float WATER_IOR = 1.333f;
    /** Ice Ih, slightly below liquid water. */
    public static final float ICE_IOR = 1.309f;

    /**
     * How a dielectric interface behaves. {@code volume} true means refract and track a medium.
     */
    public record Dielectric(float ior, boolean volume) {}

    private static final Dielectric THIN_GLASS = new Dielectric(GLASS_IOR, false);
    private static final Dielectric VOLUME_ICE = new Dielectric(ICE_IOR, true);
    public static final Dielectric WATER = new Dielectric(WATER_IOR, true);

    // Ice is the one vanilla family that genuinely reads as a solid transparent block, so it is the one
    // that earns the volume treatment. Slime and honey are translucent but visually gelatinous rather
    // than refractive, and nether portal is an emissive effect, so they stay thin.
    private static final Map<String, Dielectric> BY_SPRITE = Map.of(
            "block/ice", VOLUME_ICE,
            "block/packed_ice", VOLUME_ICE,
            "block/blue_ice", VOLUME_ICE,
            "block/frosted_ice_0", VOLUME_ICE,
            "block/frosted_ice_1", VOLUME_ICE,
            "block/frosted_ice_2", VOLUME_ICE,
            "block/frosted_ice_3", VOLUME_ICE);

    /**
     * Built-in dielectric for a sprite, or the thin-glass default when the sprite is unrecognised. Only
     * meaningful for materials the mesher classified as translucent; opaque materials ignore it.
     */
    public static Dielectric forSprite(Identifier spriteName) {
        if (spriteName == null) return THIN_GLASS;
        Dielectric known = BY_SPRITE.get(spriteName.getPath());
        return known != null ? known : THIN_GLASS;
    }

    /** The default used for fallback variants compiled without a sprite. */
    public static Dielectric defaultGlass() {
        return THIN_GLASS;
    }
}
