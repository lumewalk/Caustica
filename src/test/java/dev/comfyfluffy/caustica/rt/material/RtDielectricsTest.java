package dev.comfyfluffy.caustica.rt.material;

import net.minecraft.resources.Identifier;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class RtDielectricsTest {
    @Test
    void iceFamilyRefractsAsAVolume() {
        for (String sprite : new String[]{"block/ice", "block/packed_ice", "block/blue_ice",
                "block/frosted_ice_0", "block/frosted_ice_3"}) {
            var dielectric = RtDielectrics.forSprite(Identifier.parse("minecraft:" + sprite));
            assertTrue(dielectric.volume(), sprite + " must be a volume dielectric");
            assertEquals(RtDielectrics.ICE_IOR, dielectric.ior(), 1.0e-6f, sprite);
        }
    }

    @Test
    void windowGlassStaysThinSoItReadsAsAWindow() {
        var glass = RtDielectrics.forSprite(Identifier.parse("minecraft:block/glass"));
        assertFalse(glass.volume());
        assertEquals(RtDielectrics.GLASS_IOR, glass.ior(), 1.0e-6f);
    }

    @Test
    void unknownAndNullSpritesFallBackToThinGlass() {
        var modded = RtDielectrics.forSprite(Identifier.parse("somemod:block/weird_crystal"));
        assertFalse(modded.volume());
        assertEquals(RtDielectrics.GLASS_IOR, modded.ior(), 1.0e-6f);
        assertEquals(RtDielectrics.defaultGlass(), RtDielectrics.forSprite(null));
    }

    @Test
    void iceRefractsLessStronglyThanGlass() {
        // Ice Ih sits just below liquid water, which sits well below soda-lime glass. Getting this
        // ordering wrong is the whole reason the per-model constant was not good enough.
        assertTrue(RtDielectrics.ICE_IOR < RtDielectrics.WATER_IOR);
        assertTrue(RtDielectrics.WATER_IOR < RtDielectrics.GLASS_IOR);
    }
}
