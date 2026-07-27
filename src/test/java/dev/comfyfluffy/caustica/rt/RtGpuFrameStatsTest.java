package dev.comfyfluffy.caustica.rt;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

final class RtGpuFrameStatsTest {
    @Test
    void computesOrdinaryTimestampDelta() {
        assertEquals(150L, RtGpuFrameStats.timestampDelta(100L, 250L, 64));
    }

    @Test
    void computesTimestampDeltaAcrossCounterWrap() {
        assertEquals(11L, RtGpuFrameStats.timestampDelta(250L, 5L, 8));
    }

    @Test
    void convertsTimestampTicksToMilliseconds() {
        assertEquals(0.000225, RtGpuFrameStats.millis(150L, 1.5), 1.0e-12);
    }

    @Test
    void rejectsInvalidTimestampWidth() {
        assertThrows(IllegalArgumentException.class, () -> RtGpuFrameStats.timestampDelta(0L, 1L, 0));
        assertThrows(IllegalArgumentException.class, () -> RtGpuFrameStats.timestampDelta(0L, 1L, 65));
    }
}
