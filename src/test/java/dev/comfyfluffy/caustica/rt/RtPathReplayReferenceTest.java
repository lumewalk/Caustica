package dev.comfyfluffy.caustica.rt;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

final class RtPathReplayReferenceTest {
    private static RtPathReplayReference.State state(int segments, long transport, long proposal) {
        return new RtPathReplayReference.State(
                RtPathReplayReference.REPLAY_VERSION, segments, transport, proposal);
    }

    @Test
    void replayRequiresBothRandomStreamsAndTheSameWavefrontTopology() {
        var receiver = state(2, 101L, 202L);
        assertTrue(RtPathReplayReference.compatible(receiver, state(2, 101L, 202L)));
        assertFalse(RtPathReplayReference.compatible(receiver, state(2, 102L, 202L)));
        assertFalse(RtPathReplayReference.compatible(receiver, state(2, 101L, 203L)));
        assertFalse(RtPathReplayReference.compatible(receiver, state(1, 101L, 202L)));
    }

    @Test
    void replayStateRejectsUnsupportedVersionsAndSegmentCounts() {
        assertThrows(IllegalArgumentException.class,
                () -> new RtPathReplayReference.State(1, 1, 1L, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> state(0, 1L, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> state(RtPathReplayReference.MAX_SEGMENTS + 1, 1L, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> state(1, -1L, 1L));
    }
}
