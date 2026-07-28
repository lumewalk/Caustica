package dev.comfyfluffy.caustica.rt;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertEquals;
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

    @Test
    void shaderPcgAndPathHashAreDeterministicForStoredSeeds() {
        assertEquals(0x995312E1,
                RtPathReplayReference.pcg(0x12345678));
        assertEquals(RtPathReplayReference.pcg(0x12345678),
                RtPathReplayReference.pathHash(0x12345678));
        assertEquals(RtPathReplayReference.pcg(0x12345678),
                RtPathReplayReference.pcg(0x12345678));
    }

    @Test
    void seededSegmentsReproduceTerminalStatesAndAggregateInShaderOrder() {
        var first = new RtPathReplayReference.Segment(
                0x13579BDF, 0x2468ACE0, 0x10203040, 3, 2);
        var second = new RtPathReplayReference.Segment(
                0xCAFEBABE, 0x0BADF00D, 0x55667788, 4, 1);

        var firstState = RtPathReplayReference.replaySegment(first);
        var aggregate = RtPathReplayReference.aggregate(
                RtPathReplayReference.pathHash(0x7F4A7C15), first, second);
        var repeat = RtPathReplayReference.aggregate(
                RtPathReplayReference.pathHash(0x7F4A7C15), first, second);
        var changedSeed = RtPathReplayReference.aggregate(
                RtPathReplayReference.pathHash(0x7F4A7C15),
                new RtPathReplayReference.Segment(
                        first.replaySeed() + 1, first.proposalReplaySeed() + 1,
                        first.topologyKey(), first.transportDraws(), first.proposalDraws()),
                second);

        assertEquals(RtPathReplayReference.pcg(
                RtPathReplayReference.pcg(RtPathReplayReference.pcg(first.replaySeed()))),
                firstState.replayState());
        assertEquals(aggregate, repeat);
        assertEquals(2, aggregate.segmentCount());
        assertTrue(aggregate.replayState() != changedSeed.replayState());
        assertTrue(aggregate.proposalReplayState() != changedSeed.proposalReplayState());
    }

    @Test
    void mismatchBitsRemainStableAndIndependent() {
        assertEquals(1 << 0, RtPathReplayReference.MISMATCH_EMPTY);
        assertEquals(1 << 8, RtPathReplayReference.MISMATCH_METADATA);
        assertEquals(RtPathReplayReference.MISMATCH_PATH_STATE
                        | RtPathReplayReference.MISMATCH_PROPOSAL_STATE,
                0x18);
    }
}
