package dev.comfyfluffy.caustica.rt;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.comfyfluffy.caustica.rt.gen.PathReservoirData;
import dev.comfyfluffy.caustica.rt.gen.WorldPushData;
import org.junit.jupiter.api.Test;

final class RtPathReservoirHistoryTest {
    @Test
    void reflectedAbiIncludesReplayAndReconnectionGeometryLanes() {
        assertEquals(176, PathReservoirData.BYTE_SIZE);
        assertEquals(176, RtPathReservoirHistory.BYTES_PER_RESERVOIR);
        assertEquals(688, WorldPushData.BYTE_SIZE);
        assertEquals(603, RtPathReservoirHistory.SHIFTED_DIAGNOSTIC_COUNTER_COUNT);
        assertEquals(2412, RtPathReservoirHistory.SPATIAL_DIAGNOSTIC_COUNTER_BYTES);
        assertEquals(1 << 8, RtPathReservoirHistory.GUIDE_PREVIOUS_REPLAY_PASS_FLAG);
        assertEquals(1 << 9, RtPathReservoirHistory.GUIDE_PREVIOUS_AVAILABLE_FLAG);
        assertEquals(1 << 10, RtPathReservoirHistory.GUIDE_BRANCH_PREVIOUS_AVAILABLE_FLAG);
        assertEquals(1 << 11,
                RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_VALIDATE_PASS_FLAG);
        assertEquals(1 << 12,
                RtPathReservoirHistory.GUIDE_BRANCH_CANDIDATE_RETENTION_PASS_FLAG);
        assertEquals(1 << 13, RtPathReservoirHistory.GUIDE_BRANCH_AGE_REPLAY_PASS_FLAG);
        assertEquals(1 << 14,
                RtPathReservoirHistory.GUIDE_BRANCH_AGED_STORAGE_VALIDATE_PASS_FLAG);
        assertEquals(1 << 15,
                RtPathReservoirHistory.GUIDE_BRANCH_RECEIVER_OWNER_VALIDATE_PASS_FLAG);
        assertEquals(1 << 16,
                RtPathReservoirHistory.GUIDE_BRANCH_WINNER_STORAGE_VALIDATE_PASS_FLAG);
        assertEquals(1 << 17,
                RtPathReservoirHistory.GUIDE_BRANCH_WINNER_PREVIOUS_AVAILABLE_FLAG);
        assertEquals(1 << 18,
                RtPathReservoirHistory.GUIDE_BRANCH_WINNER_PREVIOUS_REPLAY_PASS_FLAG);
    }

    @Test
    void pingPongKeepsPreviousFrameSeparateFromWriteSlot() {
        RtHistoryState history = new RtHistoryState();
        RtPathReservoirHistory.State reservoirs = new RtPathReservoirHistory.State();

        RtPathReservoirHistory.Frame first = reservoirs.begin(history.beginFrame());
        assertEquals(0, first.writeSlot());
        assertEquals(1, first.scratchSlot());
        assertFalse(first.previousAvailable());
        reservoirs.commit(first);
        history.markProduced(first.generation());

        RtPathReservoirHistory.Frame second = reservoirs.begin(history.beginFrame());
        assertEquals(1, second.writeSlot());
        assertEquals(0, second.previousSlot());
        assertEquals(second.previousSlot(), second.scratchSlot());
        assertTrue(second.previousAvailable());
    }

    @Test
    void invalidationDropsPreviousPathReservoirs() {
        RtHistoryState history = new RtHistoryState();
        RtPathReservoirHistory.State reservoirs = new RtPathReservoirHistory.State();
        RtPathReservoirHistory.Frame first = reservoirs.begin(history.beginFrame());
        reservoirs.commit(first);
        history.markProduced(first.generation());

        history.invalidate(RtHistoryState.Reason.MATERIAL_EPOCH);
        RtPathReservoirHistory.Frame reset = reservoirs.begin(history.beginFrame());
        assertEquals(1, reset.writeSlot());
        assertEquals(-1, reset.previousSlot());
        assertFalse(reset.previousAvailable());
    }

    @Test
    void branchScratchStateAlternatesOnlyAcrossAdjacentCompatibleFrames() {
        RtPathReservoirHistory.BranchScratchState state =
                new RtPathReservoirHistory.BranchScratchState();
        RtPathReservoirHistory.Frame first = new RtPathReservoirHistory.Frame(
                7L, 0, -1, false);
        RtPathReservoirHistory.BranchScratchFrame firstScratch = state.begin(first, 10L);
        assertEquals(0, firstScratch.writeSlot());
        assertFalse(firstScratch.previousAvailable());
        state.commit(firstScratch, 10L, first.generation());

        RtPathReservoirHistory.Frame second = new RtPathReservoirHistory.Frame(
                7L, 1, 0, true);
        RtPathReservoirHistory.BranchScratchFrame secondScratch = state.begin(second, 11L);
        assertEquals(1, secondScratch.writeSlot());
        assertEquals(0, secondScratch.previousSlot());
        assertTrue(secondScratch.previousAvailable());

        RtPathReservoirHistory.Frame afterGap = new RtPathReservoirHistory.Frame(
                7L, 0, 1, true);
        RtPathReservoirHistory.BranchScratchFrame gapScratch = state.begin(afterGap, 13L);
        assertFalse(gapScratch.previousAvailable());
    }

    @Test
    void winnerScratchLifecycleIsIndependentFromBranchScratchLifecycle() {
        var branch = new RtPathReservoirHistory.BranchScratchState();
        var winner = new RtPathReservoirHistory.BranchScratchState();
        var first = new RtPathReservoirHistory.Frame(7L, 0, -1, false);
        var second = new RtPathReservoirHistory.Frame(7L, 1, 0, true);

        var firstBranch = branch.begin(first, 10L);
        branch.commit(firstBranch, 10L, first.generation());

        assertTrue(branch.begin(second, 11L).previousAvailable());
        assertFalse(winner.begin(second, 11L).previousAvailable());

        var firstWinner = winner.begin(first, 10L);
        winner.commit(firstWinner, 10L, first.generation());
        var secondWinner = winner.begin(second, 11L);
        assertTrue(secondWinner.previousAvailable());
        assertEquals(firstWinner.writeSlot(), secondWinner.previousSlot());
        assertEquals(1 - firstWinner.writeSlot(), secondWinner.writeSlot());
    }

    @Test
    void shiftedSnapshotRequiresOneFrameContinuityAndMatchingGeneration() {
        var snapshots = new RtPathReservoirHistory.ShiftedSnapshotState();
        var first = new RtPathReservoirHistory.Frame(7L, 0, -1, false);
        var second = new RtPathReservoirHistory.Frame(7L, 1, 0, true);
        var nextGeneration = new RtPathReservoirHistory.Frame(8L, 0, 1, true);

        assertFalse(snapshots.previousAvailable(first, 100L));
        snapshots.commit(first, 100L);
        assertTrue(snapshots.previousAvailable(second, 101L));
        assertFalse(snapshots.previousAvailable(second, 102L));
        assertFalse(snapshots.previousAvailable(nextGeneration, 101L));

        snapshots.reset();
        assertFalse(snapshots.previousAvailable(second, 101L));
    }

    @Test
    void memoryAccountingUsesTwoFullResolutionSlots() {
        long perSlot = RtPathReservoirHistory.bytesPerSlot(1280, 673);
        assertEquals(151_613_440L, perSlot);
        assertEquals(303_226_880L, Math.multiplyExact(perSlot, 2L));
        // View 20 lazily adds two separate full-record winner slots. They are not committed history
        // and cannot alias the persistent mapped snapshot that cross-frame replay reads.
        assertEquals(151_613_440L, RtPathReservoirHistory.bytesPerSlot(1280, 673));
        assertEquals(137_830_400L, Math.multiplyExact(1280L * 673L,
                dev.comfyfluffy.caustica.rt.gen.PathSourceRootData.BYTE_SIZE));
        assertEquals(6_891_520L,
                RtPathReservoirHistory.branchCandidateTagBytes(1280, 673));
        assertEquals(8, RtPathReservoirHistory.BRANCH_CANDIDATE_TAG_STRIDE);
        assertEquals(303_226_880L,
                RtPathReservoirHistory.branchWinnerScratchBytes(1280, 673));
        assertEquals(606_453_760L, Math.multiplyExact(
                RtPathReservoirHistory.branchWinnerScratchBytes(1280, 673),
                RtPathReservoirHistory.SLOT_COUNT));
        assertEquals(16, RtPathReservoirHistory.PATH_BRANCH_WINNER_TAG_STRIDE);
    }

    @Test
    void sortedPercentileInterpolatesBoundedDiagnosticSamples() {
        double[] values = {1.0, 2.0, 4.0, 8.0};
        assertEquals(3.0, RtPathReservoirHistory.sortedPercentile(values, 4, 0.5), 1.0e-12);
        assertEquals(7.4, RtPathReservoirHistory.sortedPercentile(values, 4, 0.95), 1.0e-12);
        assertEquals(8.0, RtPathReservoirHistory.sortedPercentile(values, 4, 1.0), 1.0e-12);
        assertThrows(IllegalArgumentException.class,
                () -> RtPathReservoirHistory.sortedPercentile(values, 0, 0.5));
    }
}
