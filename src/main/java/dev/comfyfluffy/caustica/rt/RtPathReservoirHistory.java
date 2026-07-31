package dev.comfyfluffy.caustica.rt;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.gen.PathReservoirData;
import dev.comfyfluffy.caustica.rt.gen.PathSourceRootData;
import dev.comfyfluffy.caustica.rt.pipeline.RtPathTemporalPipeline;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Arrays;
import java.util.Locale;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;

/**
 * Double-buffered GPU storage and temporal-admission diagnostics for ReSTIR PT path reservoirs.
 *
 * <p>Candidate generation remains in the wavefront ray pass. This owner validates historical
 * reprojection and strict path compatibility, while replay/reconnection and the eventual GRIS merge
 * remain explicit later stages.</p>
 */
final class RtPathReservoirHistory {
    static final int SLOT_COUNT = 2;
    static final int BYTES_PER_RESERVOIR = PathReservoirData.BYTE_SIZE;
    static final int SPATIAL_DEBUG_VIEW = 17;
    static final int SPATIAL_POLICY_DEBUG_VIEW = 18;
    static final int RECONNECTION_DEBUG_VIEW = 19;
    static final int SHIFTED_RADIANCE_DEBUG_VIEW = 20;
    static final int MAPPING_REPLAY_PASS_FLAG = 1 << 5;
    static final int CROSS_FRAME_MAPPING_REPLAY_PASS_FLAG = 1 << 6;
    static final int SPATIAL_DIAGNOSTIC_CATEGORY_COUNT = 9;
    static final int SPATIAL_DIAGNOSTIC_STRICT_PAIR_CURSOR_INDEX =
            SPATIAL_DIAGNOSTIC_CATEGORY_COUNT;
    static final int SPATIAL_DIAGNOSTIC_LIMITED_ADMITTED_INDEX = 10;
    static final int SPATIAL_DIAGNOSTIC_TOPOLOGY_RESCUED_INDEX = 11;
    static final int SPATIAL_DIAGNOSTIC_RESCUED_PAIR_CURSOR_INDEX = 12;
    static final int SPATIAL_DIAGNOSTIC_COUNTER_COUNT = 13;
    static final int SHIFTED_DIAGNOSTIC_STATE_COUNT = 15;
    static final int SHIFTED_DIAGNOSTIC_SELECTION_ELIGIBLE_INDEX = 15;
    static final int SHIFTED_DIAGNOSTIC_SOURCE_SELECTED_INDEX = 16;
    static final int SHIFTED_DIAGNOSTIC_SCRATCH_WRITTEN_INDEX = 17;
    static final int SHIFTED_DIAGNOSTIC_SCRATCH_SELECTED_INDEX = 18;
    static final int SHIFTED_DIAGNOSTIC_SCRATCH_INVALID_INDEX = 19;
    static final int SHIFTED_DIAGNOSTIC_PAIR_CURSOR_INDEX = 20;
    static final int MAPPING_REPLAY_ELIGIBLE_INDEX = 21;
    static final int MAPPING_REPLAY_ACCEPTED_INDEX = 22;
    static final int MAPPING_REPLAY_ABI_REJECT_INDEX = 23;
    static final int MAPPING_REPLAY_SOURCE_REJECT_INDEX = 24;
    static final int MAPPING_REPLAY_RECEIVER_REJECT_INDEX = 25;
    static final int MAPPING_REPLAY_GEOMETRY_REJECT_INDEX = 26;
    static final int MAPPING_REPLAY_PDF_REJECT_INDEX = 27;
    static final int MAPPING_REPLAY_VISIBILITY_REJECT_INDEX = 28;
    static final int MAPPING_REPLAY_RADIANCE_REJECT_INDEX = 29;
    static final int SHIFTED_SOURCE_ROOT_WRITTEN_INDEX = 30;
    static final int SHIFTED_SOURCE_ROOT_INVALID_INDEX = 31;
    static final int MAPPING_REPLAY_SOURCE_ROOT_REJECT_INDEX = 32;
    static final int CROSS_FRAME_ATTEMPTED_INDEX = 33;
    static final int CROSS_FRAME_RECEIVER_REPROJECTION_REJECT_INDEX = 34;
    static final int CROSS_FRAME_MAPPED_EMPTY_INDEX = 35;
    static final int CROSS_FRAME_ELIGIBLE_INDEX = 36;
    static final int CROSS_FRAME_ACCEPTED_INDEX = 37;
    static final int CROSS_FRAME_ABI_REJECT_INDEX = 38;
    static final int CROSS_FRAME_SOURCE_ROOT_REJECT_INDEX = 39;
    static final int CROSS_FRAME_SOURCE_REPROJECTION_REJECT_INDEX = 40;
    static final int CROSS_FRAME_SOURCE_REPLAY_REJECT_INDEX = 41;
    static final int CROSS_FRAME_RECEIVER_SURFACE_REJECT_INDEX = 42;
    static final int CROSS_FRAME_GEOMETRY_REJECT_INDEX = 43;
    static final int CROSS_FRAME_PDF_REJECT_INDEX = 44;
    static final int CROSS_FRAME_VISIBILITY_REJECT_INDEX = 45;
    static final int CROSS_FRAME_RADIANCE_REJECT_INDEX = 46;
    static final int CROSS_FRAME_RECEIVER_SAMPLE_REJECT_INDEX = 47;
    static final int CROSS_FRAME_RECEIVER_EDGE_REJECT_INDEX = 48;
    static final int CROSS_FRAME_RECEIVER_TOPOLOGY_REJECT_INDEX = 49;
    static final int CROSS_FRAME_RECEIVER_DEPTH_REJECT_INDEX = 50;
    static final int CROSS_FRAME_RECEIVER_TRANSPORT_REJECT_INDEX = 51;
    static final int CROSS_FRAME_RECEIVER_FOOTPRINT_REJECT_INDEX = 52;
    static final int CROSS_FRAME_SAMPLE_RESCUE_ELIGIBLE_INDEX = 53;
    static final int CROSS_FRAME_SAMPLE_RESCUED_INDEX = 54;
    static final int CROSS_FRAME_SAMPLE_RESCUE_EDGE_REJECT_INDEX = 55;
    static final int CROSS_FRAME_SAMPLE_RESCUE_TOPOLOGY_REJECT_INDEX = 56;
    static final int CROSS_FRAME_SAMPLE_RESCUE_DEPTH_REJECT_INDEX = 57;
    static final int CROSS_FRAME_SAMPLE_RESCUE_TRANSPORT_REJECT_INDEX = 58;
    static final int CROSS_FRAME_SAMPLE_RESCUE_FOOTPRINT_REJECT_INDEX = 59;
    static final int CROSS_FRAME_EDGE_BREAKDOWN_ELIGIBLE_INDEX = 60;
    static final int CROSS_FRAME_EDGE_MISSING_VALID_INDEX = 61;
    static final int CROSS_FRAME_EDGE_DEPTH_REJECT_INDEX = 62;
    static final int CROSS_FRAME_EDGE_EVENT_REJECT_INDEX = 63;
    static final int CROSS_FRAME_EDGE_MAPPING_REJECT_INDEX = 64;
    static final int CROSS_FRAME_EDGE_PDF_REJECT_INDEX = 65;
    static final int CROSS_FRAME_EDGE_FINITE_REJECT_INDEX = 66;
    static final int SHIFTED_DIAGNOSTIC_COUNTER_COUNT = 67;
    static final int SPATIAL_DIAGNOSTIC_COUNTER_BYTES =
            SHIFTED_DIAGNOSTIC_COUNTER_COUNT * Integer.BYTES;
    static final int SPATIAL_DIAGNOSTIC_PAIR_CAPACITY = 4096;
    static final int SHIFTED_DIAGNOSTIC_DENSITY_PAIR_OFFSET =
            SPATIAL_DIAGNOSTIC_PAIR_CAPACITY;
    static final int SHIFTED_DIAGNOSTIC_MERGE_PAIR_OFFSET =
            SPATIAL_DIAGNOSTIC_PAIR_CAPACITY * 2;
    static final int SHIFTED_DIAGNOSTIC_SELECTION_PAIR_OFFSET =
            SPATIAL_DIAGNOSTIC_PAIR_CAPACITY * 3;
    static final int SHIFTED_DIAGNOSTIC_PAIR_FLOAT_COUNT =
            SPATIAL_DIAGNOSTIC_PAIR_CAPACITY * 8;
    static final int SPATIAL_DIAGNOSTIC_PAIR_BYTES =
            SHIFTED_DIAGNOSTIC_PAIR_FLOAT_COUNT * Float.BYTES;

    record Frame(long generation, int writeSlot, int previousSlot, boolean previousAvailable) {
        int finalSlot() {
            return writeSlot;
        }

        int scratchSlot() {
            return 1 - writeSlot;
        }
    }

    static final class State {
        private int latestSlot = -1;
        private long latestGeneration = Long.MIN_VALUE;

        Frame begin(RtHistoryState.Frame historyFrame) {
            boolean previousAvailable = historyFrame.reuseAllowed()
                    && latestSlot >= 0
                    && latestGeneration == historyFrame.generation();
            int writeSlot = latestSlot < 0 ? 0 : 1 - latestSlot;
            return new Frame(historyFrame.generation(), writeSlot,
                    previousAvailable ? latestSlot : -1, previousAvailable);
        }

        void commit(Frame frame) {
            latestSlot = frame.writeSlot();
            latestGeneration = frame.generation();
        }

        void reset() {
            latestSlot = -1;
            latestGeneration = Long.MIN_VALUE;
        }
    }

    static final class ShiftedSnapshotState {
        private long latestFrame = Long.MIN_VALUE;
        private long latestGeneration = Long.MIN_VALUE;

        boolean previousAvailable(Frame frame, long frameIndex) {
            return frame.previousAvailable()
                    && latestFrame != Long.MIN_VALUE
                    && latestFrame + 1L == frameIndex
                    && latestGeneration == frame.generation();
        }

        void commit(Frame frame, long frameIndex) {
            latestFrame = frameIndex;
            latestGeneration = frame.generation();
        }

        void reset() {
            latestFrame = Long.MIN_VALUE;
            latestGeneration = Long.MIN_VALUE;
        }
    }

    private final State state = new State();
    private final ShiftedSnapshotState shiftedSnapshotState = new ShiftedSnapshotState();
    private final RtBuffer[] slots = new RtBuffer[SLOT_COUNT];
    private RtBuffer spatialDiagnosticCounters;
    private RtBuffer spatialDiagnosticPairs;
    private RtBuffer shiftedSourceRoots;
    private RtBuffer shiftedMappedReservoirs;
    private RtPathTemporalPipeline temporalPipeline;
    private int spatialDiagnosticViewPending;
    private int width = -1;
    private int height = -1;

    void ensure(RtContext ctx, int requestedWidth, int requestedHeight,
                RtImage receiverMotion, RtImage validationMetadata, RtImage debugColor,
                RtImage receiverPositionMaterial, RtImage receiverNormalRoughness) {
        if (ready() && width == requestedWidth && height == requestedHeight) {
            return;
        }
        destroy();
        width = requestedWidth;
        height = requestedHeight;
        long bytesPerSlot = bytesPerSlot(width, height);
        int usage = VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        for (int slot = 0; slot < SLOT_COUNT; slot++) {
            slots[slot] = ctx.createBuffer(bytesPerSlot, usage, false,
                    "path reservoir history slot " + slot + " " + width + "x" + height);
        }
        spatialDiagnosticCounters = ctx.createBuffer(SPATIAL_DIAGNOSTIC_COUNTER_BYTES,
                VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                true, "path spatial diagnostic counters");
        spatialDiagnosticPairs = ctx.createBuffer(SPATIAL_DIAGNOSTIC_PAIR_BYTES,
                VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                true, "path spatial diagnostic target pairs");
        temporalPipeline = RtPathTemporalPipeline.create(ctx, receiverMotion.view,
                validationMetadata.view, debugColor.view,
                receiverPositionMaterial.view, receiverNormalRoughness.view,
                spatialDiagnosticCounters.handle, spatialDiagnosticPairs.handle);
        state.reset();
        shiftedSnapshotState.reset();
    }

    Frame beginFrame(RtHistoryState.Frame historyFrame) {
        if (!ready()) {
            throw new IllegalStateException("Path reservoirs used before allocation");
        }
        return state.begin(historyFrame);
    }

    void commit(Frame frame) {
        state.commit(frame);
    }

    void recordTemporalAdmission(VkCommandBuffer cmd, ByteBuffer pushConstants,
                                 int spatialDiagnosticView) {
        if (temporalPipeline == null) {
            throw new IllegalStateException("Path temporal pipeline used before allocation");
        }
        boolean counterDiagnostics = spatialDiagnosticView == SPATIAL_DEBUG_VIEW
                || spatialDiagnosticView == SPATIAL_POLICY_DEBUG_VIEW;
        boolean reconnectionDiagnostics = spatialDiagnosticView == RECONNECTION_DEBUG_VIEW;
        if (spatialDiagnosticView != 0 && !counterDiagnostics && !reconnectionDiagnostics) {
            throw new IllegalArgumentException(
                    "Unsupported path spatial diagnostic view: " + spatialDiagnosticView);
        }
        if (counterDiagnostics) {
            VK10.vkCmdFillBuffer(cmd, spatialDiagnosticCounters.handle, 0L,
                    spatialDiagnosticCounters.size, 0);
            try (var stack = org.lwjgl.system.MemoryStack.stackPush()) {
                VulkanCommandEncoder.memoryBarrier(cmd, stack);
            }
        }
        temporalPipeline.dispatch(cmd, width, height, pushConstants);
        spatialDiagnosticViewPending = counterDiagnostics ? spatialDiagnosticView : 0;
    }

    void beginShiftedRadianceDiagnostics(VkCommandBuffer cmd) {
        if (spatialDiagnosticCounters == null) {
            throw new IllegalStateException("Shifted-radiance diagnostics used before allocation");
        }
        VK10.vkCmdFillBuffer(cmd, spatialDiagnosticCounters.handle, 0L,
                spatialDiagnosticCounters.size, 0);
        try (var stack = org.lwjgl.system.MemoryStack.stackPush()) {
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
        }
        spatialDiagnosticViewPending = SHIFTED_RADIANCE_DEBUG_VIEW;
    }

    void ensureShiftedSourceRoots(RtContext ctx) {
        if (!ready()) {
            throw new IllegalStateException("Shifted source roots used before path allocation");
        }
        if (shiftedSourceRoots != null && shiftedMappedReservoirs != null) {
            return;
        }
        long rootBytes = Math.multiplyExact(Math.multiplyExact((long) width, height),
                PathSourceRootData.BYTE_SIZE);
        long mappedBytes = bytesPerSlot(width, height);
        shiftedSourceRoots = ctx.createBuffer(rootBytes, VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                false, "path shifted source roots " + width + "x" + height);
        shiftedMappedReservoirs = ctx.createBuffer(mappedBytes,
                VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                false, "path shifted mapped snapshot " + width + "x" + height);
        CausticaMod.LOGGER.info(
                "RT path shifted snapshot: render={}x{}, rootStride={} B, mappedStride={} B, "
                        + "rootBytes={}, mappedBytes={}, gpuMiB={}",
                width, height, PathSourceRootData.BYTE_SIZE, BYTES_PER_RESERVOIR,
                rootBytes, mappedBytes,
                String.format(Locale.ROOT, "%.2f", (rootBytes + mappedBytes) / (1024.0 * 1024.0)));
        shiftedSnapshotState.reset();
    }

    long shiftedDiagnosticCounterAddress() {
        return spatialDiagnosticCounters == null ? 0L : spatialDiagnosticCounters.deviceAddress;
    }

    long shiftedDiagnosticPairAddress() {
        return spatialDiagnosticPairs == null ? 0L : spatialDiagnosticPairs.deviceAddress;
    }

    long shiftedSourceRootAddress() {
        return shiftedSourceRoots == null ? 0L : shiftedSourceRoots.deviceAddress;
    }

    long shiftedMappedReservoirAddress() {
        return shiftedMappedReservoirs == null ? 0L : shiftedMappedReservoirs.deviceAddress;
    }

    boolean previousShiftedSnapshotAvailable(Frame frame, long frameIndex) {
        return shiftedSnapshotState.previousAvailable(frame, frameIndex);
    }

    void beginCurrentShiftedSnapshot(VkCommandBuffer cmd) {
        if (shiftedMappedReservoirs == null) {
            throw new IllegalStateException("Shifted mapped snapshot used before allocation");
        }
        VK10.vkCmdFillBuffer(cmd, shiftedMappedReservoirs.handle, 0L,
                shiftedMappedReservoirs.size, 0);
        try (var stack = org.lwjgl.system.MemoryStack.stackPush()) {
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
        }
    }

    void commitShiftedSnapshot(Frame frame, long frameIndex) {
        shiftedSnapshotState.commit(frame, frameIndex);
    }

    void pollSpatialDiagnosticCounters(RtContext ctx, long frameIndex) {
        if (spatialDiagnosticViewPending == 0
                || frameIndex == 0L || frameIndex % 60L != 0L) {
            return;
        }
        ctx.waitIdle();
        spatialDiagnosticCounters.invalidate();
        spatialDiagnosticPairs.invalidate();
        if (spatialDiagnosticViewPending == SHIFTED_RADIANCE_DEBUG_VIEW) {
            IntBuffer counters = MemoryUtil.memIntBuffer(spatialDiagnosticCounters.mapped,
                    SHIFTED_DIAGNOSTIC_COUNTER_COUNT);
            long[] values = new long[SHIFTED_DIAGNOSTIC_STATE_COUNT];
            long total = 0L;
            for (int index = 0; index < values.length; index++) {
                values[index] = Integer.toUnsignedLong(counters.get(index));
                total += values[index];
            }
            long sampleAttempts = Integer.toUnsignedLong(
                    counters.get(SHIFTED_DIAGNOSTIC_PAIR_CURSOR_INDEX));
            long selectionEligible = Integer.toUnsignedLong(
                    counters.get(SHIFTED_DIAGNOSTIC_SELECTION_ELIGIBLE_INDEX));
            long sourceSelected = Integer.toUnsignedLong(
                    counters.get(SHIFTED_DIAGNOSTIC_SOURCE_SELECTED_INDEX));
            long scratchWritten = Integer.toUnsignedLong(
                    counters.get(SHIFTED_DIAGNOSTIC_SCRATCH_WRITTEN_INDEX));
            long scratchSelected = Integer.toUnsignedLong(
                    counters.get(SHIFTED_DIAGNOSTIC_SCRATCH_SELECTED_INDEX));
            long scratchInvalid = Integer.toUnsignedLong(
                    counters.get(SHIFTED_DIAGNOSTIC_SCRATCH_INVALID_INDEX));
            long mappingEligible = Integer.toUnsignedLong(
                    counters.get(MAPPING_REPLAY_ELIGIBLE_INDEX));
            long mappingAccepted = Integer.toUnsignedLong(
                    counters.get(MAPPING_REPLAY_ACCEPTED_INDEX));
            long mappingAbiReject = Integer.toUnsignedLong(
                    counters.get(MAPPING_REPLAY_ABI_REJECT_INDEX));
            long mappingSourceReject = Integer.toUnsignedLong(
                    counters.get(MAPPING_REPLAY_SOURCE_REJECT_INDEX));
            long mappingReceiverReject = Integer.toUnsignedLong(
                    counters.get(MAPPING_REPLAY_RECEIVER_REJECT_INDEX));
            long mappingGeometryReject = Integer.toUnsignedLong(
                    counters.get(MAPPING_REPLAY_GEOMETRY_REJECT_INDEX));
            long mappingPdfReject = Integer.toUnsignedLong(
                    counters.get(MAPPING_REPLAY_PDF_REJECT_INDEX));
            long mappingVisibilityReject = Integer.toUnsignedLong(
                    counters.get(MAPPING_REPLAY_VISIBILITY_REJECT_INDEX));
            long mappingRadianceReject = Integer.toUnsignedLong(
                    counters.get(MAPPING_REPLAY_RADIANCE_REJECT_INDEX));
            long sourceRootWritten = Integer.toUnsignedLong(
                    counters.get(SHIFTED_SOURCE_ROOT_WRITTEN_INDEX));
            long sourceRootInvalid = Integer.toUnsignedLong(
                    counters.get(SHIFTED_SOURCE_ROOT_INVALID_INDEX));
            long mappingSourceRootReject = Integer.toUnsignedLong(
                    counters.get(MAPPING_REPLAY_SOURCE_ROOT_REJECT_INDEX));
            long crossFrameAttempted = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_ATTEMPTED_INDEX));
            long crossFrameReceiverReprojectionReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_RECEIVER_REPROJECTION_REJECT_INDEX));
            long crossFrameMappedEmpty = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_MAPPED_EMPTY_INDEX));
            long crossFrameEligible = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_ELIGIBLE_INDEX));
            long crossFrameAccepted = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_ACCEPTED_INDEX));
            long crossFrameAbiReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_ABI_REJECT_INDEX));
            long crossFrameSourceRootReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_SOURCE_ROOT_REJECT_INDEX));
            long crossFrameSourceReprojectionReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_SOURCE_REPROJECTION_REJECT_INDEX));
            long crossFrameSourceReplayReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_SOURCE_REPLAY_REJECT_INDEX));
            long crossFrameReceiverSurfaceReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_RECEIVER_SURFACE_REJECT_INDEX));
            long crossFrameGeometryReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_GEOMETRY_REJECT_INDEX));
            long crossFramePdfReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_PDF_REJECT_INDEX));
            long crossFrameVisibilityReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_VISIBILITY_REJECT_INDEX));
            long crossFrameRadianceReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_RADIANCE_REJECT_INDEX));
            long crossFrameReceiverSampleReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_RECEIVER_SAMPLE_REJECT_INDEX));
            long crossFrameReceiverEdgeReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_RECEIVER_EDGE_REJECT_INDEX));
            long crossFrameReceiverTopologyReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_RECEIVER_TOPOLOGY_REJECT_INDEX));
            long crossFrameReceiverDepthReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_RECEIVER_DEPTH_REJECT_INDEX));
            long crossFrameReceiverTransportReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_RECEIVER_TRANSPORT_REJECT_INDEX));
            long crossFrameReceiverFootprintReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_RECEIVER_FOOTPRINT_REJECT_INDEX));
            long crossFrameSampleRescueEligible = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_SAMPLE_RESCUE_ELIGIBLE_INDEX));
            long crossFrameSampleRescued = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_SAMPLE_RESCUED_INDEX));
            long crossFrameSampleRescueEdgeReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_SAMPLE_RESCUE_EDGE_REJECT_INDEX));
            long crossFrameSampleRescueTopologyReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_SAMPLE_RESCUE_TOPOLOGY_REJECT_INDEX));
            long crossFrameSampleRescueDepthReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_SAMPLE_RESCUE_DEPTH_REJECT_INDEX));
            long crossFrameSampleRescueTransportReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_SAMPLE_RESCUE_TRANSPORT_REJECT_INDEX));
            long crossFrameSampleRescueFootprintReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_SAMPLE_RESCUE_FOOTPRINT_REJECT_INDEX));
            long crossFrameEdgeBreakdownEligible = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_EDGE_BREAKDOWN_ELIGIBLE_INDEX));
            long crossFrameEdgeMissingValid = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_EDGE_MISSING_VALID_INDEX));
            long crossFrameEdgeDepthReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_EDGE_DEPTH_REJECT_INDEX));
            long crossFrameEdgeEventReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_EDGE_EVENT_REJECT_INDEX));
            long crossFrameEdgeMappingReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_EDGE_MAPPING_REJECT_INDEX));
            long crossFrameEdgePdfReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_EDGE_PDF_REJECT_INDEX));
            long crossFrameEdgeFiniteReject = Integer.toUnsignedLong(
                    counters.get(CROSS_FRAME_EDGE_FINITE_REJECT_INDEX));
            long crossFrameReceiverReject = crossFrameReceiverSurfaceReject
                    + crossFrameReceiverSampleReject + crossFrameReceiverEdgeReject
                    + crossFrameReceiverTopologyReject + crossFrameReceiverDepthReject
                    + crossFrameReceiverTransportReject + crossFrameReceiverFootprintReject;
            int capturedSamples = (int) Math.min(sampleAttempts, SPATIAL_DIAGNOSTIC_PAIR_CAPACITY);
            FloatBuffer samples = MemoryUtil.memFloatBuffer(
                    spatialDiagnosticPairs.mapped, SHIFTED_DIAGNOSTIC_PAIR_FLOAT_COUNT);
            double shiftedSum = 0.0;
            double shiftedLogSum = 0.0;
            double sourceSum = 0.0;
            double receiverPdfSum = 0.0;
            double pssJacobianSum = 0.0;
            double receiverPdfMin = Double.POSITIVE_INFINITY;
            double receiverPdfMax = 0.0;
            double pssJacobianMin = Double.POSITIVE_INFINITY;
            double pssJacobianMax = 0.0;
            double[] receiverPdfSamples = new double[capturedSamples];
            double[] pssJacobianSamples = new double[capturedSamples];
            double[] sourceFinalWeightSamples = new double[capturedSamples];
            double[] mergeWeightSamples = new double[capturedSamples];
            double[] currentWeightSumSamples = new double[capturedSamples];
            double[] selectionProbabilitySamples = new double[capturedSamples];
            double shiftedMin = Double.POSITIVE_INFINITY;
            double shiftedMax = 0.0;
            int finiteSamples = 0;
            for (int sample = 0; sample < capturedSamples; sample++) {
                float shifted = samples.get(sample * 2);
                float source = samples.get(sample * 2 + 1);
                int densityOffset = SHIFTED_DIAGNOSTIC_DENSITY_PAIR_OFFSET * 2 + sample * 2;
                float receiverPdf = samples.get(densityOffset);
                float pssJacobian = samples.get(densityOffset + 1);
                int mergeOffset = SHIFTED_DIAGNOSTIC_MERGE_PAIR_OFFSET * 2 + sample * 2;
                float sourceFinalWeight = samples.get(mergeOffset);
                float mergeWeight = samples.get(mergeOffset + 1);
                int selectionOffset = SHIFTED_DIAGNOSTIC_SELECTION_PAIR_OFFSET * 2 + sample * 2;
                float currentWeightSum = samples.get(selectionOffset);
                float selectionProbability = samples.get(selectionOffset + 1);
                if (!Float.isFinite(shifted) || !Float.isFinite(source)
                        || shifted < 0.0f || source < 0.0f
                        || !Float.isFinite(receiverPdf) || receiverPdf <= 0.0f
                        || !Float.isFinite(pssJacobian) || pssJacobian <= 0.0f
                        || !Float.isFinite(sourceFinalWeight) || sourceFinalWeight <= 0.0f
                        || !Float.isFinite(mergeWeight) || mergeWeight < 0.0f
                        || !Float.isFinite(currentWeightSum) || currentWeightSum < 0.0f
                        || !Float.isFinite(selectionProbability) || selectionProbability < 0.0f
                        || selectionProbability > 1.0f) {
                    continue;
                }
                receiverPdfSamples[finiteSamples] = receiverPdf;
                pssJacobianSamples[finiteSamples] = pssJacobian;
                sourceFinalWeightSamples[finiteSamples] = sourceFinalWeight;
                mergeWeightSamples[finiteSamples] = mergeWeight;
                currentWeightSumSamples[finiteSamples] = currentWeightSum;
                selectionProbabilitySamples[finiteSamples] = selectionProbability;
                finiteSamples++;
                shiftedSum += shifted;
                shiftedLogSum += Math.log1p(shifted);
                sourceSum += source;
                receiverPdfSum += receiverPdf;
                pssJacobianSum += pssJacobian;
                receiverPdfMin = Math.min(receiverPdfMin, receiverPdf);
                receiverPdfMax = Math.max(receiverPdfMax, receiverPdf);
                pssJacobianMin = Math.min(pssJacobianMin, pssJacobian);
                pssJacobianMax = Math.max(pssJacobianMax, pssJacobian);
                shiftedMin = Math.min(shiftedMin, shifted);
                shiftedMax = Math.max(shiftedMax, shifted);
            }
            Arrays.sort(receiverPdfSamples, 0, finiteSamples);
            Arrays.sort(pssJacobianSamples, 0, finiteSamples);
            Arrays.sort(sourceFinalWeightSamples, 0, finiteSamples);
            Arrays.sort(mergeWeightSamples, 0, finiteSamples);
            Arrays.sort(currentWeightSumSamples, 0, finiteSamples);
            Arrays.sort(selectionProbabilitySamples, 0, finiteSamples);
            double invSamples = finiteSamples == 0 ? 0.0 : 1.0 / finiteSamples;
            CausticaMod.LOGGER.info(
                    "RT path shifted diagnostics: total={}, empty={} ({}%), noPair={} ({}%), "
                            + "edgeMissing={} ({}%), strictReject={} ({}%), unsupported={} ({}%), "
                            + "geometry={} ({}%), pdf={} ({}%), mass={} ({}%), jacobian={} ({}%), "
                            + "occluded={} ({}%), spectral={} ({}%), zero={} ({}%), positive={} ({}%), "
                            + "arithmetic={} ({}%), "
                            + "overflow={} ({}%), samples={}/{}, finiteSamples={}, "
                            + "shiftedTarget[min={},max={},mean={},logMean={}], sourceTargetMean={}, "
                            + "receiverPdf[min={},p50={},p95={},p99={},max={},mean={}], "
                            + "pssJacobian[min={},p50={},p95={},p99={},max={},mean={}], "
                            + "sourceFinalWeight[p50={},p95={},p99={}], "
                            + "mergeWeight[p50={},p95={},p99={}], "
                            + "currentWeightSum[p50={},p95={},p99={}], "
                            + "selectionProbability[p50={},p95={},p99={}], "
                             + "sourceSelected={}/{} ({}%), "
                             + "scratch[written={},selected={},invalid={}], "
                             + "sourceRoot[written={},invalid={}], "
                             + "mappingReplay[eligible={},accepted={},abi={},source={},receiver={},"
                             + "geometry={},pdf={},visibility={},radiance={},root={}], "
                             + "crossFrame[attempted={},receiverReprojection={},mappedEmpty={},"
                             + "eligible={},accepted={},abi={},root={},sourceReprojection={},"
                             + "sourceReplay={},receiverTotal={},receiverSurface={},receiverSample={},"
                             + "receiverEdge={},receiverTopology={},receiverDepth={},"
                             + "receiverTransport={},receiverFootprint={},geometry={},pdf={},"
                             + "visibility={},radiance={}], "
                             + "sampleRescue[eligible={},rescued={},edge={},topology={},depth={},"
                             + "transport={},footprint={}], "
                             + "edgeBreakdown[eligible={},missingValid={},depth={},event={},"
                             + "mapping={},pdf={},finite={}]",
                    total,
                    values[0], percent(values[0], total),
                    values[1], percent(values[1], total),
                    values[2], percent(values[2], total),
                    values[3], percent(values[3], total),
                    values[4], percent(values[4], total),
                    values[5], percent(values[5], total),
                    values[6], percent(values[6], total),
                    values[7], percent(values[7], total),
                    values[8], percent(values[8], total),
                    values[9], percent(values[9], total),
                    values[10], percent(values[10], total),
                    values[11], percent(values[11], total),
                    values[12], percent(values[12], total),
                    values[13], percent(values[13], total),
                    values[14], percent(values[14], total),
                    capturedSamples, sampleAttempts, finiteSamples,
                    finiteSamples == 0 ? "n/a" : metric(shiftedMin),
                    finiteSamples == 0 ? "n/a" : metric(shiftedMax),
                    finiteSamples == 0 ? "n/a" : metric(shiftedSum * invSamples),
                    finiteSamples == 0 ? "n/a" : metric(shiftedLogSum * invSamples),
                    finiteSamples == 0 ? "n/a" : metric(sourceSum * invSamples),
                    finiteSamples == 0 ? "n/a" : metric(receiverPdfMin),
                    finiteSamples == 0 ? "n/a" : metric(sortedPercentile(
                            receiverPdfSamples, finiteSamples, 0.50)),
                    finiteSamples == 0 ? "n/a" : metric(sortedPercentile(
                            receiverPdfSamples, finiteSamples, 0.95)),
                    finiteSamples == 0 ? "n/a" : metric(sortedPercentile(
                            receiverPdfSamples, finiteSamples, 0.99)),
                    finiteSamples == 0 ? "n/a" : metric(receiverPdfMax),
                    finiteSamples == 0 ? "n/a" : metric(receiverPdfSum * invSamples),
                    finiteSamples == 0 ? "n/a" : metric(pssJacobianMin),
                    finiteSamples == 0 ? "n/a" : metric(sortedPercentile(
                            pssJacobianSamples, finiteSamples, 0.50)),
                    finiteSamples == 0 ? "n/a" : metric(sortedPercentile(
                            pssJacobianSamples, finiteSamples, 0.95)),
                    finiteSamples == 0 ? "n/a" : metric(sortedPercentile(
                            pssJacobianSamples, finiteSamples, 0.99)),
                    finiteSamples == 0 ? "n/a" : metric(pssJacobianMax),
                    finiteSamples == 0 ? "n/a" : metric(pssJacobianSum * invSamples),
                    finiteSamples == 0 ? "n/a" : metric(sortedPercentile(
                            sourceFinalWeightSamples, finiteSamples, 0.50)),
                    finiteSamples == 0 ? "n/a" : metric(sortedPercentile(
                            sourceFinalWeightSamples, finiteSamples, 0.95)),
                    finiteSamples == 0 ? "n/a" : metric(sortedPercentile(
                            sourceFinalWeightSamples, finiteSamples, 0.99)),
                    finiteSamples == 0 ? "n/a" : metric(sortedPercentile(
                            mergeWeightSamples, finiteSamples, 0.50)),
                    finiteSamples == 0 ? "n/a" : metric(sortedPercentile(
                            mergeWeightSamples, finiteSamples, 0.95)),
                    finiteSamples == 0 ? "n/a" : metric(sortedPercentile(
                            mergeWeightSamples, finiteSamples, 0.99)),
                    finiteSamples == 0 ? "n/a" : metric(sortedPercentile(
                            currentWeightSumSamples, finiteSamples, 0.50)),
                    finiteSamples == 0 ? "n/a" : metric(sortedPercentile(
                            currentWeightSumSamples, finiteSamples, 0.95)),
                    finiteSamples == 0 ? "n/a" : metric(sortedPercentile(
                            currentWeightSumSamples, finiteSamples, 0.99)),
                    finiteSamples == 0 ? "n/a" : metric(sortedPercentile(
                            selectionProbabilitySamples, finiteSamples, 0.50)),
                    finiteSamples == 0 ? "n/a" : metric(sortedPercentile(
                            selectionProbabilitySamples, finiteSamples, 0.95)),
                    finiteSamples == 0 ? "n/a" : metric(sortedPercentile(
                            selectionProbabilitySamples, finiteSamples, 0.99)),
                    sourceSelected, selectionEligible,
                    percent(sourceSelected, selectionEligible),
                    scratchWritten, scratchSelected, scratchInvalid,
                    sourceRootWritten, sourceRootInvalid,
                    mappingEligible, mappingAccepted, mappingAbiReject,
                    mappingSourceReject, mappingReceiverReject,
                     mappingGeometryReject, mappingPdfReject,
                     mappingVisibilityReject, mappingRadianceReject,
                     mappingSourceRootReject,
                     crossFrameAttempted, crossFrameReceiverReprojectionReject,
                     crossFrameMappedEmpty, crossFrameEligible, crossFrameAccepted,
                     crossFrameAbiReject, crossFrameSourceRootReject,
                     crossFrameSourceReprojectionReject, crossFrameSourceReplayReject,
                     crossFrameReceiverReject, crossFrameReceiverSurfaceReject,
                     crossFrameReceiverSampleReject, crossFrameReceiverEdgeReject,
                     crossFrameReceiverTopologyReject, crossFrameReceiverDepthReject,
                     crossFrameReceiverTransportReject, crossFrameReceiverFootprintReject,
                     crossFrameGeometryReject,
                     crossFramePdfReject, crossFrameVisibilityReject,
                     crossFrameRadianceReject,
                     crossFrameSampleRescueEligible, crossFrameSampleRescued,
                     crossFrameSampleRescueEdgeReject,
                     crossFrameSampleRescueTopologyReject,
                     crossFrameSampleRescueDepthReject,
                     crossFrameSampleRescueTransportReject,
                     crossFrameSampleRescueFootprintReject,
                     crossFrameEdgeBreakdownEligible, crossFrameEdgeMissingValid,
                     crossFrameEdgeDepthReject, crossFrameEdgeEventReject,
                     crossFrameEdgeMappingReject, crossFrameEdgePdfReject,
                     crossFrameEdgeFiniteReject);
            spatialDiagnosticViewPending = 0;
            return;
        }
        IntBuffer counters = MemoryUtil.memIntBuffer(spatialDiagnosticCounters.mapped,
                SPATIAL_DIAGNOSTIC_COUNTER_COUNT);
        long[] values = new long[SPATIAL_DIAGNOSTIC_CATEGORY_COUNT];
        long total = 0L;
        for (int index = 0; index < values.length; index++) {
            values[index] = Integer.toUnsignedLong(counters.get(index));
            total += values[index];
        }
        long pairAttempts = Integer.toUnsignedLong(
                counters.get(SPATIAL_DIAGNOSTIC_STRICT_PAIR_CURSOR_INDEX));
        int capturedPairs = (int) Math.min(pairAttempts, SPATIAL_DIAGNOSTIC_PAIR_CAPACITY);
        FloatBuffer pairValues = MemoryUtil.memFloatBuffer(
                spatialDiagnosticPairs.mapped, SPATIAL_DIAGNOSTIC_PAIR_CAPACITY * 4);
        var rawMoments = new RtPathSpatialReuseReference.PairMoments();
        var logMoments = new RtPathSpatialReuseReference.PairMoments();
        addPairs(pairValues, 0, capturedPairs, rawMoments, logMoments);
        long limitedAdmitted = Integer.toUnsignedLong(
                counters.get(SPATIAL_DIAGNOSTIC_LIMITED_ADMITTED_INDEX));
        long topologyRescued = Integer.toUnsignedLong(
                counters.get(SPATIAL_DIAGNOSTIC_TOPOLOGY_RESCUED_INDEX));
        long rescuedPairAttempts = Integer.toUnsignedLong(
                counters.get(SPATIAL_DIAGNOSTIC_RESCUED_PAIR_CURSOR_INDEX));
        int capturedRescuedPairs = (int) Math.min(
                rescuedPairAttempts, SPATIAL_DIAGNOSTIC_PAIR_CAPACITY);
        var rescuedRawMoments = new RtPathSpatialReuseReference.PairMoments();
        var rescuedLogMoments = new RtPathSpatialReuseReference.PairMoments();
        addPairs(pairValues, SPATIAL_DIAGNOSTIC_PAIR_CAPACITY, capturedRescuedPairs,
                rescuedRawMoments, rescuedLogMoments);
        if (total > 0L) {
            if (spatialDiagnosticViewPending == SPATIAL_POLICY_DEBUG_VIEW) {
                CausticaMod.LOGGER.info(
                        "RT path spatial policy A/B: total={}, strictAdmitted={} ({}%), "
                                + "limitedAdmitted={} ({}%), topologyRescued={} ({}%), "
                                + "strictPairs={}/{}, rescuedPairs={}/{}, "
                                + "strictCorr={}, rescuedCorr={}, strictLogCorr={}, rescuedLogCorr={}",
                        total,
                        values[1], percent(values[1], total),
                        limitedAdmitted, percent(limitedAdmitted, total),
                        topologyRescued, percent(topologyRescued, total),
                        capturedPairs, pairAttempts,
                        capturedRescuedPairs, rescuedPairAttempts,
                        metric(rawMoments.correlation()),
                        metric(rescuedRawMoments.correlation()),
                        metric(logMoments.correlation()),
                        metric(rescuedLogMoments.correlation()));
            } else {
                CausticaMod.LOGGER.info(
                        "RT path spatial categories: total={}, empty={} ({}%), admitted={} ({}%), "
                                + "footprintReject={} ({}%), depthReject={} ({}%), "
                                + "topologyReject={} ({}%), transportReject={} ({}%), "
                                + "noReconnection={} ({}%), neighborEmpty={} ({}%), "
                                + "surfaceReject={} ({}%), pairs={}/{}, validPairs={}, "
                                + "targetCorr={}, logTargetCorr={}",
                        total,
                        values[0], percent(values[0], total),
                        values[1], percent(values[1], total),
                        values[2], percent(values[2], total),
                        values[3], percent(values[3], total),
                        values[4], percent(values[4], total),
                        values[5], percent(values[5], total),
                        values[6], percent(values[6], total),
                        values[7], percent(values[7], total),
                        values[8], percent(values[8], total),
                        capturedPairs, pairAttempts, rawMoments.count(),
                        metric(rawMoments.correlation()), metric(logMoments.correlation()));
            }
        }
        spatialDiagnosticViewPending = 0;
    }

    void reset() {
        state.reset();
        shiftedSnapshotState.reset();
    }

    RtBuffer finalBuffer(Frame frame) {
        return slot(frame.finalSlot());
    }

    RtBuffer previousBuffer(Frame frame) {
        return frame.previousAvailable() ? slot(frame.previousSlot()) : null;
    }

    RtBuffer scratchBuffer(Frame frame) {
        return slot(frame.scratchSlot());
    }

    long allocatedBytes() {
        return ready() ? Math.multiplyExact(bytesPerSlot(width, height), SLOT_COUNT) : 0L;
    }

    static long bytesPerSlot(int width, int height) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("Path reservoir extent must be positive");
        }
        return Math.multiplyExact(Math.multiplyExact((long) width, height),
                BYTES_PER_RESERVOIR);
    }

    boolean ready() {
        return slots[0] != null && slots[1] != null
                && spatialDiagnosticCounters != null && spatialDiagnosticPairs != null
                && temporalPipeline != null;
    }

    void destroy() {
        if (temporalPipeline != null) {
            temporalPipeline.destroy();
            temporalPipeline = null;
        }
        if (spatialDiagnosticCounters != null) {
            spatialDiagnosticCounters.destroy();
            spatialDiagnosticCounters = null;
        }
        if (spatialDiagnosticPairs != null) {
            spatialDiagnosticPairs.destroy();
            spatialDiagnosticPairs = null;
        }
        if (shiftedSourceRoots != null) {
            shiftedSourceRoots.destroy();
            shiftedSourceRoots = null;
        }
        if (shiftedMappedReservoirs != null) {
            shiftedMappedReservoirs.destroy();
            shiftedMappedReservoirs = null;
        }
        for (int slot = 0; slot < SLOT_COUNT; slot++) {
            if (slots[slot] != null) {
                slots[slot].destroy();
                slots[slot] = null;
            }
        }
        width = -1;
        height = -1;
        spatialDiagnosticViewPending = 0;
        state.reset();
        shiftedSnapshotState.reset();
    }

    private static String percent(long value, long total) {
        return String.format(Locale.ROOT, "%.2f", value * 100.0 / total);
    }

    private static String metric(double value) {
        return Double.isFinite(value)
                ? String.format(Locale.ROOT, "%.4f", value)
                : "n/a";
    }

    static double sortedPercentile(double[] sortedValues, int count, double quantile) {
        if (sortedValues == null || count <= 0 || count > sortedValues.length
                || !Double.isFinite(quantile) || quantile < 0.0 || quantile > 1.0) {
            throw new IllegalArgumentException("invalid sorted percentile input");
        }
        double position = quantile * (count - 1);
        int lower = (int) Math.floor(position);
        int upper = (int) Math.ceil(position);
        double fraction = position - lower;
        return sortedValues[lower] * (1.0 - fraction) + sortedValues[upper] * fraction;
    }

    private static void addPairs(
            FloatBuffer pairValues, int pairOffset, int pairCount,
            RtPathSpatialReuseReference.PairMoments rawMoments,
            RtPathSpatialReuseReference.PairMoments logMoments) {
        for (int pair = 0; pair < pairCount; pair++) {
            int floatIndex = (pairOffset + pair) * 2;
            float receiverTarget = pairValues.get(floatIndex);
            float sourceTarget = pairValues.get(floatIndex + 1);
            if (!Float.isFinite(receiverTarget) || !Float.isFinite(sourceTarget)
                    || receiverTarget < 0.0f || sourceTarget < 0.0f) {
                continue;
            }
            rawMoments.add(receiverTarget, sourceTarget);
            logMoments.add(Math.log1p(receiverTarget), Math.log1p(sourceTarget));
        }
    }

    private RtBuffer slot(int slot) {
        if (!ready()) {
            throw new IllegalStateException("Path reservoirs used before allocation");
        }
        if (slot < 0 || slot >= SLOT_COUNT) {
            throw new IllegalArgumentException("Path reservoir slot out of range: " + slot);
        }
        return slots[slot];
    }
}
