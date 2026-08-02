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
    static final int GUIDE_SCRATCH_VALIDATE_PASS_FLAG = 1 << 7;
    static final int GUIDE_PREVIOUS_REPLAY_PASS_FLAG = 1 << 8;
    static final int GUIDE_PREVIOUS_AVAILABLE_FLAG = 1 << 9;
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
    static final int RECEIVER_GUIDE_ATTEMPTED_INDEX = 67;
    static final int RECEIVER_GUIDE_INVALID_INDEX = 68;
    static final int RECEIVER_GUIDE_NO_STORED_EDGE_INDEX = 69;
    static final int RECEIVER_GUIDE_STORED_ELIGIBLE_INDEX = 70;
    static final int RECEIVER_GUIDE_ACCEPTED_INDEX = 71;
    static final int RECEIVER_GUIDE_MASS_MISMATCH_INDEX = 72;
    static final int RECEIVER_GUIDE_THROUGHPUT_MISMATCH_INDEX = 73;
    static final int RECEIVER_GUIDE_PDF_MISMATCH_INDEX = 74;
    static final int GUIDE_REMAP_ELIGIBLE_INDEX = 75;
    static final int GUIDE_REMAP_READY_INDEX = 76;
    static final int GUIDE_REMAP_GEOMETRY_REJECT_INDEX = 77;
    static final int GUIDE_REMAP_PDF_REJECT_INDEX = 78;
    static final int GUIDE_REMAP_THROUGHPUT_REJECT_INDEX = 79;
    static final int GUIDE_VISIBILITY_ELIGIBLE_INDEX = 80;
    static final int GUIDE_VISIBILITY_CLEAR_INDEX = 81;
    static final int GUIDE_VISIBILITY_TINTED_INDEX = 82;
    static final int GUIDE_VISIBILITY_OCCLUDED_INDEX = 83;
    static final int GUIDE_VISIBILITY_INVALID_INDEX = 84;
    static final int GUIDE_TARGET_ELIGIBLE_INDEX = 85;
    static final int GUIDE_TARGET_POSITIVE_INDEX = 86;
    static final int GUIDE_TARGET_ZERO_INDEX = 87;
    static final int GUIDE_TARGET_INVALID_INDEX = 88;
    static final int GUIDE_WEIGHT_ELIGIBLE_INDEX = 89;
    static final int GUIDE_WEIGHT_POSITIVE_INDEX = 90;
    static final int GUIDE_WEIGHT_ZERO_INDEX = 91;
    static final int GUIDE_WEIGHT_INVALID_INDEX = 92;
    static final int GUIDE_SELECTION_ELIGIBLE_INDEX = 93;
    static final int GUIDE_SELECTION_POSITIVE_INDEX = 94;
    static final int GUIDE_SELECTION_ZERO_INDEX = 95;
    static final int GUIDE_SELECTION_CURRENT_REJECT_INDEX = 96;
    static final int GUIDE_SELECTION_PROBABILITY_HIGH_REJECT_INDEX = 97;
    static final int GUIDE_SELECTION_ARITHMETIC_REJECT_INDEX = 98;
    static final int GUIDE_STABLE_SELECTION_ELIGIBLE_INDEX = 99;
    static final int GUIDE_STABLE_SELECTION_POSITIVE_INDEX = 100;
    static final int GUIDE_STABLE_SELECTION_ZERO_INDEX = 101;
    static final int GUIDE_STABLE_SELECTION_INVALID_INDEX = 102;
    static final int GUIDE_STABLE_BERNOULLI_ELIGIBLE_INDEX = 103;
    static final int GUIDE_STABLE_BERNOULLI_SELECTED_INDEX = 104;
    static final int GUIDE_STABLE_BERNOULLI_RETAINED_INDEX = 105;
    static final int GUIDE_STABLE_BERNOULLI_INVALID_INDEX = 106;
    static final int GUIDE_POST_SELECTION_ELIGIBLE_INDEX = 107;
    static final int GUIDE_POST_SELECTION_SELECTED_READY_INDEX = 108;
    static final int GUIDE_POST_SELECTION_RETAINED_READY_INDEX = 109;
    static final int GUIDE_POST_SELECTION_EMPTY_INDEX = 110;
    static final int GUIDE_POST_SELECTION_SAMPLE_REJECT_INDEX = 111;
    static final int GUIDE_POST_SELECTION_ARITHMETIC_REJECT_INDEX = 112;
    static final int GUIDE_SAMPLE_COPY_ELIGIBLE_INDEX = 113;
    static final int GUIDE_SAMPLE_COPY_SELECTED_INDEX = 114;
    static final int GUIDE_SAMPLE_COPY_RETAINED_INDEX = 115;
    static final int GUIDE_SAMPLE_COPY_EMPTY_INDEX = 116;
    static final int GUIDE_SAMPLE_COPY_METADATA_REJECT_INDEX = 117;
    static final int GUIDE_SAMPLE_COPY_ARITHMETIC_REJECT_INDEX = 118;
    static final int GUIDE_SCRATCH_STORAGE_ELIGIBLE_INDEX = 119;
    static final int GUIDE_SCRATCH_STORAGE_SELECTED_INDEX = 120;
    static final int GUIDE_SCRATCH_STORAGE_RETAINED_INDEX = 121;
    static final int GUIDE_SCRATCH_STORAGE_EMPTY_INDEX = 122;
    static final int GUIDE_SCRATCH_STORAGE_METADATA_REJECT_INDEX = 123;
    static final int GUIDE_SCRATCH_STORAGE_ARITHMETIC_REJECT_INDEX = 124;
    static final int GUIDE_SCRATCH_STORAGE_INVALID_INDEX = 125;
    static final int GUIDE_ROOT_STORAGE_ELIGIBLE_INDEX = 126;
    static final int GUIDE_ROOT_STORAGE_SELECTED_READY_INDEX = 127;
    static final int GUIDE_ROOT_STORAGE_RETAINED_READY_INDEX = 128;
    static final int GUIDE_ROOT_STORAGE_MISSING_INDEX = 129;
    static final int GUIDE_ROOT_STORAGE_METADATA_REJECT_INDEX = 130;
    static final int GUIDE_PREVIOUS_REPLAY_ATTEMPTED_INDEX = 131;
    static final int GUIDE_PREVIOUS_REPLAY_LIFECYCLE_REJECT_INDEX = 132;
    static final int GUIDE_PREVIOUS_REPLAY_RECEIVER_REPROJECTION_REJECT_INDEX = 133;
    static final int GUIDE_PREVIOUS_REPLAY_EMPTY_INDEX = 134;
    static final int GUIDE_PREVIOUS_REPLAY_METADATA_REJECT_INDEX = 135;
    static final int GUIDE_PREVIOUS_REPLAY_RECEIVER_SURFACE_REJECT_INDEX = 136;
    static final int GUIDE_PREVIOUS_REPLAY_SOURCE_REPROJECTION_REJECT_INDEX = 137;
    static final int GUIDE_PREVIOUS_REPLAY_SOURCE_SURFACE_REJECT_INDEX = 138;
    static final int GUIDE_PREVIOUS_REPLAY_SOURCE_REPLAY_REJECT_INDEX = 139;
    static final int GUIDE_PREVIOUS_REPLAY_SELECTED_ACCEPTED_INDEX = 140;
    static final int GUIDE_PREVIOUS_REPLAY_RETAINED_ACCEPTED_INDEX = 141;
    static final int GUIDE_PREVIOUS_REMAP_ELIGIBLE_INDEX = 142;
    static final int GUIDE_PREVIOUS_REMAP_RECEIVER_GUIDE_REJECT_INDEX = 143;
    static final int GUIDE_PREVIOUS_REMAP_READY_INDEX = 144;
    static final int GUIDE_PREVIOUS_REMAP_GEOMETRY_REJECT_INDEX = 145;
    static final int GUIDE_PREVIOUS_REMAP_PDF_REJECT_INDEX = 146;
    static final int GUIDE_PREVIOUS_REMAP_THROUGHPUT_REJECT_INDEX = 147;
    static final int GUIDE_PREVIOUS_VISIBILITY_ELIGIBLE_INDEX = 148;
    static final int GUIDE_PREVIOUS_VISIBILITY_CLEAR_INDEX = 149;
    static final int GUIDE_PREVIOUS_VISIBILITY_TINTED_INDEX = 150;
    static final int GUIDE_PREVIOUS_VISIBILITY_OCCLUDED_INDEX = 151;
    static final int GUIDE_PREVIOUS_VISIBILITY_INVALID_INDEX = 152;
    static final int GUIDE_PREVIOUS_TARGET_ELIGIBLE_INDEX = 153;
    static final int GUIDE_PREVIOUS_TARGET_POSITIVE_INDEX = 154;
    static final int GUIDE_PREVIOUS_TARGET_ZERO_INDEX = 155;
    static final int GUIDE_PREVIOUS_TARGET_INVALID_INDEX = 156;
    static final int GUIDE_PREVIOUS_STORED_METADATA_ELIGIBLE_INDEX = 157;
    static final int GUIDE_PREVIOUS_STORED_TARGET_MATCH_INDEX = 158;
    static final int GUIDE_PREVIOUS_STORED_TARGET_MISMATCH_INDEX = 159;
    static final int GUIDE_PREVIOUS_STORED_TARGET_INVALID_INDEX = 160;
    static final int GUIDE_PREVIOUS_STORED_THROUGHPUT_MATCH_INDEX = 161;
    static final int GUIDE_PREVIOUS_STORED_THROUGHPUT_MISMATCH_INDEX = 162;
    static final int GUIDE_PREVIOUS_STORED_THROUGHPUT_INVALID_INDEX = 163;
    static final int GUIDE_PREVIOUS_WEIGHT_ELIGIBLE_INDEX = 164;
    static final int GUIDE_PREVIOUS_WEIGHT_ZERO_INDEX = 165;
    static final int GUIDE_PREVIOUS_WEIGHT_POSITIVE_INDEX = 166;
    static final int GUIDE_PREVIOUS_WEIGHT_INVALID_INDEX = 167;
    static final int GUIDE_PREVIOUS_POST_WEIGHT_ELIGIBLE_INDEX = 168;
    static final int GUIDE_PREVIOUS_POST_WEIGHT_NEXT_VALID_INDEX = 169;
    static final int GUIDE_PREVIOUS_POST_WEIGHT_NEXT_INVALID_INDEX = 170;
    static final int GUIDE_PREVIOUS_POST_WEIGHT_CURRENT_VALID_INDEX = 171;
    static final int GUIDE_PREVIOUS_POST_WEIGHT_CURRENT_INVALID_INDEX = 172;
    static final int GUIDE_PREVIOUS_POST_WEIGHT_SOURCE_VALID_INDEX = 173;
    static final int GUIDE_PREVIOUS_POST_WEIGHT_SOURCE_INVALID_INDEX = 174;
    static final int GUIDE_PREVIOUS_SELECTION_PROBABILITY_ELIGIBLE_INDEX = 175;
    static final int GUIDE_PREVIOUS_SELECTION_DENOMINATOR_VALID_INDEX = 176;
    static final int GUIDE_PREVIOUS_SELECTION_DENOMINATOR_INVALID_INDEX = 177;
    static final int GUIDE_PREVIOUS_SELECTION_PROBABILITY_VALID_INDEX = 178;
    static final int GUIDE_PREVIOUS_SELECTION_PROBABILITY_INVALID_INDEX = 179;
    static final int GUIDE_PREVIOUS_SELECTION_CURRENT_DENOMINATOR_VALID_INDEX = 180;
    static final int GUIDE_PREVIOUS_SELECTION_CURRENT_DENOMINATOR_INVALID_INDEX = 181;
    static final int GUIDE_PREVIOUS_SELECTION_SOURCE_DENOMINATOR_VALID_INDEX = 182;
    static final int GUIDE_PREVIOUS_SELECTION_SOURCE_DENOMINATOR_INVALID_INDEX = 183;
    static final int GUIDE_PREVIOUS_SELECTION_PROBABILITY_HIGH_REJECT_INDEX = 184;
    static final int GUIDE_PREVIOUS_STABLE_SELECTION_ELIGIBLE_INDEX = 185;
    static final int GUIDE_PREVIOUS_STABLE_SELECTION_POSITIVE_INDEX = 186;
    static final int GUIDE_PREVIOUS_STABLE_SELECTION_ZERO_INDEX = 187;
    static final int GUIDE_PREVIOUS_STABLE_SELECTION_INVALID_INDEX = 188;
    static final int GUIDE_PREVIOUS_STABLE_BERNOULLI_ELIGIBLE_INDEX = 189;
    static final int GUIDE_PREVIOUS_STABLE_BERNOULLI_SELECTED_INDEX = 190;
    static final int GUIDE_PREVIOUS_STABLE_BERNOULLI_RETAINED_INDEX = 191;
    static final int GUIDE_PREVIOUS_STABLE_BERNOULLI_INVALID_INDEX = 192;
    static final int GUIDE_PREVIOUS_BRANCH_ELIGIBLE_INDEX = 193;
    static final int GUIDE_PREVIOUS_BRANCH_SELECTED_FINAL_VALID_INDEX = 194;
    static final int GUIDE_PREVIOUS_BRANCH_SELECTED_FINAL_INVALID_INDEX = 195;
    static final int GUIDE_PREVIOUS_BRANCH_RETAINED_FINAL_VALID_INDEX = 196;
    static final int GUIDE_PREVIOUS_BRANCH_RETAINED_FINAL_INVALID_INDEX = 197;
    static final int GUIDE_PREVIOUS_BRANCH_SELECTED_PAYLOAD_VALID_INDEX = 198;
    static final int GUIDE_PREVIOUS_BRANCH_SELECTED_PAYLOAD_INVALID_INDEX = 199;
    static final int GUIDE_PREVIOUS_BRANCH_RETAINED_PAYLOAD_VALID_INDEX = 200;
    static final int GUIDE_PREVIOUS_BRANCH_RETAINED_PAYLOAD_INVALID_INDEX = 201;
    static final int SHIFTED_DIAGNOSTIC_COUNTER_COUNT = 202;
    static final int SHIFTED_RECEIVER_GUIDE_STRIDE = 8 * Float.BYTES;
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
    private final ShiftedSnapshotState guideScratchSnapshotState = new ShiftedSnapshotState();
    private final RtBuffer[] slots = new RtBuffer[SLOT_COUNT];
    private RtBuffer spatialDiagnosticCounters;
    private RtBuffer spatialDiagnosticPairs;
    private RtBuffer shiftedSourceRoots;
    private RtBuffer shiftedMappedReservoirs;
    private RtBuffer shiftedReceiverGuides;
    private RtBuffer shiftedGuideScratchReservoirs;
    private RtBuffer shiftedGuideScratchSourceRoots;
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
        guideScratchSnapshotState.reset();
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
        if (spatialDiagnosticCounters == null || shiftedGuideScratchReservoirs == null
                || shiftedGuideScratchSourceRoots == null) {
            throw new IllegalStateException("Shifted-radiance diagnostics used before allocation");
        }
        VK10.vkCmdFillBuffer(cmd, spatialDiagnosticCounters.handle, 0L,
                spatialDiagnosticCounters.size, 0);
        try (var stack = org.lwjgl.system.MemoryStack.stackPush()) {
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
        }
        spatialDiagnosticViewPending = SHIFTED_RADIANCE_DEBUG_VIEW;
    }

    void beginCurrentGuideScratch(VkCommandBuffer cmd) {
        if (shiftedGuideScratchReservoirs == null
                || shiftedGuideScratchSourceRoots == null) {
            throw new IllegalStateException("Shifted guide scratch used before allocation");
        }
        VK10.vkCmdFillBuffer(cmd, shiftedGuideScratchReservoirs.handle, 0L,
                shiftedGuideScratchReservoirs.size, 0);
        VK10.vkCmdFillBuffer(cmd, shiftedGuideScratchSourceRoots.handle, 0L,
                shiftedGuideScratchSourceRoots.size, 0);
        try (var stack = org.lwjgl.system.MemoryStack.stackPush()) {
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
        }
    }

    void ensureShiftedSourceRoots(RtContext ctx) {
        if (!ready()) {
            throw new IllegalStateException("Shifted source roots used before path allocation");
        }
        if (shiftedSourceRoots != null && shiftedMappedReservoirs != null
                && shiftedReceiverGuides != null && shiftedGuideScratchReservoirs != null
                && shiftedGuideScratchSourceRoots != null) {
            return;
        }
        long rootBytes = Math.multiplyExact(Math.multiplyExact((long) width, height),
                PathSourceRootData.BYTE_SIZE);
        long mappedBytes = bytesPerSlot(width, height);
        long receiverGuideBytes = shiftedReceiverGuideBytes(width, height);
        shiftedSourceRoots = ctx.createBuffer(rootBytes, VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                false, "path shifted source roots " + width + "x" + height);
        shiftedMappedReservoirs = ctx.createBuffer(mappedBytes,
                VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                false, "path shifted mapped snapshot " + width + "x" + height);
        shiftedReceiverGuides = ctx.createBuffer(receiverGuideBytes,
                VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                false, "path shifted receiver guides " + width + "x" + height);
        shiftedGuideScratchReservoirs = ctx.createBuffer(mappedBytes,
                VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                false, "path shifted guide scratch reservoirs " + width + "x" + height);
        shiftedGuideScratchSourceRoots = ctx.createBuffer(rootBytes,
                VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                false, "path shifted guide scratch source roots " + width + "x" + height);
        CausticaMod.LOGGER.info(
                "RT path shifted snapshot: render={}x{}, rootStride={} B, mappedStride={} B, "
                        + "receiverGuideStride={} B, guideScratchStride={} B, "
                        + "guideRootScratchStride={} B, rootBytes={}, mappedBytes={}, "
                        + "receiverGuideBytes={}, guideScratchBytes={}, guideRootScratchBytes={}, "
                        + "gpuMiB={}",
                width, height, PathSourceRootData.BYTE_SIZE, BYTES_PER_RESERVOIR,
                SHIFTED_RECEIVER_GUIDE_STRIDE, BYTES_PER_RESERVOIR,
                PathSourceRootData.BYTE_SIZE, rootBytes, mappedBytes,
                receiverGuideBytes, mappedBytes, rootBytes,
                String.format(Locale.ROOT, "%.2f",
                        (rootBytes + mappedBytes + receiverGuideBytes + mappedBytes + rootBytes)
                                / (1024.0 * 1024.0)));
        shiftedSnapshotState.reset();
        guideScratchSnapshotState.reset();
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

    long shiftedReceiverGuideAddress() {
        return shiftedReceiverGuides == null ? 0L : shiftedReceiverGuides.deviceAddress;
    }

    long shiftedGuideScratchAddress() {
        return shiftedGuideScratchReservoirs == null
                ? 0L : shiftedGuideScratchReservoirs.deviceAddress;
    }

    long shiftedGuideRootScratchAddress() {
        return shiftedGuideScratchSourceRoots == null
                ? 0L : shiftedGuideScratchSourceRoots.deviceAddress;
    }

    boolean previousShiftedSnapshotAvailable(Frame frame, long frameIndex) {
        return shiftedSnapshotState.previousAvailable(frame, frameIndex);
    }

    boolean previousGuideScratchAvailable(Frame frame, long frameIndex) {
        return guideScratchSnapshotState.previousAvailable(frame, frameIndex);
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

    void commitGuideScratchSnapshot(Frame frame, long frameIndex) {
        guideScratchSnapshotState.commit(frame, frameIndex);
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
            long receiverGuideAttempted = Integer.toUnsignedLong(
                    counters.get(RECEIVER_GUIDE_ATTEMPTED_INDEX));
            long receiverGuideInvalid = Integer.toUnsignedLong(
                    counters.get(RECEIVER_GUIDE_INVALID_INDEX));
            long receiverGuideNoStoredEdge = Integer.toUnsignedLong(
                    counters.get(RECEIVER_GUIDE_NO_STORED_EDGE_INDEX));
            long receiverGuideStoredEligible = Integer.toUnsignedLong(
                    counters.get(RECEIVER_GUIDE_STORED_ELIGIBLE_INDEX));
            long receiverGuideAccepted = Integer.toUnsignedLong(
                    counters.get(RECEIVER_GUIDE_ACCEPTED_INDEX));
            long receiverGuideMassMismatch = Integer.toUnsignedLong(
                    counters.get(RECEIVER_GUIDE_MASS_MISMATCH_INDEX));
            long receiverGuideThroughputMismatch = Integer.toUnsignedLong(
                    counters.get(RECEIVER_GUIDE_THROUGHPUT_MISMATCH_INDEX));
            long receiverGuidePdfMismatch = Integer.toUnsignedLong(
                    counters.get(RECEIVER_GUIDE_PDF_MISMATCH_INDEX));
            long guideRemapEligible = Integer.toUnsignedLong(
                    counters.get(GUIDE_REMAP_ELIGIBLE_INDEX));
            long guideRemapReady = Integer.toUnsignedLong(
                    counters.get(GUIDE_REMAP_READY_INDEX));
            long guideRemapGeometryReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_REMAP_GEOMETRY_REJECT_INDEX));
            long guideRemapPdfReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_REMAP_PDF_REJECT_INDEX));
            long guideRemapThroughputReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_REMAP_THROUGHPUT_REJECT_INDEX));
            long guideVisibilityEligible = Integer.toUnsignedLong(
                    counters.get(GUIDE_VISIBILITY_ELIGIBLE_INDEX));
            long guideVisibilityClear = Integer.toUnsignedLong(
                    counters.get(GUIDE_VISIBILITY_CLEAR_INDEX));
            long guideVisibilityTinted = Integer.toUnsignedLong(
                    counters.get(GUIDE_VISIBILITY_TINTED_INDEX));
            long guideVisibilityOccluded = Integer.toUnsignedLong(
                    counters.get(GUIDE_VISIBILITY_OCCLUDED_INDEX));
            long guideVisibilityInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_VISIBILITY_INVALID_INDEX));
            long guideTargetEligible = Integer.toUnsignedLong(
                    counters.get(GUIDE_TARGET_ELIGIBLE_INDEX));
            long guideTargetPositive = Integer.toUnsignedLong(
                    counters.get(GUIDE_TARGET_POSITIVE_INDEX));
            long guideTargetZero = Integer.toUnsignedLong(
                    counters.get(GUIDE_TARGET_ZERO_INDEX));
            long guideTargetInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_TARGET_INVALID_INDEX));
            long guideWeightEligible = Integer.toUnsignedLong(
                    counters.get(GUIDE_WEIGHT_ELIGIBLE_INDEX));
            long guideWeightPositive = Integer.toUnsignedLong(
                    counters.get(GUIDE_WEIGHT_POSITIVE_INDEX));
            long guideWeightZero = Integer.toUnsignedLong(
                    counters.get(GUIDE_WEIGHT_ZERO_INDEX));
            long guideWeightInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_WEIGHT_INVALID_INDEX));
            long guideSelectionEligible = Integer.toUnsignedLong(
                    counters.get(GUIDE_SELECTION_ELIGIBLE_INDEX));
            long guideSelectionPositive = Integer.toUnsignedLong(
                    counters.get(GUIDE_SELECTION_POSITIVE_INDEX));
            long guideSelectionZero = Integer.toUnsignedLong(
                    counters.get(GUIDE_SELECTION_ZERO_INDEX));
            long guideSelectionCurrentReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_SELECTION_CURRENT_REJECT_INDEX));
            long guideSelectionProbabilityHighReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_SELECTION_PROBABILITY_HIGH_REJECT_INDEX));
            long guideSelectionArithmeticReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_SELECTION_ARITHMETIC_REJECT_INDEX));
            long guideStableSelectionEligible = Integer.toUnsignedLong(
                    counters.get(GUIDE_STABLE_SELECTION_ELIGIBLE_INDEX));
            long guideStableSelectionPositive = Integer.toUnsignedLong(
                    counters.get(GUIDE_STABLE_SELECTION_POSITIVE_INDEX));
            long guideStableSelectionZero = Integer.toUnsignedLong(
                    counters.get(GUIDE_STABLE_SELECTION_ZERO_INDEX));
            long guideStableSelectionInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_STABLE_SELECTION_INVALID_INDEX));
            long guideStableBernoulliEligible = Integer.toUnsignedLong(
                    counters.get(GUIDE_STABLE_BERNOULLI_ELIGIBLE_INDEX));
            long guideStableBernoulliSelected = Integer.toUnsignedLong(
                    counters.get(GUIDE_STABLE_BERNOULLI_SELECTED_INDEX));
            long guideStableBernoulliRetained = Integer.toUnsignedLong(
                    counters.get(GUIDE_STABLE_BERNOULLI_RETAINED_INDEX));
            long guideStableBernoulliInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_STABLE_BERNOULLI_INVALID_INDEX));
            long guidePostSelectionEligible = Integer.toUnsignedLong(
                    counters.get(GUIDE_POST_SELECTION_ELIGIBLE_INDEX));
            long guidePostSelectionSelectedReady = Integer.toUnsignedLong(
                    counters.get(GUIDE_POST_SELECTION_SELECTED_READY_INDEX));
            long guidePostSelectionRetainedReady = Integer.toUnsignedLong(
                    counters.get(GUIDE_POST_SELECTION_RETAINED_READY_INDEX));
            long guidePostSelectionEmpty = Integer.toUnsignedLong(
                    counters.get(GUIDE_POST_SELECTION_EMPTY_INDEX));
            long guidePostSelectionSampleReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_POST_SELECTION_SAMPLE_REJECT_INDEX));
            long guidePostSelectionArithmeticReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_POST_SELECTION_ARITHMETIC_REJECT_INDEX));
            long guideSampleCopyEligible = Integer.toUnsignedLong(
                    counters.get(GUIDE_SAMPLE_COPY_ELIGIBLE_INDEX));
            long guideSampleCopySelected = Integer.toUnsignedLong(
                    counters.get(GUIDE_SAMPLE_COPY_SELECTED_INDEX));
            long guideSampleCopyRetained = Integer.toUnsignedLong(
                    counters.get(GUIDE_SAMPLE_COPY_RETAINED_INDEX));
            long guideSampleCopyEmpty = Integer.toUnsignedLong(
                    counters.get(GUIDE_SAMPLE_COPY_EMPTY_INDEX));
            long guideSampleCopyMetadataReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_SAMPLE_COPY_METADATA_REJECT_INDEX));
            long guideSampleCopyArithmeticReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_SAMPLE_COPY_ARITHMETIC_REJECT_INDEX));
            long guideScratchStorageEligible = Integer.toUnsignedLong(
                    counters.get(GUIDE_SCRATCH_STORAGE_ELIGIBLE_INDEX));
            long guideScratchStorageSelected = Integer.toUnsignedLong(
                    counters.get(GUIDE_SCRATCH_STORAGE_SELECTED_INDEX));
            long guideScratchStorageRetained = Integer.toUnsignedLong(
                    counters.get(GUIDE_SCRATCH_STORAGE_RETAINED_INDEX));
            long guideScratchStorageEmpty = Integer.toUnsignedLong(
                    counters.get(GUIDE_SCRATCH_STORAGE_EMPTY_INDEX));
            long guideScratchStorageMetadataReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_SCRATCH_STORAGE_METADATA_REJECT_INDEX));
            long guideScratchStorageArithmeticReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_SCRATCH_STORAGE_ARITHMETIC_REJECT_INDEX));
            long guideScratchStorageInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_SCRATCH_STORAGE_INVALID_INDEX));
            long guideRootStorageEligible = Integer.toUnsignedLong(
                    counters.get(GUIDE_ROOT_STORAGE_ELIGIBLE_INDEX));
            long guideRootStorageSelectedReady = Integer.toUnsignedLong(
                    counters.get(GUIDE_ROOT_STORAGE_SELECTED_READY_INDEX));
            long guideRootStorageRetainedReady = Integer.toUnsignedLong(
                    counters.get(GUIDE_ROOT_STORAGE_RETAINED_READY_INDEX));
            long guideRootStorageMissing = Integer.toUnsignedLong(
                    counters.get(GUIDE_ROOT_STORAGE_MISSING_INDEX));
            long guideRootStorageMetadataReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_ROOT_STORAGE_METADATA_REJECT_INDEX));
            long guidePreviousReplayAttempted = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_REPLAY_ATTEMPTED_INDEX));
            long guidePreviousReplayLifecycleReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_REPLAY_LIFECYCLE_REJECT_INDEX));
            long guidePreviousReplayReceiverReprojectionReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_REPLAY_RECEIVER_REPROJECTION_REJECT_INDEX));
            long guidePreviousReplayEmpty = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_REPLAY_EMPTY_INDEX));
            long guidePreviousReplayMetadataReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_REPLAY_METADATA_REJECT_INDEX));
            long guidePreviousReplayReceiverSurfaceReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_REPLAY_RECEIVER_SURFACE_REJECT_INDEX));
            long guidePreviousReplaySourceReprojectionReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_REPLAY_SOURCE_REPROJECTION_REJECT_INDEX));
            long guidePreviousReplaySourceSurfaceReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_REPLAY_SOURCE_SURFACE_REJECT_INDEX));
            long guidePreviousReplaySourceReplayReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_REPLAY_SOURCE_REPLAY_REJECT_INDEX));
            long guidePreviousReplaySelectedAccepted = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_REPLAY_SELECTED_ACCEPTED_INDEX));
            long guidePreviousReplayRetainedAccepted = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_REPLAY_RETAINED_ACCEPTED_INDEX));
            long guidePreviousRemapEligible = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_REMAP_ELIGIBLE_INDEX));
            long guidePreviousRemapReceiverGuideReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_REMAP_RECEIVER_GUIDE_REJECT_INDEX));
            long guidePreviousRemapReady = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_REMAP_READY_INDEX));
            long guidePreviousRemapGeometryReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_REMAP_GEOMETRY_REJECT_INDEX));
            long guidePreviousRemapPdfReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_REMAP_PDF_REJECT_INDEX));
            long guidePreviousRemapThroughputReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_REMAP_THROUGHPUT_REJECT_INDEX));
            long guidePreviousVisibilityEligible = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_VISIBILITY_ELIGIBLE_INDEX));
            long guidePreviousVisibilityClear = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_VISIBILITY_CLEAR_INDEX));
            long guidePreviousVisibilityTinted = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_VISIBILITY_TINTED_INDEX));
            long guidePreviousVisibilityOccluded = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_VISIBILITY_OCCLUDED_INDEX));
            long guidePreviousVisibilityInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_VISIBILITY_INVALID_INDEX));
            long guidePreviousTargetEligible = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_TARGET_ELIGIBLE_INDEX));
            long guidePreviousTargetPositive = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_TARGET_POSITIVE_INDEX));
            long guidePreviousTargetZero = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_TARGET_ZERO_INDEX));
            long guidePreviousTargetInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_TARGET_INVALID_INDEX));
            long guidePreviousStoredMetadataEligible = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_STORED_METADATA_ELIGIBLE_INDEX));
            long guidePreviousStoredTargetMatch = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_STORED_TARGET_MATCH_INDEX));
            long guidePreviousStoredTargetMismatch = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_STORED_TARGET_MISMATCH_INDEX));
            long guidePreviousStoredTargetInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_STORED_TARGET_INVALID_INDEX));
            long guidePreviousStoredThroughputMatch = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_STORED_THROUGHPUT_MATCH_INDEX));
            long guidePreviousStoredThroughputMismatch = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_STORED_THROUGHPUT_MISMATCH_INDEX));
            long guidePreviousStoredThroughputInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_STORED_THROUGHPUT_INVALID_INDEX));
            long guidePreviousWeightEligible = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_WEIGHT_ELIGIBLE_INDEX));
            long guidePreviousWeightZero = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_WEIGHT_ZERO_INDEX));
            long guidePreviousWeightPositive = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_WEIGHT_POSITIVE_INDEX));
            long guidePreviousWeightInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_WEIGHT_INVALID_INDEX));
            long guidePreviousPostWeightEligible = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_POST_WEIGHT_ELIGIBLE_INDEX));
            long guidePreviousPostWeightNextValid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_POST_WEIGHT_NEXT_VALID_INDEX));
            long guidePreviousPostWeightNextInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_POST_WEIGHT_NEXT_INVALID_INDEX));
            long guidePreviousPostWeightCurrentValid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_POST_WEIGHT_CURRENT_VALID_INDEX));
            long guidePreviousPostWeightCurrentInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_POST_WEIGHT_CURRENT_INVALID_INDEX));
            long guidePreviousPostWeightSourceValid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_POST_WEIGHT_SOURCE_VALID_INDEX));
            long guidePreviousPostWeightSourceInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_POST_WEIGHT_SOURCE_INVALID_INDEX));
            long guidePreviousSelectionProbabilityEligible = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_SELECTION_PROBABILITY_ELIGIBLE_INDEX));
            long guidePreviousSelectionDenominatorValid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_SELECTION_DENOMINATOR_VALID_INDEX));
            long guidePreviousSelectionDenominatorInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_SELECTION_DENOMINATOR_INVALID_INDEX));
            long guidePreviousSelectionProbabilityValid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_SELECTION_PROBABILITY_VALID_INDEX));
            long guidePreviousSelectionProbabilityInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_SELECTION_PROBABILITY_INVALID_INDEX));
            long guidePreviousSelectionCurrentDenominatorValid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_SELECTION_CURRENT_DENOMINATOR_VALID_INDEX));
            long guidePreviousSelectionCurrentDenominatorInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_SELECTION_CURRENT_DENOMINATOR_INVALID_INDEX));
            long guidePreviousSelectionSourceDenominatorValid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_SELECTION_SOURCE_DENOMINATOR_VALID_INDEX));
            long guidePreviousSelectionSourceDenominatorInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_SELECTION_SOURCE_DENOMINATOR_INVALID_INDEX));
            long guidePreviousSelectionProbabilityHighReject = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_SELECTION_PROBABILITY_HIGH_REJECT_INDEX));
            long guidePreviousStableSelectionEligible = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_STABLE_SELECTION_ELIGIBLE_INDEX));
            long guidePreviousStableSelectionPositive = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_STABLE_SELECTION_POSITIVE_INDEX));
            long guidePreviousStableSelectionZero = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_STABLE_SELECTION_ZERO_INDEX));
            long guidePreviousStableSelectionInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_STABLE_SELECTION_INVALID_INDEX));
            long guidePreviousStableBernoulliEligible = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_STABLE_BERNOULLI_ELIGIBLE_INDEX));
            long guidePreviousStableBernoulliSelected = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_STABLE_BERNOULLI_SELECTED_INDEX));
            long guidePreviousStableBernoulliRetained = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_STABLE_BERNOULLI_RETAINED_INDEX));
            long guidePreviousStableBernoulliInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_STABLE_BERNOULLI_INVALID_INDEX));
            long guidePreviousBranchEligible = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_BRANCH_ELIGIBLE_INDEX));
            long guidePreviousBranchSelectedFinalValid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_BRANCH_SELECTED_FINAL_VALID_INDEX));
            long guidePreviousBranchSelectedFinalInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_BRANCH_SELECTED_FINAL_INVALID_INDEX));
            long guidePreviousBranchRetainedFinalValid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_BRANCH_RETAINED_FINAL_VALID_INDEX));
            long guidePreviousBranchRetainedFinalInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_BRANCH_RETAINED_FINAL_INVALID_INDEX));
            long guidePreviousBranchSelectedPayloadValid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_BRANCH_SELECTED_PAYLOAD_VALID_INDEX));
            long guidePreviousBranchSelectedPayloadInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_BRANCH_SELECTED_PAYLOAD_INVALID_INDEX));
            long guidePreviousBranchRetainedPayloadValid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_BRANCH_RETAINED_PAYLOAD_VALID_INDEX));
            long guidePreviousBranchRetainedPayloadInvalid = Integer.toUnsignedLong(
                    counters.get(GUIDE_PREVIOUS_BRANCH_RETAINED_PAYLOAD_INVALID_INDEX));
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
                             + "mapping={},pdf={},finite={}], "
                             + "receiverGuide[attempted={},invalid={},noStoredEdge={},"
                             + "storedEligible={},accepted={},mass={},throughput={},pdf={}], "
                             + "guideRemap[eligible={},ready={},geometry={},pdf={},throughput={}], "
                             + "guideVisibility[eligible={},clear={},tinted={},occluded={},invalid={}], "
                             + "guideTarget[eligible={},positive={},zero={},invalid={}], "
                             + "guideWeight[eligible={},positive={},zero={},invalid={}], "
                             + "guideSelection[eligible={},positive={},zero={},current={},high={},"
                             + "arithmetic={}], "
                             + "guideStableSelection[eligible={},positive={},zero={},invalid={}], "
                             + "guideStableBernoulli[eligible={},selected={},retained={},invalid={}], "
                             + "guidePostSelection[eligible={},selectedReady={},retainedReady={},"
                             + "empty={},sample={},arithmetic={}], "
                             + "guideSampleCopy[eligible={},selected={},retained={},empty={},"
                             + "metadata={},arithmetic={}]",
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
                     crossFrameEdgeFiniteReject,
                     receiverGuideAttempted, receiverGuideInvalid,
                     receiverGuideNoStoredEdge, receiverGuideStoredEligible,
                     receiverGuideAccepted, receiverGuideMassMismatch,
                     receiverGuideThroughputMismatch, receiverGuidePdfMismatch,
                     guideRemapEligible, guideRemapReady, guideRemapGeometryReject,
                     guideRemapPdfReject, guideRemapThroughputReject,
                     guideVisibilityEligible, guideVisibilityClear, guideVisibilityTinted,
                     guideVisibilityOccluded, guideVisibilityInvalid,
                     guideTargetEligible, guideTargetPositive, guideTargetZero, guideTargetInvalid,
                     guideWeightEligible, guideWeightPositive, guideWeightZero, guideWeightInvalid,
                     guideSelectionEligible, guideSelectionPositive, guideSelectionZero,
                     guideSelectionCurrentReject, guideSelectionProbabilityHighReject,
                     guideSelectionArithmeticReject,
                     guideStableSelectionEligible, guideStableSelectionPositive,
                     guideStableSelectionZero, guideStableSelectionInvalid,
                     guideStableBernoulliEligible, guideStableBernoulliSelected,
                     guideStableBernoulliRetained, guideStableBernoulliInvalid,
                     guidePostSelectionEligible, guidePostSelectionSelectedReady,
                     guidePostSelectionRetainedReady, guidePostSelectionEmpty,
                     guidePostSelectionSampleReject, guidePostSelectionArithmeticReject,
                     guideSampleCopyEligible, guideSampleCopySelected,
                     guideSampleCopyRetained, guideSampleCopyEmpty,
                     guideSampleCopyMetadataReject, guideSampleCopyArithmeticReject);
            long guideScratchStorageTerminal = guideScratchStorageSelected
                    + guideScratchStorageRetained + guideScratchStorageEmpty
                    + guideScratchStorageMetadataReject + guideScratchStorageArithmeticReject
                    + guideScratchStorageInvalid;
            CausticaMod.LOGGER.info(
                    "RT path guide scratch storage: eligible={}, selected={}, retained={}, empty={}, "
                            + "metadata={}, arithmetic={}, invalid={}, terminal={}, "
                            + "copyDelta[eligible={},selected={},retained={},empty={},metadata={},"
                            + "arithmetic={}]",
                    guideScratchStorageEligible, guideScratchStorageSelected,
                    guideScratchStorageRetained, guideScratchStorageEmpty,
                    guideScratchStorageMetadataReject, guideScratchStorageArithmeticReject,
                    guideScratchStorageInvalid, guideScratchStorageTerminal,
                    guideScratchStorageEligible - guideSampleCopyEligible,
                    guideScratchStorageSelected - guideSampleCopySelected,
                    guideScratchStorageRetained - guideSampleCopyRetained,
                    guideScratchStorageEmpty - guideSampleCopyEmpty,
                    guideScratchStorageMetadataReject - guideSampleCopyMetadataReject,
                    guideScratchStorageArithmeticReject - guideSampleCopyArithmeticReject);
            long guideRootStorageTerminal = guideRootStorageSelectedReady
                    + guideRootStorageRetainedReady + guideRootStorageMissing
                    + guideRootStorageMetadataReject;
            CausticaMod.LOGGER.info(
                    "RT path guide root storage: eligible={}, selectedReady={}, retainedReady={}, "
                            + "missing={}, metadata={}, terminal={}, "
                            + "scratchDelta[eligible={},selected={},retained={}]",
                    guideRootStorageEligible, guideRootStorageSelectedReady,
                    guideRootStorageRetainedReady, guideRootStorageMissing,
                    guideRootStorageMetadataReject, guideRootStorageTerminal,
                    guideRootStorageEligible
                            - guideScratchStorageSelected - guideScratchStorageRetained,
                    guideRootStorageSelectedReady - guideScratchStorageSelected,
                    guideRootStorageRetainedReady - guideScratchStorageRetained);
            long guidePreviousReplayTerminal = guidePreviousReplayLifecycleReject
                    + guidePreviousReplayReceiverReprojectionReject
                    + guidePreviousReplayEmpty + guidePreviousReplayMetadataReject
                    + guidePreviousReplayReceiverSurfaceReject
                    + guidePreviousReplaySourceReprojectionReject
                    + guidePreviousReplaySourceSurfaceReject
                    + guidePreviousReplaySourceReplayReject
                    + guidePreviousReplaySelectedAccepted
                    + guidePreviousReplayRetainedAccepted;
            CausticaMod.LOGGER.info(
                    "RT path guide previous replay: attempted={}, lifecycle={}, receiverReprojection={}, "
                            + "empty={}, metadata={}, receiverSurface={}, sourceReprojection={}, "
                            + "sourceSurface={}, sourceReplay={}, selectedAccepted={}, "
                            + "retainedAccepted={}, terminal={}, delta={}",
                    guidePreviousReplayAttempted, guidePreviousReplayLifecycleReject,
                    guidePreviousReplayReceiverReprojectionReject, guidePreviousReplayEmpty,
                    guidePreviousReplayMetadataReject, guidePreviousReplayReceiverSurfaceReject,
                    guidePreviousReplaySourceReprojectionReject,
                    guidePreviousReplaySourceSurfaceReject,
                    guidePreviousReplaySourceReplayReject,
                    guidePreviousReplaySelectedAccepted,
                    guidePreviousReplayRetainedAccepted, guidePreviousReplayTerminal,
                    guidePreviousReplayAttempted - guidePreviousReplayTerminal);
            long guidePreviousRemapTerminal = guidePreviousRemapReceiverGuideReject
                    + guidePreviousRemapGeometryReject + guidePreviousRemapPdfReject
                    + guidePreviousRemapThroughputReject + guidePreviousRemapReady;
            long guidePreviousVisibilityTerminal = guidePreviousVisibilityClear
                    + guidePreviousVisibilityTinted + guidePreviousVisibilityOccluded
                    + guidePreviousVisibilityInvalid;
            long guidePreviousTargetTerminal = guidePreviousTargetPositive
                    + guidePreviousTargetZero + guidePreviousTargetInvalid;
            CausticaMod.LOGGER.info(
                    "RT path guide previous remap: eligible={}, receiverGuide={}, geometry={}, "
                            + "pdf={}, throughput={}, ready={}, terminal={}, delta={}, "
                            + "visibility[eligible={},clear={},tinted={},occluded={},invalid={},delta={}] "
                            + "target[eligible={},positive={},zero={},invalid={},delta={}]",
                    guidePreviousRemapEligible, guidePreviousRemapReceiverGuideReject,
                    guidePreviousRemapGeometryReject, guidePreviousRemapPdfReject,
                    guidePreviousRemapThroughputReject, guidePreviousRemapReady,
                    guidePreviousRemapTerminal,
                    guidePreviousRemapEligible - guidePreviousRemapTerminal,
                    guidePreviousVisibilityEligible, guidePreviousVisibilityClear,
                    guidePreviousVisibilityTinted, guidePreviousVisibilityOccluded,
                    guidePreviousVisibilityInvalid,
                    guidePreviousVisibilityEligible - guidePreviousVisibilityTerminal,
                    guidePreviousTargetEligible, guidePreviousTargetPositive,
                    guidePreviousTargetZero, guidePreviousTargetInvalid,
                    guidePreviousTargetEligible - guidePreviousTargetTerminal);
            long guidePreviousStoredTargetTerminal = guidePreviousStoredTargetMatch
                    + guidePreviousStoredTargetMismatch + guidePreviousStoredTargetInvalid;
            long guidePreviousStoredThroughputTerminal = guidePreviousStoredThroughputMatch
                    + guidePreviousStoredThroughputMismatch
                    + guidePreviousStoredThroughputInvalid;
            CausticaMod.LOGGER.info(
                    "RT path guide previous stored metadata: eligible={}, "
                            + "target[match={},mismatch={},invalid={},delta={}] "
                            + "throughput[match={},mismatch={},invalid={},delta={}]",
                    guidePreviousStoredMetadataEligible,
                    guidePreviousStoredTargetMatch, guidePreviousStoredTargetMismatch,
                    guidePreviousStoredTargetInvalid,
                    guidePreviousStoredMetadataEligible - guidePreviousStoredTargetTerminal,
                    guidePreviousStoredThroughputMatch,
                    guidePreviousStoredThroughputMismatch,
                    guidePreviousStoredThroughputInvalid,
                    guidePreviousStoredMetadataEligible
                            - guidePreviousStoredThroughputTerminal);
            long guidePreviousWeightTerminal = guidePreviousWeightZero
                    + guidePreviousWeightPositive + guidePreviousWeightInvalid;
            CausticaMod.LOGGER.info(
                    "RT path guide previous weight: eligible={}, zero={}, positive={}, "
                            + "invalid={}, terminal={}, delta={}",
                    guidePreviousWeightEligible, guidePreviousWeightZero,
                    guidePreviousWeightPositive, guidePreviousWeightInvalid,
                    guidePreviousWeightTerminal,
                    guidePreviousWeightEligible - guidePreviousWeightTerminal);
            CausticaMod.LOGGER.info(
                    "RT path guide previous post-weight: eligible={}, "
                            + "next[valid={},invalid={},delta={}] "
                            + "current[valid={},invalid={},delta={}] "
                            + "source[valid={},invalid={},delta={}]",
                    guidePreviousPostWeightEligible,
                    guidePreviousPostWeightNextValid, guidePreviousPostWeightNextInvalid,
                    guidePreviousPostWeightEligible
                            - guidePreviousPostWeightNextValid
                            - guidePreviousPostWeightNextInvalid,
                    guidePreviousPostWeightCurrentValid,
                    guidePreviousPostWeightCurrentInvalid,
                    guidePreviousPostWeightEligible
                            - guidePreviousPostWeightCurrentValid
                            - guidePreviousPostWeightCurrentInvalid,
                    guidePreviousPostWeightSourceValid,
                    guidePreviousPostWeightSourceInvalid,
                    guidePreviousPostWeightEligible
                            - guidePreviousPostWeightSourceValid
                            - guidePreviousPostWeightSourceInvalid);
            CausticaMod.LOGGER.info(
                    "RT path guide previous selection denominator: eligible={}, "
                            + "next[valid={},invalid={},delta={}] "
                            + "cappedProbability[valid={},high={},invalid={},delta={}] "
                            + "current[valid={},invalid={},delta={}] "
                            + "source[valid={},invalid={},delta={}] "
                            + "stable[eligible={},positive={},zero={},invalid={},delta={}]",
                    guidePreviousSelectionProbabilityEligible,
                    guidePreviousSelectionDenominatorValid,
                    guidePreviousSelectionDenominatorInvalid,
                    guidePreviousSelectionProbabilityEligible
                            - guidePreviousSelectionDenominatorValid
                            - guidePreviousSelectionDenominatorInvalid,
                    guidePreviousSelectionProbabilityValid,
                    guidePreviousSelectionProbabilityHighReject,
                    guidePreviousSelectionProbabilityInvalid,
                    guidePreviousSelectionProbabilityEligible
                            - guidePreviousSelectionProbabilityValid
                            - guidePreviousSelectionProbabilityHighReject
                            - guidePreviousSelectionProbabilityInvalid,
                    guidePreviousSelectionCurrentDenominatorValid,
                    guidePreviousSelectionCurrentDenominatorInvalid,
                    guidePreviousSelectionProbabilityEligible
                            - guidePreviousSelectionCurrentDenominatorValid
                            - guidePreviousSelectionCurrentDenominatorInvalid,
                    guidePreviousSelectionSourceDenominatorValid,
                    guidePreviousSelectionSourceDenominatorInvalid,
                    guidePreviousSelectionProbabilityEligible
                            - guidePreviousSelectionSourceDenominatorValid
                            - guidePreviousSelectionSourceDenominatorInvalid,
                    guidePreviousStableSelectionEligible,
                    guidePreviousStableSelectionPositive,
                    guidePreviousStableSelectionZero,
                    guidePreviousStableSelectionInvalid,
                    guidePreviousStableSelectionEligible
                            - guidePreviousStableSelectionPositive
                            - guidePreviousStableSelectionZero
                            - guidePreviousStableSelectionInvalid);
            CausticaMod.LOGGER.info(
                    "RT path guide previous stable Bernoulli: eligible={}, "
                            + "selected={},retained={},invalid={},delta={}",
                    guidePreviousStableBernoulliEligible,
                    guidePreviousStableBernoulliSelected,
                    guidePreviousStableBernoulliRetained,
                    guidePreviousStableBernoulliInvalid,
                    guidePreviousStableBernoulliEligible
                            - guidePreviousStableBernoulliSelected
                            - guidePreviousStableBernoulliRetained
                            - guidePreviousStableBernoulliInvalid);
            long guidePreviousBranchSelected = guidePreviousStableBernoulliSelected;
            long guidePreviousBranchRetained = guidePreviousStableBernoulliRetained;
            CausticaMod.LOGGER.info(
                    "RT path guide previous branch arithmetic: eligible={}, "
                            + "selectedFinal[valid={},invalid={},delta={}] "
                            + "retainedFinal[valid={},invalid={},delta={}] "
                            + "selectedPayload[valid={},invalid={},delta={}] "
                            + "retainedPayload[valid={},invalid={},delta={}]",
                    guidePreviousBranchEligible,
                    guidePreviousBranchSelectedFinalValid,
                    guidePreviousBranchSelectedFinalInvalid,
                    guidePreviousBranchSelected
                            - guidePreviousBranchSelectedFinalValid
                            - guidePreviousBranchSelectedFinalInvalid,
                    guidePreviousBranchRetainedFinalValid,
                    guidePreviousBranchRetainedFinalInvalid,
                    guidePreviousBranchRetained
                            - guidePreviousBranchRetainedFinalValid
                            - guidePreviousBranchRetainedFinalInvalid,
                    guidePreviousBranchSelectedPayloadValid,
                    guidePreviousBranchSelectedPayloadInvalid,
                    guidePreviousBranchSelected
                            - guidePreviousBranchSelectedPayloadValid
                            - guidePreviousBranchSelectedPayloadInvalid,
                    guidePreviousBranchRetainedPayloadValid,
                    guidePreviousBranchRetainedPayloadInvalid,
                    guidePreviousBranchRetained
                            - guidePreviousBranchRetainedPayloadValid
                            - guidePreviousBranchRetainedPayloadInvalid);
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
        guideScratchSnapshotState.reset();
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

    static long shiftedReceiverGuideBytes(int width, int height) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("Shifted receiver-guide extent must be positive");
        }
        return Math.multiplyExact(Math.multiplyExact((long) width, height),
                SHIFTED_RECEIVER_GUIDE_STRIDE);
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
        if (shiftedReceiverGuides != null) {
            shiftedReceiverGuides.destroy();
            shiftedReceiverGuides = null;
        }
        if (shiftedGuideScratchReservoirs != null) {
            shiftedGuideScratchReservoirs.destroy();
            shiftedGuideScratchReservoirs = null;
        }
        if (shiftedGuideScratchSourceRoots != null) {
            shiftedGuideScratchSourceRoots.destroy();
            shiftedGuideScratchSourceRoots = null;
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
        guideScratchSnapshotState.reset();
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
