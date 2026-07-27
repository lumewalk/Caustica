package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.pipeline.RtTemporalValidationPipeline;
import org.joml.Matrix4fc;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;

/**
 * Reprojects the current primary surface into the previous frame and produces conservative reuse
 * metadata. Bit 7 is the stable "history accepted" contract; lower bits retain rejection reasons for
 * diagnostics and future policy tuning.
 */
final class RtTemporalValidation {
    static final int BYTES_PER_PIXEL = Integer.BYTES;

    static final int HISTORY_AVAILABLE = 1 << 0;
    static final int REPROJECTED_IN_BOUNDS = 1 << 1;
    static final int SURFACE_PRESENT = 1 << 2;
    static final int NORMAL_COMPATIBLE = 1 << 3;
    static final int ROUGHNESS_COMPATIBLE = 1 << 4;
    static final int DEPTH_COMPATIBLE = 1 << 5;
    static final int VALID = 1 << 7;

    static final Policy DEFAULT_POLICY = new Policy(
            0.90f,
            0.20f,
            0.05f,
            0.005f,
            0.50f);

    record Policy(float normalCosMin, float roughnessTolerance,
                  float depthBaseTolerance, float depthRelativeTolerance,
                  float depthMaxTolerance) {
        Policy {
            if (!(normalCosMin >= -1.0f && normalCosMin <= 1.0f)) {
                throw new IllegalArgumentException("normal cosine threshold must be in [-1, 1]");
            }
            if (roughnessTolerance < 0.0f || depthBaseTolerance < 0.0f
                    || depthRelativeTolerance < 0.0f || depthMaxTolerance < depthBaseTolerance) {
                throw new IllegalArgumentException("temporal validation tolerances must be non-negative");
            }
        }

        float depthTolerance(float viewDistance) {
            return Math.clamp(viewDistance * depthRelativeTolerance,
                    depthBaseTolerance, depthMaxTolerance);
        }
    }

    private RtTemporalValidationPipeline pipeline;
    private RtImage metadata;
    private int width = -1;
    private int height = -1;

    void ensure(RtContext ctx, int requestedWidth, int requestedHeight,
                RtImage currentNormalRoughness, RtImage currentDepth, RtImage currentMotion,
                RtSurfaceHistory history, RtImage debugColor) {
        if (ready() && width == requestedWidth && height == requestedHeight) {
            return;
        }
        destroy();
        width = requestedWidth;
        height = requestedHeight;
        metadata = ctx.createStorageImage(width, height, VK10.VK_FORMAT_R32_UINT,
                "temporal validation metadata " + width + "x" + height);
        long[] previousNormalViews = new long[RtSurfaceHistory.SLOT_COUNT];
        long[] previousDepthViews = new long[RtSurfaceHistory.SLOT_COUNT];
        for (int slot = 0; slot < RtSurfaceHistory.SLOT_COUNT; slot++) {
            previousNormalViews[slot] = history.normalRoughnessSlot(slot).view;
            previousDepthViews[slot] = history.depthSlot(slot).view;
        }
        pipeline = RtTemporalValidationPipeline.create(
                ctx,
                currentNormalRoughness.view,
                currentDepth.view,
                currentMotion.view,
                previousNormalViews,
                previousDepthViews,
                metadata.view,
                debugColor.view);
    }

    void record(VkCommandBuffer cmd, RtSurfaceHistory.Frame historyFrame,
                Matrix4fc currentFromPreviousClip, int debugView, Matrix4fc projection) {
        if (!ready()) {
            throw new IllegalStateException("Temporal validation used before allocation");
        }
        int previousSlot = historyFrame.previousAvailable() ? historyFrame.previousSlot() : 0;
        Policy policy = DEFAULT_POLICY;
        pipeline.dispatch(cmd, previousSlot, width, height,
                currentFromPreviousClip, historyFrame.previousAvailable(), debugView,
                policy.normalCosMin(), policy.roughnessTolerance(),
                policy.depthBaseTolerance(), policy.depthRelativeTolerance(),
                policy.depthMaxTolerance(),
                projection.m22(), projection.m23(), projection.m32(), projection.m33());
    }

    RtImage metadata() {
        if (!ready()) {
            throw new IllegalStateException("Temporal validation used before allocation");
        }
        return metadata;
    }

    long allocatedBytes() {
        return ready() ? Math.multiplyExact(
                Math.multiplyExact((long) width, (long) height), BYTES_PER_PIXEL) : 0L;
    }

    boolean ready() {
        return pipeline != null && metadata != null;
    }

    void destroy() {
        // Drop descriptor references before freeing any image they name.
        if (pipeline != null) {
            pipeline.destroy();
            pipeline = null;
        }
        if (metadata != null) {
            metadata.destroy();
            metadata = null;
        }
        width = -1;
        height = -1;
    }
}
