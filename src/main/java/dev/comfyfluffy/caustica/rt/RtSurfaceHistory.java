package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.rt.accel.RtImage;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkImageCopy;

/**
 * Double-buffered deterministic surface history owned by the composite render graph.
 *
 * <p>The primary pass owns the current guide images. At the history-capture boundary their
 * normal/roughness and depth are copied into the write slot while a future temporal pass may read the
 * other slot. No lighting estimator consumes this yet; the resource and generation contract lands first.</p>
 */
final class RtSurfaceHistory {
    static final int SLOT_COUNT = 2;
    static final int BYTES_PER_PIXEL_PER_SLOT = 12; // RGBA16F normal/roughness + R32F depth

    record Frame(long generation, int writeSlot, int previousSlot, boolean previousAvailable) {
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

    private final State state = new State();
    private final RtImage[] normalRoughness = new RtImage[SLOT_COUNT];
    private final RtImage[] depth = new RtImage[SLOT_COUNT];
    private int width = -1;
    private int height = -1;

    void ensure(RtContext ctx, int requestedWidth, int requestedHeight) {
        if (ready() && width == requestedWidth && height == requestedHeight) {
            return;
        }
        destroy();
        width = requestedWidth;
        height = requestedHeight;
        for (int slot = 0; slot < SLOT_COUNT; slot++) {
            normalRoughness[slot] = ctx.createStorageImage(width, height,
                    VK10.VK_FORMAT_R16G16B16A16_SFLOAT,
                    "surface history normal roughness slot " + slot + " " + width + "x" + height);
            depth[slot] = ctx.createStorageImage(width, height, VK10.VK_FORMAT_R32_SFLOAT,
                    "surface history depth slot " + slot + " " + width + "x" + height);
        }
        state.reset();
    }

    Frame beginFrame(RtHistoryState.Frame historyFrame) {
        if (!ready()) {
            throw new IllegalStateException("Surface history used before allocation");
        }
        return state.begin(historyFrame);
    }

    void recordCapture(VkCommandBuffer cmd, MemoryStack stack, RtImage currentNormalRoughness,
                       RtImage currentDepth, Frame frame) {
        requireCompatible(currentNormalRoughness, "normal/roughness");
        requireCompatible(currentDepth, "depth");
        int slot = frame.writeSlot();
        VK10.vkCmdCopyImage(cmd, currentNormalRoughness.image, VK10.VK_IMAGE_LAYOUT_GENERAL,
                normalRoughness[slot].image, VK10.VK_IMAGE_LAYOUT_GENERAL, copyRegion(stack));
        VK10.vkCmdCopyImage(cmd, currentDepth.image, VK10.VK_IMAGE_LAYOUT_GENERAL,
                depth[slot].image, VK10.VK_IMAGE_LAYOUT_GENERAL, copyRegion(stack));
    }

    void commit(Frame frame) {
        state.commit(frame);
    }

    RtImage previousNormalRoughness(Frame frame) {
        return frame.previousAvailable() ? normalRoughness[frame.previousSlot()] : null;
    }

    RtImage previousDepth(Frame frame) {
        return frame.previousAvailable() ? depth[frame.previousSlot()] : null;
    }

    RtImage normalRoughnessSlot(int slot) {
        requireSlot(slot);
        return normalRoughness[slot];
    }

    RtImage depthSlot(int slot) {
        requireSlot(slot);
        return depth[slot];
    }

    long allocatedBytes() {
        if (!ready()) {
            return 0L;
        }
        return Math.multiplyExact(
                Math.multiplyExact((long) width, (long) height),
                (long) SLOT_COUNT * BYTES_PER_PIXEL_PER_SLOT);
    }

    void destroy() {
        for (int slot = 0; slot < SLOT_COUNT; slot++) {
            if (normalRoughness[slot] != null) {
                normalRoughness[slot].destroy();
                normalRoughness[slot] = null;
            }
            if (depth[slot] != null) {
                depth[slot].destroy();
                depth[slot] = null;
            }
        }
        width = -1;
        height = -1;
        state.reset();
    }

    private boolean ready() {
        return normalRoughness[0] != null && normalRoughness[1] != null
                && depth[0] != null && depth[1] != null;
    }

    private void requireSlot(int slot) {
        if (!ready()) {
            throw new IllegalStateException("Surface history used before allocation");
        }
        if (slot < 0 || slot >= SLOT_COUNT) {
            throw new IllegalArgumentException("Surface history slot out of range: " + slot);
        }
    }

    private void requireCompatible(RtImage image, String name) {
        if (image == null || image.width != width || image.height != height) {
            throw new IllegalArgumentException("Current " + name + " guide does not match surface history "
                    + width + "x" + height);
        }
    }

    private VkImageCopy.Buffer copyRegion(MemoryStack stack) {
        VkImageCopy.Buffer region = VkImageCopy.calloc(1, stack);
        region.get(0).srcSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                .mipLevel(0).baseArrayLayer(0).layerCount(1);
        region.get(0).dstSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                .mipLevel(0).baseArrayLayer(0).layerCount(1);
        region.get(0).extent().set(width, height, 1);
        return region;
    }
}
