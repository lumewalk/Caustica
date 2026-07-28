package dev.comfyfluffy.caustica.rt.pipeline;

import static dev.comfyfluffy.caustica.rt.RtContext.check;

import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.gen.WorldPushConstantsData;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkComputePipelineCreateInfo;
import org.lwjgl.vulkan.VkDescriptorBufferInfo;
import org.lwjgl.vulkan.VkDescriptorImageInfo;
import org.lwjgl.vulkan.VkDescriptorPoolCreateInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetAllocateInfo;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkDescriptorSetLayoutCreateInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo;
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo;
import org.lwjgl.vulkan.VkPushConstantRange;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

/** GPU temporal-admission diagnostics for path reservoirs; it deliberately performs no GRIS merge. */
public final class RtPathTemporalPipeline {
    private static final String SHADER_DIR = "/caustica/rt/";
    private static final int IMAGE_COUNT = 5;
    private static final int BUFFER_COUNT = 2;
    private static final int RESOURCE_COUNT = IMAGE_COUNT + BUFFER_COUNT;

    private final RtContext ctx;
    private final long descriptorSetLayout;
    private final long descriptorPool;
    private final long descriptorSet;
    private final long pipelineLayout;
    private final long pipeline;
    private boolean destroyed;

    private RtPathTemporalPipeline(RtContext ctx, long descriptorSetLayout, long descriptorPool,
                                   long descriptorSet, long pipelineLayout, long pipeline) {
        this.ctx = ctx;
        this.descriptorSetLayout = descriptorSetLayout;
        this.descriptorPool = descriptorPool;
        this.descriptorSet = descriptorSet;
        this.pipelineLayout = pipelineLayout;
        this.pipeline = pipeline;
    }

    public static RtPathTemporalPipeline create(RtContext ctx, long receiverMotionView,
                                                long validationMetadataView, long debugColorView,
                                                long receiverPositionMaterialView,
                                                long receiverNormalRoughnessView,
                                                long spatialDiagnosticCounterBuffer,
                                                long spatialDiagnosticPairBuffer) {
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer bindings =
                    VkDescriptorSetLayoutBinding.calloc(RESOURCE_COUNT, stack);
            for (int binding = 0; binding < RESOURCE_COUNT; binding++) {
                bindings.get(binding).binding(binding)
                        .descriptorType(binding < IMAGE_COUNT
                                ? VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                : VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1)
                        .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }
            VkDescriptorSetLayoutCreateInfo dslInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType$Default().pBindings(bindings);
            LongBuffer handle = stack.mallocLong(1);
            check(VK10.vkCreateDescriptorSetLayout(vk, dslInfo, null, handle),
                    "vkCreateDescriptorSetLayout(path temporal)");
            long descriptorSetLayout = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, descriptorSetLayout,
                    "path temporal descriptor set layout");

            VkDescriptorPoolSize.Buffer poolSize = VkDescriptorPoolSize.calloc(2, stack);
            poolSize.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(IMAGE_COUNT);
            poolSize.get(1).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(BUFFER_COUNT);
            VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType$Default().maxSets(1).pPoolSizes(poolSize);
            check(VK10.vkCreateDescriptorPool(vk, poolInfo, null, handle),
                    "vkCreateDescriptorPool(path temporal)");
            long descriptorPool = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, descriptorPool,
                    "path temporal descriptor pool");

            VkDescriptorSetAllocateInfo allocateInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType$Default().descriptorPool(descriptorPool)
                    .pSetLayouts(stack.longs(descriptorSetLayout));
            check(VK10.vkAllocateDescriptorSets(vk, allocateInfo, handle),
                    "vkAllocateDescriptorSets(path temporal)");
            long descriptorSet = handle.get(0);

            VkPushConstantRange.Buffer pushRange = VkPushConstantRange.calloc(1, stack);
            pushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                    .offset(0).size(WorldPushConstantsData.BYTE_SIZE);
            VkPipelineLayoutCreateInfo layoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                    .sType$Default().pSetLayouts(stack.longs(descriptorSetLayout))
                    .pPushConstantRanges(pushRange);
            check(VK10.vkCreatePipelineLayout(vk, layoutInfo, null, handle),
                    "vkCreatePipelineLayout(path temporal)");
            long pipelineLayout = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT, pipelineLayout,
                    "path temporal pipeline layout");

            long module = loadModule(vk, stack, "path_temporal_validate.comp.spv");
            VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack)
                    .sType$Default().stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                    .module(module).pName(stack.UTF8("main"));
            VkComputePipelineCreateInfo.Buffer pipelineInfo =
                    VkComputePipelineCreateInfo.calloc(1, stack);
            pipelineInfo.get(0).sType$Default().stage(stage).layout(pipelineLayout);
            check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE,
                    pipelineInfo, null, handle), "vkCreateComputePipelines(path temporal)");
            long pipeline = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, pipeline,
                    "path temporal admission pipeline");
            VK10.vkDestroyShaderModule(vk, module, null);

            bindImages(vk, stack, descriptorSet,
                    receiverMotionView, validationMetadataView, debugColorView,
                    receiverPositionMaterialView, receiverNormalRoughnessView,
                    spatialDiagnosticCounterBuffer, spatialDiagnosticPairBuffer);
            return new RtPathTemporalPipeline(ctx, descriptorSetLayout, descriptorPool,
                    descriptorSet, pipelineLayout, pipeline);
        }
    }

    public void dispatch(VkCommandBuffer cmd, int width, int height, ByteBuffer pushConstants) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd,
                     "path temporal admission")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipelineLayout, 0, stack.longs(descriptorSet), null);
            VK10.vkCmdPushConstants(cmd, pipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT,
                    0, pushConstants);
            VK10.vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        }
    }

    public void destroy() {
        if (destroyed) return;
        VkDevice vk = ctx.vk();
        VK10.vkDestroyPipeline(vk, pipeline, null);
        VK10.vkDestroyPipelineLayout(vk, pipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, descriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, descriptorSetLayout, null);
        destroyed = true;
    }

    private static void bindImages(VkDevice vk, MemoryStack stack, long set,
                                   long receiverMotionView, long validationMetadataView,
                                   long debugColorView, long receiverPositionMaterialView,
                                   long receiverNormalRoughnessView,
                                   long spatialDiagnosticCounterBuffer,
                                   long spatialDiagnosticPairBuffer) {
        long[] views = {receiverMotionView, validationMetadataView, debugColorView,
                receiverPositionMaterialView, receiverNormalRoughnessView};
        VkDescriptorImageInfo.Buffer imageInfos = VkDescriptorImageInfo.calloc(IMAGE_COUNT, stack);
        VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(RESOURCE_COUNT, stack);
        for (int binding = 0; binding < IMAGE_COUNT; binding++) {
            imageInfos.get(binding).imageView(views[binding])
                    .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            writes.get(binding).sType$Default().dstSet(set).dstBinding(binding)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .pImageInfo(VkDescriptorImageInfo.create(imageInfos.address(binding), 1));
        }
        long[] buffers = {spatialDiagnosticCounterBuffer, spatialDiagnosticPairBuffer};
        VkDescriptorBufferInfo.Buffer bufferInfos =
                VkDescriptorBufferInfo.calloc(BUFFER_COUNT, stack);
        for (int buffer = 0; buffer < BUFFER_COUNT; buffer++) {
            bufferInfos.get(buffer).buffer(buffers[buffer])
                    .offset(0L).range(VK10.VK_WHOLE_SIZE);
            int binding = IMAGE_COUNT + buffer;
            writes.get(binding).sType$Default().dstSet(set).dstBinding(binding)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .pBufferInfo(VkDescriptorBufferInfo.create(bufferInfos.address(buffer), 1));
        }
        VK10.vkUpdateDescriptorSets(vk, writes, null);
    }

    private static long loadModule(VkDevice vk, MemoryStack stack, String name) {
        byte[] bytes;
        try (InputStream in = RtPathTemporalPipeline.class.getResourceAsStream(SHADER_DIR + name)) {
            if (in == null) {
                throw new IllegalStateException("missing SPIR-V resource: " + SHADER_DIR + name);
            }
            bytes = in.readAllBytes();
        } catch (IOException e) {
            throw new IllegalStateException("failed to read SPIR-V resource: " + SHADER_DIR + name, e);
        }
        ByteBuffer code = MemoryUtil.memAlloc(bytes.length).put(bytes);
        code.flip();
        try {
            VkShaderModuleCreateInfo moduleInfo = VkShaderModuleCreateInfo.calloc(stack)
                    .sType$Default().pCode(code);
            LongBuffer module = stack.mallocLong(1);
            check(VK10.vkCreateShaderModule(vk, moduleInfo, null, module),
                    "vkCreateShaderModule(" + name + ")");
            return module.get(0);
        } finally {
            MemoryUtil.memFree(code);
        }
    }
}
