//for hdr write
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>   //for load obj in

#include <nvvk/descriptorsets_vk.hpp>   //for nvvk::DescriptorSetContainer
#include <nvh/fileoperations.hpp>  //for nvh::loadFile
#include <nvvk/shaders_vk.hpp>   //for creating shader Modules
#include <nvvk/context_vk.hpp>
#include <nvvk/structs_vk.hpp>  //Fpr nvvk::make
#include <nvvk/raytraceKHR_vk.hpp>  //For nvvk::RaytracingBuilderKHR
#include <nvvk/resourceallocator_vk.hpp>  //for nvvk memory allocators
#include <nvvk/error_vk.hpp>              // For NVVK_CHECK (check function running)
#include <cassert>
#include <array>

static const uint64_t render_width = 800;
static const uint64_t render_height = 600;
static const uint32_t workgroup_width = 16;
static const uint32_t workgroup_height = 8;

VkCommandBuffer AllocateAndBeginOneTimeCommandBuffer(VkDevice device, VkCommandPool cmdPool);
void EndSubmitWaitAndFreeCommandBuffer(VkDevice device, VkQueue queue, VkCommandPool cmdPool, VkCommandBuffer& cmdBuffer);
VkCommandBuffer AllocateAndBeginOneTimeCommandBuffer(VkDevice device, VkCommandPool cmdPool);
VkDeviceAddress GetBufferDeviceAddress(VkDevice device, VkBuffer buffer);


int main(int argc, const char** argv)
{
    //create the vulkan context given the context of nvidia
    nvvk::ContextCreateInfo deviceInfo;
    nvvk::Context context;      // encapsulates device state in a single object
    
    //request the exact api version we want utilize before initialization
    deviceInfo.apiMajor = 1;
    deviceInfo.apiMinor = 2;   //request 1.2.0

    //add required device extensions
    deviceInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures = nvvk::make<VkPhysicalDeviceAccelerationStructureFeaturesKHR>();
    deviceInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &asFeatures);

    VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures = nvvk::make<VkPhysicalDeviceRayQueryFeaturesKHR>();
    deviceInfo.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &rayQueryFeatures);

    //load into vulkan shared libraries
    context.init(deviceInfo);     //intialize the context

     // Device must support acceleration structures and ray queries: (use VK_False and VK_True to represent support)
    assert(asFeatures.accelerationStructure == VK_TRUE && rayQueryFeatures.rayQuery == VK_TRUE);

    // Create resource allocator (gpu-malloc) (generally 4096 limitation)
    nvvk::ResourceAllocatorDedicated allocator;
    allocator.init(context, context.m_physicalDevice);

    //Create a buffer holding the rendering image
    // host visible and host coherent for cpu memory mapping
    // storage buffer for gpu read and write 
    VkDeviceSize bufferSizeBytes = render_width * render_height * 3 * sizeof(float);   //3*float for rgb
    VkBufferCreateInfo bufferCreateInfo = nvvk::make<VkBufferCreateInfo>();
    bufferCreateInfo.size = bufferSizeBytes;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    // VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT means that the CPU can read this buffer's memory.
  // VK_MEMORY_PROPERTY_HOST_CACHED_BIT means that the CPU caches this memory.
  // VK_MEMORY_PROPERTY_HOST_COHERENT_BIT means that the CPU side of cache management
  // is handled automatically, with potentially slower reads/writes.

    nvvk::Buffer buffer = allocator.createBuffer(bufferCreateInfo,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);


    //define a path for finding shader
    const std::string exePath(argv[0], std::string(argv[0]).find_last_of("/\\")+1);  //for both unix and windows system
    std::vector<std::string> searchPaths = { exePath + PROJECT_RELDIRECTORY, exePath + PROJECT_RELDIRECTORY "..",
                                          exePath + PROJECT_RELDIRECTORY "../..", exePath + PROJECT_NAME };

    tinyobj::ObjReader reader;   //used  to read an OBJ file
    reader.ParseFromFile(nvh::findFile("scenes/Cornellbox-Original-Merged.obj", searchPaths));
    assert(reader.Valid());  // Make sure tinyobj was able to parse this file

    //retrive data from obj (vertex positions and colors, etc.)
    const std::vector<tinyobj::real_t> objVertices = reader.GetAttrib().GetVertices();

    //An Obj might contain multiple shapes for indices of meshes
    const std::vector<tinyobj::shape_t>& objShapes = reader.GetShapes(); //All shapes in the file
    assert(objShapes.size() == 1);   //check to make sure only one shape included
    const tinyobj::shape_t &objShape = objShapes[0];

    //get the indices from the shape mesh
    std::vector<uint32_t> objIndices;
    objIndices.reserve(objShape.mesh.indices.size());
    for (const tinyobj::index_t& index : objShape.mesh.indices)
    {
        objIndices.push_back(index.vertex_index);
    }

    //Create the command pool
    VkCommandPoolCreateInfo cmdPoolInfo = nvvk::make<VkCommandPoolCreateInfo>();
    cmdPoolInfo.queueFamilyIndex = context.m_queueGCT;  //graphics, compute, transfer
    VkCommandPool cmdPool;
    //check to make sure it is created, nullptr means default-cpu allocator
    NVVK_CHECK(vkCreateCommandPool(context, &cmdPoolInfo, nullptr, &cmdPool));

    //Upload obj data (CPU) to the GPU
    nvvk::Buffer vertexBuffer, indexBuffer;
    {
        //Start a command buffer for uploading the buffers
        VkCommandBuffer uploadCmdBuffer = AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);
        
        //create two buffers with usageFlags (as storage buffers)
        // Get these buffers' device addresses and use them as storage buffers
        const VkBufferUsageFlags usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

        vertexBuffer = allocator.createBuffer(uploadCmdBuffer, objVertices, usage);
        indexBuffer = allocator.createBuffer(uploadCmdBuffer, objIndices, usage);
    
         // end submit command buffer and release temporary stagin memory as well
         EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, uploadCmdBuffer);
         allocator.finalizeAndReleaseStaging();
    }

    //Describe the boottom-level acceleration structure (BLAS)
    // push back blas into vector
    std::vector<nvvk::RaytracingBuilderKHR::BlasInput> blases;
    {
        nvvk::RaytracingBuilderKHR::BlasInput blas;
        //Get the device address of the vertex and index buffers
        VkDeviceAddress  vertexBufferAddress = GetBufferDeviceAddress(context, vertexBuffer.buffer);
        VkDeviceAddress  indexBufferAddress  = GetBufferDeviceAddress(context, indexBuffer.buffer);
    
        // Specify where the builder can find the vertices and indices for triangles, and their formats:
        VkAccelerationStructureGeometryTrianglesDataKHR triangles = nvvk::make<VkAccelerationStructureGeometryTrianglesDataKHR>();
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData.deviceAddress = vertexBufferAddress;
        triangles.vertexStride = 3 * sizeof(float);    
        //objVertices = 3* Vertex
        triangles.maxVertex = static_cast<uint32_t>(objVertices.size()/3 - 1);  // [0, size/3-1]
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = indexBufferAddress;
        triangles.transformData.deviceAddress = 0; // No transofrm
        
        //Create a a VkAccelerationStructureGeometryKHR object that says it handles opaque triangles and points to the above:
        //handle geometry structure
        VkAccelerationStructureGeometryKHR geometry = nvvk::make<VkAccelerationStructureGeometryKHR>();
        geometry.geometry.triangles = triangles;
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR ;
        blas.asGeometry.push_back(geometry);

        // Create offset info that allows us to say how many triangles and vertices to read
        VkAccelerationStructureBuildRangeInfoKHR offsetInfo;
        offsetInfo.firstVertex = 0;
        offsetInfo.primitiveCount = static_cast<uint32_t>(objIndices.size() / 3);   // Number of triangles
        offsetInfo.primitiveOffset = 0;
        offsetInfo.transformOffset = 0;
        blas.asBuildOffsetInfo.push_back(offsetInfo);
        blases.push_back(blas);

    }

    //Create the BLAS with builder
    nvvk::RaytracingBuilderKHR raytracingBuilder;
    raytracingBuilder.setup(context, &allocator, context.m_queueGCT);
    raytracingBuilder.buildBlas(blases, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR);

    // Create an instance pointing to this BLAS, and build it into a TLAS:
    //TLAS ÔÚÉÏ²ã -> VkAccelerationStructureInstanceKHR
    std::vector<VkAccelerationStructureInstanceKHR> instances;
    {
    VkAccelerationStructureInstanceKHR instance{};
    instance.accelerationStructureReference = raytracingBuilder.getBlasDeviceAddress(0);  //point to the BLAS instance

    //Set the instance transform to the identity matrix
    instance.transform.matrix[0][0] = instance.transform.matrix[1][1] = instance.transform.matrix[2][2] = 1.0f;
    instance.instanceCustomIndex = 0;  //24 bits

    //Used for a shader offset index
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;  //Dedine how to trace this instance
    instance.mask = 0xFF;
    instances.push_back(instance);
    
    }
    //launch the TLAS build
    raytracingBuilder.buildTlas(instances, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

    //set up descrioptor set for binding
    // Here's the list of bindings for the descriptor set layout, from raytrace.comp.glsl:
    // 0 - a storage buffer (the buffer `buffer`)
    // That's it for now!

    //create a descriptor for trace rays from a shader
    nvvk::DescriptorSetContainer descriptorSetContainer(context);
    // 0 - a storage buffer
    // 1 - an acceleration structure (the TLAS)
    descriptorSetContainer.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);  //for the image
    descriptorSetContainer.addBinding(1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT);  //for the ray-tracing
    descriptorSetContainer.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);  //for the vertex buffer
    descriptorSetContainer.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);  //for the index buffer

    //Create a layout from the list of bindings
    descriptorSetContainer.initLayout();

    //Create a descriptor pool from the list of bindings with space for 1 set
    descriptorSetContainer.initPool(1);

    //Create a simple pipeline layout from the descriptor set layout
    descriptorSetContainer.initPipeLayout();

    //make the descriptor pointing to the buffer (descriptor for data be accessed from the compute shader)
    //Write a single write descriptor for the buffer in the descriptor set
    //Write values into the descriptor set (array for multiple descriptor set)

    std::array<VkWriteDescriptorSet, 4> writedescriptorSets;
    // 0   -image
    VkDescriptorBufferInfo descriptorBufferInfo{};
    descriptorBufferInfo.buffer = buffer.buffer;   //The VkBuffer Object
    descriptorBufferInfo.range = bufferSizeBytes;  // The Length of memory to bind
    writedescriptorSets[0] = descriptorSetContainer.makeWrite(0 /*set index*/, 0 /*binding*/, &descriptorBufferInfo);

    //1
    VkWriteDescriptorSetAccelerationStructureKHR descriptorAS  = nvvk::make<VkWriteDescriptorSetAccelerationStructureKHR>();
    VkAccelerationStructureKHR tlasCopy = raytracingBuilder.getAccelerationStructure();  // So that we can take its address
    descriptorAS.accelerationStructureCount = 1;
    descriptorAS.pAccelerationStructures = &tlasCopy;
    writedescriptorSets[1] = descriptorSetContainer.makeWrite(0, 1, &descriptorAS);

    //2   -vertex buffer
    VkDescriptorBufferInfo vertexDescriptorBufferInfo{};
    vertexDescriptorBufferInfo.buffer = vertexBuffer.buffer;
    vertexDescriptorBufferInfo.range = VK_WHOLE_SIZE;
    writedescriptorSets[2] = descriptorSetContainer.makeWrite(0, 2, &vertexDescriptorBufferInfo);

    //3 - index buffer
    VkDescriptorBufferInfo indexDescriptorBufferInfo{};
    indexDescriptorBufferInfo.buffer = indexBuffer.buffer;
    indexDescriptorBufferInfo.range = VK_WHOLE_SIZE;
    writedescriptorSets[3] = descriptorSetContainer.makeWrite(0, 3, &indexDescriptorBufferInfo);


    // update the descriptor to the context
    vkUpdateDescriptorSets(context,
                           static_cast<uint32_t>(writedescriptorSets.size()), // Number of VkWriteDescriptorSet objects
                           writedescriptorSets.data(),   // Pointer to VkWriteDescriptorSet objects
                           0, nullptr);   //an array of vkcopydescriptorset no for now

    //shader loading and pipeline creation
    VkShaderModule rayTraceModule =
    nvvk::createShaderModule(context, nvh::loadFile("shaders/raytrace.comp.glsl.spv", true, searchPaths));

    //Describes the entrypoint and the stage
    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = nvvk::make<VkPipelineShaderStageCreateInfo>();
    shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module = rayTraceModule;
    shaderStageCreateInfo.pName = "main";

    // For the moment, create an empty pipeline layout. You can ignore this code
    // for now; we'll replace it in the next chapter.
    //VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = nvvk::make<VkPipelineLayoutCreateInfo>();
    //pipelineLayoutCreateInfo.setLayoutCount = 0;
    //pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
    //VkPipelineLayout pipelineLayout;
    //NVVK_CHECK(vkCreatePipelineLayout(context, &pipelineLayoutCreateInfo, VK_NULL_HANDLE, &pipelineLayout));

    //create the compute pipeline with VKComputePipelineCreateInfo
    VkComputePipelineCreateInfo pipelineCreateInfo = nvvk::make<VkComputePipelineCreateInfo>();
    pipelineCreateInfo.stage = shaderStageCreateInfo;    //assign the shaders for this pipeline
    pipelineCreateInfo.layout = descriptorSetContainer.getPipeLayout();  //use the pipeline layour from descriptor

    VkPipeline computePipeline;
    NVVK_CHECK(vkCreateComputePipelines(context,   //Device
                VK_NULL_HANDLE,    //Pipeline cache (uses default)
                1, &pipelineCreateInfo, //Compute pipeline create info
                VK_NULL_HANDLE,    //Allocator (uses defult)
                &computePipeline));  //Output
                

   
    // Create and start recording a command buffer
    VkCommandBuffer cmdBuffer = AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);

    //Fill this buffer with filling data command
    //const float fillValue = 0.5f;
    //const uint32_t& fillValueU32 = reinterpret_cast<const uint32_t&>(fillValue);  //transfer to the fill buffer needed
    //vkCmdFillBuffer(cmdBuffer, buffer.buffer, 0, bufferSizeBytes, fillValueU32);  

    //fill this cmdbuffer with compute shader
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    //Bind the descriptor set to the subsequence compute dispatches
    VkDescriptorSet descriptorSet = descriptorSetContainer.getSet(0);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, descriptorSetContainer.getPipeLayout(), 0, 1,
                            &descriptorSet,0, nullptr);
    
    //Run the compute shader with workgroup 
    uint32_t groupnum_w = (uint32_t(render_width)+workgroup_width-1) / workgroup_width;
    uint32_t groupnum_h = (uint32_t(render_height)+workgroup_height-1) / workgroup_height;
    vkCmdDispatch(cmdBuffer, groupnum_w, groupnum_h, 1);

    // add memory barrier for making sure cpu can read (flush the GPU caches)
    // Add a command that says "Make it so that memory writes by the vkCmdFillBuffer call
    // are available to read from the CPU." (In other words, "Flush the GPU caches
    // so the CPU can read the data.") To do this, we use a memory barrier.
    VkMemoryBarrier memoryBarrier = nvvk::make<VkMemoryBarrier>();
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;  // make gpu transfer write happen 
    memoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT; //Readable by the CPU

    vkCmdPipelineBarrier(cmdBuffer,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,   // from the transfer/shader stage
    VK_PIPELINE_STAGE_HOST_BIT,   //to the cpu
    0,    // no special flags
    1, &memoryBarrier,   //an array of memory barriers
    0, nullptr, 0, nullptr);  // no other barriers

    //Simple GPU-CPU Sync
    EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, cmdBuffer);
    
    //Get the image data back from the GPU (map way)
    void* data = allocator.map(buffer);
    //float* fltData = reinterpret_cast<float*> (data);
    //printf("First three elements: %f, %f, %f\n", fltData[0], fltData[1], fltData[2]);
    //write into the image
    stbi_write_hdr("out.hdr", render_width, render_height, 3, reinterpret_cast<float*>(data));
    allocator.unmap(buffer);

    //Free the compute shader stuff
    vkDestroyPipeline(context, computePipeline, nullptr);
    vkDestroyShaderModule(context, rayTraceModule, nullptr);
    descriptorSetContainer.deinit();   

    raytracingBuilder.destroy();
    allocator.destroy(vertexBuffer);
    allocator.destroy(indexBuffer);

    //Free command buffers and pool
    vkDestroyCommandPool(context, cmdPool, nullptr);

    allocator.destroy(buffer);   //screen image buffer
    allocator.deinit();  //deinit allocator before nvvk context
    context.deinit();   //clean up the context at the end of the program
}


VkCommandBuffer AllocateAndBeginOneTimeCommandBuffer(VkDevice device, VkCommandPool cmdPool)
{
    VkCommandBufferAllocateInfo cmdAllocInfo = nvvk::make<VkCommandBufferAllocateInfo>();
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandPool = cmdPool;
    cmdAllocInfo.commandBufferCount = 1;
    VkCommandBuffer cmdBuffer;
    NVVK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmdBuffer));

    //Begin Recording
    VkCommandBufferBeginInfo beginInfo = nvvk::make<VkCommandBufferBeginInfo>();
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    NVVK_CHECK(vkBeginCommandBuffer(cmdBuffer, &beginInfo));
    return cmdBuffer;
}

void EndSubmitWaitAndFreeCommandBuffer(VkDevice device, VkQueue queue, VkCommandPool cmdPool, VkCommandBuffer& cmdBuffer)
{
    //End Recording
    NVVK_CHECK(vkEndCommandBuffer(cmdBuffer));
    //Subimt the command buffer (should batch commands and submit less)
    VkSubmitInfo submitInfo = nvvk::make<VkSubmitInfo>();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuffer;
    NVVK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

    //wait for the GPU to finish
    NVVK_CHECK(vkQueueWaitIdle(queue));
    vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
}

//get the address of a piece of memory on GPU
VkDeviceAddress GetBufferDeviceAddress(VkDevice device, VkBuffer buffer)
{
   VkBufferDeviceAddressInfo addressInfo = nvvk::make<VkBufferDeviceAddressInfo>();
   addressInfo.buffer = buffer;
   return vkGetBufferDeviceAddress(device, &addressInfo);
}

