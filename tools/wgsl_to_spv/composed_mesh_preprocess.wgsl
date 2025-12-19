// GPU mesh transforming and culling.
//
// This is a compute shader that expands each `MeshInputUniform` out to a full
// `MeshUniform` for each view before rendering. (Thus `MeshInputUniform` and
// `MeshUniform` are in a 1:N relationship.) It runs in parallel for all meshes
// for all views. As part of this process, the shader gathers each mesh's
// transform on the previous frame and writes it into the `MeshUniform` so that
// TAA works. It also performs frustum culling and occlusion culling, if
// requested.
//
// If occlusion culling is on, this shader runs twice: once to prepare the
// meshes that were visible last frame, and once to prepare the meshes that
// weren't visible last frame but became visible this frame. The two invocations
// are known as *early mesh preprocessing* and *late mesh preprocessing*
// respectively.

// Types needed for GPU mesh uniform building.


// Per-frame data that the CPU supplies to the GPU.
struct MeshInput {
    // The model transform.
    world_from_local: mat3x4<f32>,
    // The lightmap UV rect, packed into 64 bits.
    lightmap_uv_rect: vec2<u32>,
    // Various flags.
    flags: u32,
    previous_input_index: u32,
    first_vertex_index: u32,
    first_index_index: u32,
    index_count: u32,
    current_skin_index: u32,
    // Low 16 bits: index of the material inside the bind group data.
    // High 16 bits: index of the lightmap in the binding array.
    material_and_lightmap_bind_group_slot: u32,
    timestamp: u32,
    // User supplied index to identify the mesh instance
    tag: u32,
    pad: u32,
}

// The `wgpu` indirect parameters structure. This is a union of two structures.
// For more information, see the corresponding comment in
// `gpu_preprocessing.rs`.
struct IndirectParametersIndexed {
    // `vertex_count` or `index_count`.
    index_count: u32,
    // `instance_count` in both structures.
    instance_count: u32,
    // `first_vertex` or `first_index`.
    first_index: u32,
    // `base_vertex` or `first_instance`.
    base_vertex: u32,
    // A read-only copy of `instance_index`.
    first_instance: u32,
}

struct IndirectParametersNonIndexed {
    vertex_count: u32,
    instance_count: u32,
    base_vertex: u32,
    first_instance: u32,
}

struct IndirectParametersCpuMetadata {
    base_output_index: u32,
    batch_set_index: u32,
}

struct IndirectParametersGpuMetadata {
    mesh_index: u32,
#ifdef WRITE_INDIRECT_PARAMETERS_METADATA
    early_instance_count: atomic<u32>,
    late_instance_count: atomic<u32>,
#else   // WRITE_INDIRECT_PARAMETERS_METADATA
    early_instance_count: u32,
    late_instance_count: u32,
#endif  // WRITE_INDIRECT_PARAMETERS_METADATA
}

struct IndirectBatchSet {
    indirect_parameters_count: atomic<u32>,
    indirect_parameters_base: u32,
}
    IndirectParametersCpuMetadata, IndirectParametersGpuMetadata, MeshInput
}

struct Mesh {
    // Affine 4x3 matrices transposed to 3x4
    // Use bevy_render::maths::affine3_to_square to unpack
    world_from_local: mat3x4<f32>,
    previous_world_from_local: mat3x4<f32>,
    // 3x3 matrix packed in mat2x4 and f32 as:
    // [0].xyz, [1].x,
    // [1].yz, [2].xy
    // [2].z
    // Use bevy_pbr::mesh_functions::mat2x4_f32_to_mat3x3_unpack to unpack
    local_from_world_transpose_a: mat2x4<f32>,
    local_from_world_transpose_b: f32,
    // 'flags' is a bit field indicating various options. u32 is 32 bits so we have up to 32 options.
    flags: u32,
    lightmap_uv_rect: vec2<u32>,
    // The index of the mesh's first vertex in the vertex buffer.
    first_vertex_index: u32,
    current_skin_index: u32,
    // Low 16 bits: index of the material inside the bind group data.
    // High 16 bits: index of the lightmap in the binding array.
    material_and_lightmap_bind_group_slot: u32,
    // User supplied index to identify the mesh instance
    tag: u32,
    pad: u32,
};

#ifdef SKINNED
struct SkinnedMesh {
    data: array<mat4x4<f32>, 256u>,
};
#endif

#ifdef MORPH_TARGETS
struct MorphWeights {
    weights: array<vec4<f32>, 64u>, // 64 = 256 / 4 (256 = MAX_MORPH_WEIGHTS)
};
#endif

// [2^0, 2^16)
const MESH_FLAGS_VISIBILITY_RANGE_INDEX_BITS: u32     = (1u << 16u) - 1u;
const MESH_FLAGS_NO_FRUSTUM_CULLING_BIT: u32          = 1u << 28u;
const MESH_FLAGS_SHADOW_RECEIVER_BIT: u32             = 1u << 29u;
const MESH_FLAGS_TRANSMITTED_SHADOW_RECEIVER_BIT: u32 = 1u << 30u;
// if the flag is set, the sign is positive, else it is negative
const MESH_FLAGS_SIGN_DETERMINANT_MODEL_3X3_BIT: u32  = 1u << 31u;


struct ClusterableObject {
    // For point lights: the lower-right 2x2 values of the projection matrix [2][2] [2][3] [3][2] [3][3]
    // For spot lights: the direction (x,z), spot_scale and spot_offset
    light_custom_data: vec4<f32>,
    color_inverse_square_range: vec4<f32>,
    position_radius: vec4<f32>,
    // 'flags' is a bit field indicating various options. u32 is 32 bits so we have up to 32 options.
    flags: u32,
    shadow_depth_bias: f32,
    shadow_normal_bias: f32,
    spot_light_tan_angle: f32,
    soft_shadow_size: f32,
    shadow_map_near_z: f32,
    decal_index: u32,
    pad: f32,
};

const POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT: u32                    = 1u << 0u;
const POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVE: u32                  = 1u << 1u;
const POINT_LIGHT_FLAGS_VOLUMETRIC_BIT: u32                         = 1u << 2u;
const POINT_LIGHT_FLAGS_AFFECTS_LIGHTMAPPED_MESH_DIFFUSE_BIT: u32   = 1u << 3u;

struct DirectionalCascade {
    clip_from_world: mat4x4<f32>,
    texel_size: f32,
    far_bound: f32,
}

struct DirectionalLight {
    cascades: array<DirectionalCascade, #{MAX_CASCADES_PER_LIGHT}>,
    color: vec4<f32>,
    direction_to_light: vec3<f32>,
    // 'flags' is a bit field indicating various options. u32 is 32 bits so we have up to 32 options.
    flags: u32,
    soft_shadow_size: f32,
    shadow_depth_bias: f32,
    shadow_normal_bias: f32,
    num_cascades: u32,
    cascades_overlap_proportion: f32,
    depth_texture_base_index: u32,
    decal_index: u32,
    sun_disk_angular_size: f32,
    sun_disk_intensity: f32,
};

const DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT: u32                  = 1u << 0u;
const DIRECTIONAL_LIGHT_FLAGS_VOLUMETRIC_BIT: u32                       = 1u << 1u;
const DIRECTIONAL_LIGHT_FLAGS_AFFECTS_LIGHTMAPPED_MESH_DIFFUSE_BIT: u32 = 1u << 2u;

struct Lights {
    // NOTE: this array size must be kept in sync with the constants defined in bevy_pbr/src/render/light.rs
    directional_lights: array<DirectionalLight, #{MAX_DIRECTIONAL_LIGHTS}u>,
    ambient_color: vec4<f32>,
    // x/y/z dimensions and n_clusters in w
    cluster_dimensions: vec4<u32>,
    // xy are vec2<f32>(cluster_dimensions.xy) / vec2<f32>(view.width, view.height)
    //
    // For perspective projections:
    // z is cluster_dimensions.z / log(far / near)
    // w is cluster_dimensions.z * log(near) / log(far / near)
    //
    // For orthographic projections:
    // NOTE: near and far are +ve but -z is infront of the camera
    // z is -near
    // w is cluster_dimensions.z / (-far - -near)
    cluster_factors: vec4<f32>,
    n_directional_lights: u32,
    spot_light_shadowmap_offset: i32,
    ambient_light_affects_lightmapped_meshes: u32
};

struct Fog {
    base_color: vec4<f32>,
    directional_light_color: vec4<f32>,
    // `be` and `bi` are allocated differently depending on the fog mode
    //
    // For Linear Fog:
    //     be.x = start, be.y = end
    // For Exponential and ExponentialSquared Fog:
    //     be.x = density
    // For Atmospheric Fog:
    //     be = per-channel extinction density
    //     bi = per-channel inscattering density
    be: vec3<f32>,
    directional_light_exponent: f32,
    bi: vec3<f32>,
    mode: u32,
}

// Important: These must be kept in sync with `fog.rs`
const FOG_MODE_OFF: u32                   = 0u;
const FOG_MODE_LINEAR: u32                = 1u;
const FOG_MODE_EXPONENTIAL: u32           = 2u;
const FOG_MODE_EXPONENTIAL_SQUARED: u32   = 3u;
const FOG_MODE_ATMOSPHERIC: u32           = 4u;

#if AVAILABLE_STORAGE_BUFFER_BINDINGS >= 3
struct ClusterableObjects {
    data: array<ClusterableObject>,
};
struct ClusterLightIndexLists {
    data: array<u32>,
};
struct ClusterOffsetsAndCounts {
    data: array<array<vec4<u32>, 2>>,
};
#else
struct ClusterableObjects {
    data: array<ClusterableObject, 204u>,
};
struct ClusterLightIndexLists {
    // each u32 contains 4 u8 indices into the ClusterableObjects array
    data: array<vec4<u32>, 1024u>,
};
struct ClusterOffsetsAndCounts {
    // each u32 contains a 24-bit index into ClusterLightIndexLists in the high 24 bits
    // and an 8-bit count of the number of lights in the low 8 bits
    data: array<vec4<u32>, 1024u>,
};
#endif

struct LightProbe {
    // This is stored as the transpose in order to save space in this structure.
    // It'll be transposed in the `environment_map_light` function.
    light_from_world_transposed: mat3x4<f32>,
    cubemap_index: i32,
    intensity: f32,
    // Whether this light probe contributes diffuse light to lightmapped meshes.
    affects_lightmapped_mesh_diffuse: u32,
};

struct LightProbes {
    // This must match `MAX_VIEW_REFLECTION_PROBES` on the Rust side.
    reflection_probes: array<LightProbe, 8u>,
    irradiance_volumes: array<LightProbe, 8u>,
    reflection_probe_count: i32,
    irradiance_volume_count: i32,
    // The index of the view environment map cubemap binding, or -1 if there's
    // no such cubemap.
    view_cubemap_index: i32,
    // The smallest valid mipmap level for the specular environment cubemap
    // associated with the view.
    smallest_specular_mip_level_for_view: u32,
    // The intensity of the environment map associated with the view.
    intensity_for_view: f32,
    // Whether the environment map attached to the view affects the diffuse
    // lighting for lightmapped meshes.
    view_environment_map_affects_lightmapped_mesh_diffuse: u32,
};

// Settings for screen space reflections.
//
// For more information on these settings, see the documentation for
// `bevy_pbr::ssr::ScreenSpaceReflections`.
struct ScreenSpaceReflectionsSettings {
    perceptual_roughness_threshold: f32,
    thickness: f32,
    linear_steps: u32,
    linear_march_exponent: f32,
    bisection_steps: u32,
    use_secant: u32,
};

struct EnvironmentMapUniform {
    // Transformation matrix for the environment cubemaps in world space.
    transform: mat4x4<f32>,
};

// Shader version of the order independent transparency settings component.
struct OrderIndependentTransparencySettings {
  layers_count: i32,
  alpha_threshold: f32,
};

struct ClusteredDecal {
    local_from_world: mat4x4<f32>,
    base_color_texture_index: i32,
    normal_map_texture_index: i32,
    metallic_roughness_texture_index: i32,
    emissive_texture_index: i32,
    tag: u32,
    pad_a: u32,
    pad_b: u32,
    pad_c: u32,
}

struct ClusteredDecals {
    decals: array<ClusteredDecal>,
}
    view::View,
    globals::Globals,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> lights: types::Lights;
#ifdef NO_CUBE_ARRAY_TEXTURES_SUPPORT
@group(0) @binding(2) var point_shadow_textures: texture_depth_cube;
#else
@group(0) @binding(2) var point_shadow_textures: texture_depth_cube_array;
#endif
@group(0) @binding(3) var point_shadow_textures_comparison_sampler: sampler_comparison;
#ifdef PCSS_SAMPLERS_AVAILABLE
@group(0) @binding(4) var point_shadow_textures_linear_sampler: sampler;
#endif  // PCSS_SAMPLERS_AVAILABLE
#ifdef NO_ARRAY_TEXTURES_SUPPORT
@group(0) @binding(5) var directional_shadow_textures: texture_depth_2d;
#else
@group(0) @binding(5) var directional_shadow_textures: texture_depth_2d_array;
#endif
@group(0) @binding(6) var directional_shadow_textures_comparison_sampler: sampler_comparison;
#ifdef PCSS_SAMPLERS_AVAILABLE
@group(0) @binding(7) var directional_shadow_textures_linear_sampler: sampler;
#endif  // PCSS_SAMPLERS_AVAILABLE

#if AVAILABLE_STORAGE_BUFFER_BINDINGS >= 3
@group(0) @binding(8) var<storage> clusterable_objects: types::ClusterableObjects;
@group(0) @binding(9) var<storage> clusterable_object_index_lists: types::ClusterLightIndexLists;
@group(0) @binding(10) var<storage> cluster_offsets_and_counts: types::ClusterOffsetsAndCounts;
#else
@group(0) @binding(8) var<uniform> clusterable_objects: types::ClusterableObjects;
@group(0) @binding(9) var<uniform> clusterable_object_index_lists: types::ClusterLightIndexLists;
@group(0) @binding(10) var<uniform> cluster_offsets_and_counts: types::ClusterOffsetsAndCounts;
#endif

@group(0) @binding(11) var<uniform> globals: Globals;
@group(0) @binding(12) var<uniform> fog: types::Fog;
@group(0) @binding(13) var<uniform> light_probes: types::LightProbes;

const VISIBILITY_RANGE_UNIFORM_BUFFER_SIZE: u32 = 64u;
#if AVAILABLE_STORAGE_BUFFER_BINDINGS >= 6
@group(0) @binding(14) var<storage> visibility_ranges: array<vec4<f32>>;
#else
@group(0) @binding(14) var<uniform> visibility_ranges: array<vec4<f32>, VISIBILITY_RANGE_UNIFORM_BUFFER_SIZE>;
#endif

@group(0) @binding(15) var<uniform> ssr_settings: types::ScreenSpaceReflectionsSettings;
@group(0) @binding(16) var screen_space_ambient_occlusion_texture: texture_2d<f32>;
@group(0) @binding(17) var<uniform> environment_map_uniform: types::EnvironmentMapUniform;

// NB: If you change these, make sure to update `tonemapping_shared.wgsl` too.
@group(0) @binding(18) var dt_lut_texture: texture_3d<f32>;
@group(0) @binding(19) var dt_lut_sampler: sampler;

#ifdef MULTISAMPLED
#ifdef DEPTH_PREPASS
@group(0) @binding(20) var depth_prepass_texture: texture_depth_multisampled_2d;
#endif // DEPTH_PREPASS
#ifdef NORMAL_PREPASS
@group(0) @binding(21) var normal_prepass_texture: texture_multisampled_2d<f32>;
#endif // NORMAL_PREPASS
#ifdef MOTION_VECTOR_PREPASS
@group(0) @binding(22) var motion_vector_prepass_texture: texture_multisampled_2d<f32>;
#endif // MOTION_VECTOR_PREPASS

#else // MULTISAMPLED

#ifdef DEPTH_PREPASS
@group(0) @binding(20) var depth_prepass_texture: texture_depth_2d;
#endif // DEPTH_PREPASS
#ifdef NORMAL_PREPASS
@group(0) @binding(21) var normal_prepass_texture: texture_2d<f32>;
#endif // NORMAL_PREPASS
#ifdef MOTION_VECTOR_PREPASS
@group(0) @binding(22) var motion_vector_prepass_texture: texture_2d<f32>;
#endif // MOTION_VECTOR_PREPASS

#endif // MULTISAMPLED

#ifdef DEFERRED_PREPASS
@group(0) @binding(23) var deferred_prepass_texture: texture_2d<u32>;
#endif // DEFERRED_PREPASS

@group(0) @binding(24) var view_transmission_texture: texture_2d<f32>;
@group(0) @binding(25) var view_transmission_sampler: sampler;

#ifdef OIT_ENABLED
@group(0) @binding(26) var<storage, read_write> oit_layers: array<vec2<u32>>;
@group(0) @binding(27) var<storage, read_write> oit_layer_ids: array<atomic<i32>>;
@group(0) @binding(28) var<uniform> oit_settings: types::OrderIndependentTransparencySettings;
#endif // OIT_ENABLED

#ifdef ATMOSPHERE
@group(0) @binding(29) var atmosphere_transmittance_texture: texture_2d<f32>;
@group(0) @binding(30) var atmosphere_transmittance_sampler: sampler;
@group(0) @binding(31) var<storage> atmosphere_data: atmosphere::AtmosphereData;
#endif // ATMOSPHERE

#ifdef MULTIPLE_LIGHT_PROBES_IN_ARRAY
@group(1) @binding(0) var diffuse_environment_maps: binding_array<texture_cube<f32>, 8u>;
@group(1) @binding(1) var specular_environment_maps: binding_array<texture_cube<f32>, 8u>;
#else
@group(1) @binding(0) var diffuse_environment_map: texture_cube<f32>;
@group(1) @binding(1) var specular_environment_map: texture_cube<f32>;
#endif
@group(1) @binding(2) var environment_map_sampler: sampler;

#ifdef IRRADIANCE_VOLUMES_ARE_USABLE
#ifdef MULTIPLE_LIGHT_PROBES_IN_ARRAY
@group(1) @binding(3) var irradiance_volumes: binding_array<texture_3d<f32>, 8u>;
#else
@group(1) @binding(3) var irradiance_volume: texture_3d<f32>;
#endif
@group(1) @binding(4) var irradiance_volume_sampler: sampler;
#endif

#ifdef CLUSTERED_DECALS_ARE_USABLE
@group(1) @binding(5) var<storage> clustered_decals: types::ClusteredDecals;
@group(1) @binding(6) var clustered_decal_textures: binding_array<texture_2d<f32>, 8u>;
@group(1) @binding(7) var clustered_decal_sampler: sampler;
#endif  // CLUSTERED_DECALS_ARE_USABLE
// Occlusion culling utility functions.


fn get_aabb_size_in_pixels(aabb: vec4<f32>, depth_pyramid: texture_2d<f32>) -> vec2<f32> {
    let depth_pyramid_size_mip_0 = vec2<f32>(textureDimensions(depth_pyramid, 0));
    let aabb_width_pixels = (aabb.z - aabb.x) * depth_pyramid_size_mip_0.x;
    let aabb_height_pixels = (aabb.w - aabb.y) * depth_pyramid_size_mip_0.y;
    return vec2(aabb_width_pixels, aabb_height_pixels);
}

fn get_occluder_depth(
    aabb: vec4<f32>,
    aabb_pixel_size: vec2<f32>,
    depth_pyramid: texture_2d<f32>
) -> f32 {
    let aabb_width_pixels = aabb_pixel_size.x;
    let aabb_height_pixels = aabb_pixel_size.y;

    let depth_pyramid_size_mip_0 = vec2<f32>(textureDimensions(depth_pyramid, 0));
    let depth_level = max(0, i32(ceil(log2(max(aabb_width_pixels, aabb_height_pixels))))); // TODO: Naga doesn't like this being a u32
    let depth_pyramid_size = vec2<f32>(textureDimensions(depth_pyramid, depth_level));
    let aabb_top_left = vec2<u32>(aabb.xy * depth_pyramid_size);

    let depth_quad_a = textureLoad(depth_pyramid, aabb_top_left, depth_level).x;
    let depth_quad_b = textureLoad(depth_pyramid, aabb_top_left + vec2(1u, 0u), depth_level).x;
    let depth_quad_c = textureLoad(depth_pyramid, aabb_top_left + vec2(0u, 1u), depth_level).x;
    let depth_quad_d = textureLoad(depth_pyramid, aabb_top_left + vec2(1u, 1u), depth_level).x;
    return min(min(depth_quad_a, depth_quad_b), min(depth_quad_c, depth_quad_d));
}

struct PreviousViewUniforms {
    view_from_world: mat4x4<f32>,
    clip_from_world: mat4x4<f32>,
    clip_from_view: mat4x4<f32>,
    world_from_clip: mat4x4<f32>,
    view_from_clip: mat4x4<f32>,
}

@group(0) @binding(2) var<uniform> previous_view_uniforms: PreviousViewUniforms;

// Material bindings will be in @group(2)


/// World space:
/// +y is up

/// View space:
/// -z is forward, +x is right, +y is up
/// Forward is from the camera position into the scene.
/// (0.0, 0.0, -1.0) is linear distance of 1.0 in front of the camera's view relative to the camera's rotation
/// (0.0, 1.0, 0.0) is linear distance of 1.0 above the camera's view relative to the camera's rotation

/// NDC (normalized device coordinate):
/// https://www.w3.org/TR/webgpu/#coordinate-systems
/// (-1.0, -1.0) in NDC is located at the bottom-left corner of NDC
/// (1.0, 1.0) in NDC is located at the top-right corner of NDC
/// Z is depth where: 
///    1.0 is near clipping plane
///    Perspective projection: 0.0 is inf far away
///    Orthographic projection: 0.0 is far clipping plane

/// Clip space:
/// This is NDC before the perspective divide, still in homogenous coordinate space.
/// Dividing a clip space point by its w component yields a point in NDC space.

/// UV space:
/// 0.0, 0.0 is the top left
/// 1.0, 1.0 is the bottom right


// -----------------
// TO WORLD --------
// -----------------

/// Convert a view space position to world space
fn position_view_to_world(view_pos: vec3<f32>) -> vec3<f32> {
    let world_pos = view_bindings::view.world_from_view * vec4(view_pos, 1.0);
    return world_pos.xyz;
}

/// Convert a clip space position to world space
fn position_clip_to_world(clip_pos: vec4<f32>) -> vec3<f32> {
    let world_pos = view_bindings::view.world_from_clip * clip_pos;
    return world_pos.xyz;
}

/// Convert a ndc space position to world space
fn position_ndc_to_world(ndc_pos: vec3<f32>) -> vec3<f32> {
    let world_pos = view_bindings::view.world_from_clip * vec4(ndc_pos, 1.0);
    return world_pos.xyz / world_pos.w;
}

/// Convert a view space direction to world space
fn direction_view_to_world(view_dir: vec3<f32>) -> vec3<f32> {
    let world_dir = view_bindings::view.world_from_view * vec4(view_dir, 0.0);
    return world_dir.xyz;
}

/// Convert a clip space direction to world space
fn direction_clip_to_world(clip_dir: vec4<f32>) -> vec3<f32> {
    let world_dir = view_bindings::view.world_from_clip * clip_dir;
    return world_dir.xyz;
}

// -----------------
// TO VIEW ---------
// -----------------

/// Convert a world space position to view space
fn position_world_to_view(world_pos: vec3<f32>) -> vec3<f32> {
    let view_pos = view_bindings::view.view_from_world * vec4(world_pos, 1.0);
    return view_pos.xyz;
}

/// Convert a clip space position to view space
fn position_clip_to_view(clip_pos: vec4<f32>) -> vec3<f32> {
    let view_pos = view_bindings::view.view_from_clip * clip_pos;
    return view_pos.xyz;
}

/// Convert a ndc space position to view space
fn position_ndc_to_view(ndc_pos: vec3<f32>) -> vec3<f32> {
    let view_pos = view_bindings::view.view_from_clip * vec4(ndc_pos, 1.0);
    return view_pos.xyz / view_pos.w;
}

/// Convert a world space direction to view space
fn direction_world_to_view(world_dir: vec3<f32>) -> vec3<f32> {
    let view_dir = view_bindings::view.view_from_world * vec4(world_dir, 0.0);
    return view_dir.xyz;
}

/// Convert a clip space direction to view space
fn direction_clip_to_view(clip_dir: vec4<f32>) -> vec3<f32> {
    let view_dir = view_bindings::view.view_from_clip * clip_dir;
    return view_dir.xyz;
}

// -----------------
// TO PREV. VIEW ---
// -----------------

fn position_world_to_prev_view(world_pos: vec3<f32>) -> vec3<f32> {
    let view_pos = prepass_bindings::previous_view_uniforms.view_from_world *
        vec4(world_pos, 1.0);
    return view_pos.xyz;
}

fn position_world_to_prev_ndc(world_pos: vec3<f32>) -> vec3<f32> {
    let ndc_pos = prepass_bindings::previous_view_uniforms.clip_from_world *
        vec4(world_pos, 1.0);
    return ndc_pos.xyz / ndc_pos.w;
}

// -----------------
// TO CLIP ---------
// -----------------

/// Convert a world space position to clip space
fn position_world_to_clip(world_pos: vec3<f32>) -> vec4<f32> {
    let clip_pos = view_bindings::view.clip_from_world * vec4(world_pos, 1.0);
    return clip_pos;
}

/// Convert a view space position to clip space
fn position_view_to_clip(view_pos: vec3<f32>) -> vec4<f32> {
    let clip_pos = view_bindings::view.clip_from_view * vec4(view_pos, 1.0);
    return clip_pos;
}

/// Convert a world space direction to clip space
fn direction_world_to_clip(world_dir: vec3<f32>) -> vec4<f32> {
    let clip_dir = view_bindings::view.clip_from_world * vec4(world_dir, 0.0);
    return clip_dir;
}

/// Convert a view space direction to clip space
fn direction_view_to_clip(view_dir: vec3<f32>) -> vec4<f32> {
    let clip_dir = view_bindings::view.clip_from_view * vec4(view_dir, 0.0);
    return clip_dir;
}

// -----------------
// TO NDC ----------
// -----------------

/// Convert a world space position to ndc space
fn position_world_to_ndc(world_pos: vec3<f32>) -> vec3<f32> {
    let ndc_pos = view_bindings::view.clip_from_world * vec4(world_pos, 1.0);
    return ndc_pos.xyz / ndc_pos.w;
}

/// Convert a view space position to ndc space
fn position_view_to_ndc(view_pos: vec3<f32>) -> vec3<f32> {
    let ndc_pos = view_bindings::view.clip_from_view * vec4(view_pos, 1.0);
    return ndc_pos.xyz / ndc_pos.w;
}

// -----------------
// DEPTH -----------
// -----------------

/// Retrieve the perspective camera near clipping plane
fn perspective_camera_near() -> f32 {
    return view_bindings::view.clip_from_view[3][2];
}

/// Convert ndc depth to linear view z. 
/// Note: Depth values in front of the camera will be negative as -z is forward
fn depth_ndc_to_view_z(ndc_depth: f32) -> f32 {
#ifdef VIEW_PROJECTION_PERSPECTIVE
    return -perspective_camera_near() / ndc_depth;
#else ifdef VIEW_PROJECTION_ORTHOGRAPHIC
    return -(view_bindings::view.clip_from_view[3][2] - ndc_depth) / view_bindings::view.clip_from_view[2][2];
#else
    let view_pos = view_bindings::view.view_from_clip * vec4(0.0, 0.0, ndc_depth, 1.0);
    return view_pos.z / view_pos.w;
#endif
}

/// Convert linear view z to ndc depth. 
/// Note: View z input should be negative for values in front of the camera as -z is forward
fn view_z_to_depth_ndc(view_z: f32) -> f32 {
#ifdef VIEW_PROJECTION_PERSPECTIVE
    return -perspective_camera_near() / view_z;
#else ifdef VIEW_PROJECTION_ORTHOGRAPHIC
    return view_bindings::view.clip_from_view[3][2] + view_z * view_bindings::view.clip_from_view[2][2];
#else
    let ndc_pos = view_bindings::view.clip_from_view * vec4(0.0, 0.0, view_z, 1.0);
    return ndc_pos.z / ndc_pos.w;
#endif
}

fn prev_view_z_to_depth_ndc(view_z: f32) -> f32 {
#ifdef VIEW_PROJECTION_PERSPECTIVE
    return -perspective_camera_near() / view_z;
#else ifdef VIEW_PROJECTION_ORTHOGRAPHIC
    return prepass_bindings::previous_view_uniforms.clip_from_view[3][2] +
        view_z * prepass_bindings::previous_view_uniforms.clip_from_view[2][2];
#else
    let ndc_pos = prepass_bindings::previous_view_uniforms.clip_from_view *
        vec4(0.0, 0.0, view_z, 1.0);
    return ndc_pos.z / ndc_pos.w;
#endif
}

// -----------------
// UV --------------
// -----------------

/// Convert ndc space xy coordinate [-1.0 .. 1.0] to uv [0.0 .. 1.0]
fn ndc_to_uv(ndc: vec2<f32>) -> vec2<f32> {
    return ndc * vec2(0.5, -0.5) + vec2(0.5);
}

/// Convert uv [0.0 .. 1.0] coordinate to ndc space xy [-1.0 .. 1.0]
fn uv_to_ndc(uv: vec2<f32>) -> vec2<f32> {
    return uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
}

/// returns the (0.0, 0.0) .. (1.0, 1.0) position within the viewport for the current render target
/// [0 .. render target viewport size] eg. [(0.0, 0.0) .. (1280.0, 720.0)] to [(0.0, 0.0) .. (1.0, 1.0)]
fn frag_coord_to_uv(frag_coord: vec2<f32>) -> vec2<f32> {
    return (frag_coord - view_bindings::view.viewport.xy) / view_bindings::view.viewport.zw;
}

/// Convert frag coord to ndc
fn frag_coord_to_ndc(frag_coord: vec4<f32>) -> vec3<f32> {
    return vec3(uv_to_ndc(frag_coord_to_uv(frag_coord.xy)), frag_coord.z);
}

/// Convert ndc space xy coordinate [-1.0 .. 1.0] to [0 .. render target
/// viewport size]
fn ndc_to_frag_coord(ndc: vec2<f32>) -> vec2<f32> {
    return ndc_to_uv(ndc) * view_bindings::view.viewport.zw;
}
    position_world_to_ndc, position_world_to_view, ndc_to_uv, view_z_to_depth_ndc,
    position_world_to_prev_ndc, position_world_to_prev_view, prev_view_z_to_depth_ndc
}

const PI: f32 = 3.141592653589793;      // π
const PI_2: f32 = 6.283185307179586;    // 2π
const HALF_PI: f32 = 1.57079632679;     // π/2
const FRAC_PI_3: f32 = 1.0471975512;    // π/3
const E: f32 = 2.718281828459045;       // exp(1)

fn affine2_to_square(affine: mat3x2<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(
        vec3<f32>(affine[0].xy, 0.0),
        vec3<f32>(affine[1].xy, 0.0),
        vec3<f32>(affine[2].xy, 1.0),
    );
}

fn affine3_to_square(affine: mat3x4<f32>) -> mat4x4<f32> {
    return transpose(mat4x4<f32>(
        affine[0],
        affine[1],
        affine[2],
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    ));
}

fn mat2x4_f32_to_mat3x3_unpack(
    a: mat2x4<f32>,
    b: f32,
) -> mat3x3<f32> {
    return mat3x3<f32>(
        a[0].xyz,
        vec3<f32>(a[0].w, a[1].xy),
        vec3<f32>(a[1].zw, b),
    );
}

// Extracts the square portion of an affine matrix: i.e. discards the
// translation.
fn affine3_to_mat3x3(affine: mat4x3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(affine[0].xyz, affine[1].xyz, affine[2].xyz);
}

// Returns the inverse of a 3x3 matrix.
fn inverse_mat3x3(matrix: mat3x3<f32>) -> mat3x3<f32> {
    let tmp0 = cross(matrix[1], matrix[2]);
    let tmp1 = cross(matrix[2], matrix[0]);
    let tmp2 = cross(matrix[0], matrix[1]);
    let inv_det = 1.0 / dot(matrix[2], tmp2);
    return transpose(mat3x3<f32>(tmp0 * inv_det, tmp1 * inv_det, tmp2 * inv_det));
}

// Returns the inverse of an affine matrix.
//
// https://en.wikipedia.org/wiki/Affine_transformation#Groups
fn inverse_affine3(affine: mat4x3<f32>) -> mat4x3<f32> {
    let matrix3 = affine3_to_mat3x3(affine);
    let inv_matrix3 = inverse_mat3x3(matrix3);
    return mat4x3<f32>(inv_matrix3[0], inv_matrix3[1], inv_matrix3[2], -(inv_matrix3 * affine[3]));
}

// Extracts the upper 3x3 portion of a 4x4 matrix.
fn mat4x4_to_mat3x3(m: mat4x4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
}

// Copy the sign bit from B onto A.
// copysign allows proper handling of negative zero to match the rust implementation of orthonormalize
fn copysign(a: f32, b: f32) -> f32 {
    return bitcast<f32>((bitcast<u32>(a) & 0x7FFFFFFF) | (bitcast<u32>(b) & 0x80000000));
}

// Constructs a right-handed orthonormal basis from a given unit Z vector.
//
// NOTE: requires unit-length (normalized) input to function properly.
//
// https://jcgt.org/published/0006/01/01/paper.pdf
// this method of constructing a basis from a vec3 is also used by `glam::Vec3::any_orthonormal_pair`
// the construction of the orthonormal basis up and right vectors here needs to precisely match the rust
// implementation in bevy_light/spot_light.rs:spot_light_world_from_view
fn orthonormalize(z_basis: vec3<f32>) -> mat3x3<f32> {
    let sign = copysign(1.0, z_basis.z);
    let a = -1.0 / (sign + z_basis.z);
    let b = z_basis.x * z_basis.y * a;
    let x_basis = vec3(1.0 + sign * z_basis.x * z_basis.x * a, sign * b, -sign * z_basis.x);
    let y_basis = vec3(b, sign + z_basis.y * z_basis.y * a, -z_basis.y);
    return mat3x3(x_basis, y_basis, z_basis);
}

// Returns true if any part of a sphere is on the positive side of a plane.
//
// `sphere_center.w` should be 1.0.
//
// This is used for frustum culling.
fn sphere_intersects_plane_half_space(
    plane: vec4<f32>,
    sphere_center: vec4<f32>,
    sphere_radius: f32
) -> bool {
    return dot(plane, sphere_center) + sphere_radius > 0.0;
}

// Returns the distances along the ray to its intersections with a sphere
// centered at the origin.
//
// r: distance from the sphere center to the ray origin
// mu: cosine of the zenith angle
// sphere_radius: radius of the sphere
//
// Returns vec2(t0, t1). If there is no intersection, returns vec2(-1.0).
fn ray_sphere_intersect(r: f32, mu: f32, sphere_radius: f32) -> vec2<f32> {
    let discriminant = r * r * (mu * mu - 1.0) + sphere_radius * sphere_radius;
    
    // No intersection
    if discriminant < 0.0 {
        return vec2(-1.0);
    }
    
    let q = -r * mu;
    let sqrt_discriminant = sqrt(discriminant);
    
    // Return both intersection distances
    return vec2(
        q - sqrt_discriminant,
        q + sqrt_discriminant
    );
}

// pow() but safe for NaNs/negatives
fn powsafe(color: vec3<f32>, power: f32) -> vec3<f32> {
    return pow(abs(color), vec3(power)) * sign(color);
}

// https://en.wikipedia.org/wiki/Vector_projection#Vector_projection_2
fn project_onto(lhs: vec3<f32>, rhs: vec3<f32>) -> vec3<f32> {
    let other_len_sq_rcp = 1.0 / dot(rhs, rhs);
    return rhs * dot(lhs, rhs) * other_len_sq_rcp;
}

// Below are fast approximations of common irrational and trig functions. These
// are likely most useful when raymarching, for example, where complete numeric
// accuracy can be sacrificed for greater sample count.

// Slightly less accurate than fast_acos_4, but much simpler.
fn fast_acos(in_x: f32) -> f32 {
    let x = abs(in_x);
    var res = -0.156583 * x + HALF_PI;
    res *= sqrt(1.0 - x);
    return select(PI - res, res, in_x >= 0.0);
}

// 4th order polynomial approximation
// 4 VGRP, 16 ALU Full Rate
// 7 * 10^-5 radians precision
// Reference : Handbook of Mathematical Functions (chapter : Elementary Transcendental Functions), M. Abramowitz and I.A. Stegun, Ed.
fn fast_acos_4(x: f32) -> f32 {
    let x1 = abs(x);
    let x2 = x1 * x1;
    let x3 = x2 * x1;
    var s: f32;

    s = -0.2121144 * x1 + 1.5707288;
    s = 0.0742610 * x2 + s;
    s = -0.0187293 * x3 + s;
    s = sqrt(1.0 - x1) * s;

	// acos function mirroring
    return select(PI - s, s, x >= 0.0);
}

fn fast_atan2(y: f32, x: f32) -> f32 {
    var t0 = max(abs(x), abs(y));
    var t1 = min(abs(x), abs(y));
    var t3 = t1 / t0;
    var t4 = t3 * t3;

    t0 = 0.0872929;
    t0 = t0 * t4 - 0.301895;
    t0 = t0 * t4 + 1.0;
    t3 = t0 * t3;

    t3 = select(t3, (0.5 * PI) - t3, abs(y) > abs(x));
    t3 = select(t3, PI - t3, x < 0);
    t3 = select(-t3, t3, y > 0);

    return t3;
}

struct ColorGrading {
    balance: mat3x3<f32>,
    saturation: vec3<f32>,
    contrast: vec3<f32>,
    gamma: vec3<f32>,
    gain: vec3<f32>,
    lift: vec3<f32>,
    midtone_range: vec2<f32>,
    exposure: f32,
    hue: f32,
    post_saturation: f32,
}

struct View {
    clip_from_world: mat4x4<f32>,
    unjittered_clip_from_world: mat4x4<f32>,
    world_from_clip: mat4x4<f32>,
    world_from_view: mat4x4<f32>,
    view_from_world: mat4x4<f32>,
    // Typically a column-major right-handed projection matrix, one of either:
    //
    // Perspective (infinite reverse z)
    // ```
    // f = 1 / tan(fov_y_radians / 2)
    //
    // ⎡ f / aspect  0   0     0 ⎤
    // ⎢          0  f   0     0 ⎥
    // ⎢          0  0   0  near ⎥
    // ⎣          0  0  -1     0 ⎦
    // ```
    //
    // Orthographic
    // ```
    // w = right - left
    // h = top - bottom
    // d = far - near
    // cw = -right - left
    // ch = -top - bottom
    //
    // ⎡ 2 / w      0      0   cw / w ⎤
    // ⎢     0  2 / h      0   ch / h ⎥
    // ⎢     0      0  1 / d  far / d ⎥
    // ⎣     0      0      0        1 ⎦
    // ```
    //
    // `clip_from_view[3][3] == 1.0` is the standard way to check if a projection is orthographic
    //
    // Wgsl matrices are column major, so for example getting the near plane of a perspective projection is `clip_from_view[3][2]`
    //
    // Custom projections are also possible however.
    clip_from_view: mat4x4<f32>,
    view_from_clip: mat4x4<f32>,
    world_position: vec3<f32>,
    exposure: f32,
    // viewport(x_origin, y_origin, width, height)
    viewport: vec4<f32>,
    main_pass_viewport: vec4<f32>,
    // 6 world-space half spaces (normal: vec3, distance: f32) ordered left, right, top, bottom, near, far.
    // The normal vectors point towards the interior of the frustum.
    // A half space contains `p` if `normal.dot(p) + distance > 0.`
    frustum: array<vec4<f32>, 6>,
    color_grading: ColorGrading,
    mip_bias: f32,
    frame_count: u32,
};

/// World space:
/// +y is up

/// View space:
/// -z is forward, +x is right, +y is up
/// Forward is from the camera position into the scene.
/// (0.0, 0.0, -1.0) is linear distance of 1.0 in front of the camera's view relative to the camera's rotation
/// (0.0, 1.0, 0.0) is linear distance of 1.0 above the camera's view relative to the camera's rotation

/// NDC (normalized device coordinate):
/// https://www.w3.org/TR/webgpu/#coordinate-systems
/// (-1.0, -1.0) in NDC is located at the bottom-left corner of NDC
/// (1.0, 1.0) in NDC is located at the top-right corner of NDC
/// Z is depth where:
///    1.0 is near clipping plane
///    Perspective projection: 0.0 is inf far away
///    Orthographic projection: 0.0 is far clipping plane

/// Clip space:
/// This is NDC before the perspective divide, still in homogenous coordinate space.
/// Dividing a clip space point by its w component yields a point in NDC space.

/// UV space:
/// 0.0, 0.0 is the top left
/// 1.0, 1.0 is the bottom right


// -----------------
// TO WORLD --------
// -----------------

/// Convert a view space position to world space
fn position_view_to_world(view_pos: vec3<f32>, world_from_view: mat4x4<f32>) -> vec3<f32> {
    let world_pos = world_from_view * vec4(view_pos, 1.0);
    return world_pos.xyz;
}

/// Convert a clip space position to world space
fn position_clip_to_world(clip_pos: vec4<f32>, world_from_clip: mat4x4<f32>) -> vec3<f32> {
    let world_pos = world_from_clip * clip_pos;
    return world_pos.xyz;
}

/// Convert a ndc space position to world space
fn position_ndc_to_world(ndc_pos: vec3<f32>, world_from_clip: mat4x4<f32>) -> vec3<f32> {
    let world_pos = world_from_clip * vec4(ndc_pos, 1.0);
    return world_pos.xyz / world_pos.w;
}

/// Convert a view space direction to world space
fn direction_view_to_world(view_dir: vec3<f32>, world_from_view: mat4x4<f32>) -> vec3<f32> {
    let world_dir = world_from_view * vec4(view_dir, 0.0);
    return world_dir.xyz;
}

/// Convert a clip space direction to world space
fn direction_clip_to_world(clip_dir: vec4<f32>, world_from_clip: mat4x4<f32>) -> vec3<f32> {
    let world_dir = world_from_clip * clip_dir;
    return world_dir.xyz;
}

// -----------------
// TO VIEW ---------
// -----------------

/// Convert a world space position to view space
fn position_world_to_view(world_pos: vec3<f32>, view_from_world: mat4x4<f32>) -> vec3<f32> {
    let view_pos = view_from_world * vec4(world_pos, 1.0);
    return view_pos.xyz;
}

/// Convert a clip space position to view space
fn position_clip_to_view(clip_pos: vec4<f32>, view_from_clip: mat4x4<f32>) -> vec3<f32> {
    let view_pos = view_from_clip * clip_pos;
    return view_pos.xyz;
}

/// Convert a ndc space position to view space
fn position_ndc_to_view(ndc_pos: vec3<f32>, view_from_clip: mat4x4<f32>) -> vec3<f32> {
    let view_pos = view_from_clip * vec4(ndc_pos, 1.0);
    return view_pos.xyz / view_pos.w;
}

/// Convert a world space direction to view space
fn direction_world_to_view(world_dir: vec3<f32>, view_from_world: mat4x4<f32>) -> vec3<f32> {
    let view_dir = view_from_world * vec4(world_dir, 0.0);
    return view_dir.xyz;
}

/// Convert a clip space direction to view space
fn direction_clip_to_view(clip_dir: vec4<f32>, view_from_clip: mat4x4<f32>) -> vec3<f32> {
    let view_dir = view_from_clip * clip_dir;
    return view_dir.xyz;
}

// -----------------
// TO CLIP ---------
// -----------------

/// Convert a world space position to clip space
fn position_world_to_clip(world_pos: vec3<f32>, clip_from_world: mat4x4<f32>) -> vec4<f32> {
    let clip_pos = clip_from_world * vec4(world_pos, 1.0);
    return clip_pos;
}

/// Convert a view space position to clip space
fn position_view_to_clip(view_pos: vec3<f32>, clip_from_view: mat4x4<f32>) -> vec4<f32> {
    let clip_pos = clip_from_view * vec4(view_pos, 1.0);
    return clip_pos;
}

/// Convert a world space direction to clip space
fn direction_world_to_clip(world_dir: vec3<f32>, clip_from_world: mat4x4<f32>) -> vec4<f32> {
    let clip_dir = clip_from_world * vec4(world_dir, 0.0);
    return clip_dir;
}

/// Convert a view space direction to clip space
fn direction_view_to_clip(view_dir: vec3<f32>, clip_from_view: mat4x4<f32>) -> vec4<f32> {
    let clip_dir = clip_from_view * vec4(view_dir, 0.0);
    return clip_dir;
}

// -----------------
// TO NDC ----------
// -----------------

/// Convert a world space position to ndc space
fn position_world_to_ndc(world_pos: vec3<f32>, clip_from_world: mat4x4<f32>) -> vec3<f32> {
    let ndc_pos = clip_from_world * vec4(world_pos, 1.0);
    return ndc_pos.xyz / ndc_pos.w;
}

/// Convert a view space position to ndc space
fn position_view_to_ndc(view_pos: vec3<f32>, clip_from_view: mat4x4<f32>) -> vec3<f32> {
    let ndc_pos = clip_from_view * vec4(view_pos, 1.0);
    return ndc_pos.xyz / ndc_pos.w;
}

// -----------------
// DEPTH -----------
// -----------------

/// Retrieve the perspective camera near clipping plane
fn perspective_camera_near(clip_from_view: mat4x4<f32>) -> f32 {
    return clip_from_view[3][2];
}

/// Convert ndc depth to linear view z.
/// Note: Depth values in front of the camera will be negative as -z is forward
fn depth_ndc_to_view_z(ndc_depth: f32, clip_from_view: mat4x4<f32>, view_from_clip: mat4x4<f32>) -> f32 {
#ifdef VIEW_PROJECTION_PERSPECTIVE
    return -perspective_camera_near(clip_from_view) / ndc_depth;
#else ifdef VIEW_PROJECTION_ORTHOGRAPHIC
    return -(clip_from_view[3][2] - ndc_depth) / clip_from_view[2][2];
#else
    let view_pos = view_from_clip * vec4(0.0, 0.0, ndc_depth, 1.0);
    return view_pos.z / view_pos.w;
#endif
}

/// Convert linear view z to ndc depth.
/// Note: View z input should be negative for values in front of the camera as -z is forward
fn view_z_to_depth_ndc(view_z: f32, clip_from_view: mat4x4<f32>) -> f32 {
#ifdef VIEW_PROJECTION_PERSPECTIVE
    return -perspective_camera_near(clip_from_view) / view_z;
#else ifdef VIEW_PROJECTION_ORTHOGRAPHIC
    return clip_from_view[3][2] + view_z * clip_from_view[2][2];
#else
    let ndc_pos = clip_from_view * vec4(0.0, 0.0, view_z, 1.0);
    return ndc_pos.z / ndc_pos.w;
#endif
}

// -----------------
// UV --------------
// -----------------

/// Convert ndc space xy coordinate [-1.0 .. 1.0] to uv [0.0 .. 1.0]
fn ndc_to_uv(ndc: vec2<f32>) -> vec2<f32> {
    return ndc * vec2(0.5, -0.5) + vec2(0.5);
}

/// Convert uv [0.0 .. 1.0] coordinate to ndc space xy [-1.0 .. 1.0]
fn uv_to_ndc(uv: vec2<f32>) -> vec2<f32> {
    return uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
}

/// returns the (0.0, 0.0) .. (1.0, 1.0) position within the viewport for the current render target
/// [0 .. render target viewport size] eg. [(0.0, 0.0) .. (1280.0, 720.0)] to [(0.0, 0.0) .. (1.0, 1.0)]
fn frag_coord_to_uv(frag_coord: vec2<f32>, viewport: vec4<f32>) -> vec2<f32> {
    return (frag_coord - viewport.xy) / viewport.zw;
}

/// Convert frag coord to ndc
fn frag_coord_to_ndc(frag_coord: vec4<f32>, viewport: vec4<f32>) -> vec3<f32> {
    return vec3(uv_to_ndc(frag_coord_to_uv(frag_coord.xy, viewport)), frag_coord.z);
}

/// Convert ndc space xy coordinate [-1.0 .. 1.0] to [0 .. render target
/// viewport size]
fn ndc_to_frag_coord(ndc: vec2<f32>, viewport: vec4<f32>) -> vec2<f32> {
    return ndc_to_uv(ndc) * viewport.zw;
}

// Information about each mesh instance needed to cull it on GPU.
//
// At the moment, this just consists of its axis-aligned bounding box (AABB).
struct MeshCullingData {
    // The 3D center of the AABB in model space, padded with an extra unused
    // float value.
    aabb_center: vec4<f32>,
    // The 3D extents of the AABB in model space, divided by two, padded with
    // an extra unused float value.
    aabb_half_extents: vec4<f32>,
}

// One invocation of this compute shader: i.e. one mesh instance in a view.
struct PreprocessWorkItem {
    // The index of the `MeshInput` in the `current_input` buffer that we read
    // from.
    input_index: u32,
    // In direct mode, the index of the `Mesh` in `output` that we write to. In
    // indirect mode, the index of the `IndirectParameters` in
    // `indirect_parameters` that we write to.
    output_or_indirect_parameters_index: u32,
}

// The parameters for the indirect compute dispatch for the late mesh
// preprocessing phase.
struct LatePreprocessWorkItemIndirectParameters {
    // The number of workgroups we're going to dispatch.
    //
    // This value should always be equal to `ceil(work_item_count / 64)`.
    dispatch_x: atomic<u32>,
    // The number of workgroups in the Y direction; always 1.
    dispatch_y: u32,
    // The number of workgroups in the Z direction; always 1.
    dispatch_z: u32,
    // The precise number of work items.
    work_item_count: atomic<u32>,
    // Padding.
    //
    // This isn't the usual structure padding; it's needed because some hardware
    // requires indirect compute dispatch parameters to be aligned on 64-byte
    // boundaries.
    pad: vec4<u32>,
}

// These have to be in a structure because of Naga limitations on DX12.
struct PushConstants {
    // The offset into the `late_preprocess_work_item_indirect_parameters`
    // buffer.
    late_preprocess_work_item_indirect_offset: u32,
}

// The current frame's `MeshInput`.
@group(0) @binding(3) var<storage> current_input: array<MeshInput>;
// The `MeshInput` values from the previous frame.
@group(0) @binding(4) var<storage> previous_input: array<MeshInput>;
// Indices into the `MeshInput` buffer.
//
// There may be many indices that map to the same `MeshInput`.
@group(0) @binding(5) var<storage> work_items: array<PreprocessWorkItem>;
// The output array of `Mesh`es.
@group(0) @binding(6) var<storage, read_write> output: array<Mesh>;

#ifdef INDIRECT
// The array of indirect parameters for drawcalls.
@group(0) @binding(7) var<storage> indirect_parameters_cpu_metadata:
    array<IndirectParametersCpuMetadata>;

@group(0) @binding(8) var<storage, read_write> indirect_parameters_gpu_metadata:
    array<IndirectParametersGpuMetadata>;
#endif

#ifdef FRUSTUM_CULLING
// Data needed to cull the meshes.
//
// At the moment, this consists only of AABBs.
@group(0) @binding(9) var<storage> mesh_culling_data: array<MeshCullingData>;
#endif  // FRUSTUM_CULLING

#ifdef OCCLUSION_CULLING
@group(0) @binding(10) var depth_pyramid: texture_2d<f32>;

#ifdef EARLY_PHASE
@group(0) @binding(11) var<storage, read_write> late_preprocess_work_items:
    array<PreprocessWorkItem>;
#endif  // EARLY_PHASE

@group(0) @binding(12) var<storage, read_write> late_preprocess_work_item_indirect_parameters:
    array<LatePreprocessWorkItemIndirectParameters>;

var<push_constant> push_constants: PushConstants;
#endif  // OCCLUSION_CULLING

#ifdef FRUSTUM_CULLING
// Returns true if the view frustum intersects an oriented bounding box (OBB).
//
// `aabb_center.w` should be 1.0.
fn view_frustum_intersects_obb(
    world_from_local: mat4x4<f32>,
    aabb_center: vec4<f32>,
    aabb_half_extents: vec3<f32>,
) -> bool {

    for (var i = 0; i < 5; i += 1) {
        // Calculate relative radius of the sphere associated with this plane.
        let plane_normal = view.frustum[i];
        let relative_radius = dot(
            abs(
                vec3(
                    dot(plane_normal.xyz, world_from_local[0].xyz),
                    dot(plane_normal.xyz, world_from_local[1].xyz),
                    dot(plane_normal.xyz, world_from_local[2].xyz),
                )
            ),
            aabb_half_extents
        );

        // Check the frustum plane.
        if (!maths::sphere_intersects_plane_half_space(
                plane_normal, aabb_center, relative_radius)) {
            return false;
        }
    }

    return true;
}
#endif

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    // Figure out our instance index. If this thread doesn't correspond to any
    // index, bail.
    let instance_index = global_invocation_id.x;

#ifdef LATE_PHASE
    if (instance_index >= atomicLoad(&late_preprocess_work_item_indirect_parameters[
            push_constants.late_preprocess_work_item_indirect_offset].work_item_count)) {
        return;
    }
#else   // LATE_PHASE
    if (instance_index >= arrayLength(&work_items)) {
        return;
    }
#endif

    // Unpack the work item.
    let input_index = work_items[instance_index].input_index;
#ifdef INDIRECT
    let indirect_parameters_index = work_items[instance_index].output_or_indirect_parameters_index;

    // If we're the first mesh instance in this batch, write the index of our
    // `MeshInput` into the appropriate slot so that the indirect parameters
    // building shader can access it.
#ifndef LATE_PHASE
    // https://github.com/gfx-rs/wgpu/issues/4394
    if (instance_index == 0u) {
        indirect_parameters_gpu_metadata[indirect_parameters_index].mesh_index = input_index;
    } else if (work_items[instance_index - 1].output_or_indirect_parameters_index != indirect_parameters_index) {
        indirect_parameters_gpu_metadata[indirect_parameters_index].mesh_index = input_index;
    }
#endif  // LATE_PHASE

#else   // INDIRECT
    let mesh_output_index = work_items[instance_index].output_or_indirect_parameters_index;
#endif  // INDIRECT

    // Unpack the input matrix.
    let world_from_local_affine_transpose = current_input[input_index].world_from_local;
    let world_from_local = maths::affine3_to_square(world_from_local_affine_transpose);

    // Frustum cull if necessary.
#ifdef FRUSTUM_CULLING
    if ((current_input[input_index].flags & MESH_FLAGS_NO_FRUSTUM_CULLING_BIT) == 0u) {
        let aabb_center = mesh_culling_data[input_index].aabb_center.xyz;
        let aabb_half_extents = mesh_culling_data[input_index].aabb_half_extents.xyz;

        // Do an OBB-based frustum cull.
        let model_center = world_from_local * vec4(aabb_center, 1.0);
        if (!view_frustum_intersects_obb(world_from_local, model_center, aabb_half_extents)) {
            return;
        }
    }
#endif

    // See whether the `MeshInputUniform` was updated on this frame. If it
    // wasn't, then we know the transforms of this mesh must be identical to
    // those on the previous frame, and therefore we don't need to access the
    // `previous_input_index` (in fact, we can't; that index are only valid for
    // one frame and will be invalid).
    let timestamp = current_input[input_index].timestamp;
    let mesh_changed_this_frame = timestamp == view.frame_count;

    // Look up the previous model matrix, if it could have been.
    let previous_input_index = current_input[input_index].previous_input_index;
    var previous_world_from_local_affine_transpose: mat3x4<f32>;
    if (mesh_changed_this_frame && previous_input_index != 0xffffffffu) {
        previous_world_from_local_affine_transpose =
            previous_input[previous_input_index].world_from_local;
    } else {
        previous_world_from_local_affine_transpose = world_from_local_affine_transpose;
    }
    let previous_world_from_local =
        maths::affine3_to_square(previous_world_from_local_affine_transpose);

    // Occlusion cull if necessary. This is done by calculating the screen-space
    // axis-aligned bounding box (AABB) of the mesh and testing it against the
    // appropriate level of the depth pyramid (a.k.a. hierarchical Z-buffer). If
    // no part of the AABB is in front of the corresponding pixel quad in the
    // hierarchical Z-buffer, then this mesh must be occluded, and we can skip
    // rendering it.
#ifdef OCCLUSION_CULLING
    let aabb_center = mesh_culling_data[input_index].aabb_center.xyz;
    let aabb_half_extents = mesh_culling_data[input_index].aabb_half_extents.xyz;

    // Initialize the AABB and the maximum depth.
    let infinity = bitcast<f32>(0x7f800000u);
    let neg_infinity = bitcast<f32>(0xff800000u);
    var aabb = vec4(infinity, infinity, neg_infinity, neg_infinity);
    var max_depth_view = neg_infinity;

    // Build up the AABB by taking each corner of this mesh's OBB, transforming
    // it, and updating the AABB and depth accordingly.
    for (var i = 0u; i < 8u; i += 1u) {
        let local_pos = aabb_center + select(
            vec3(-1.0),
            vec3(1.0),
            vec3((i & 1) != 0, (i & 2) != 0, (i & 4) != 0)
        ) * aabb_half_extents;

#ifdef EARLY_PHASE
        // If we're in the early phase, we're testing against the last frame's
        // depth buffer, so we need to use the previous frame's transform.
        let prev_world_pos = (previous_world_from_local * vec4(local_pos, 1.0)).xyz;
        let view_pos = position_world_to_prev_view(prev_world_pos);
        let ndc_pos = position_world_to_prev_ndc(prev_world_pos);
#else   // EARLY_PHASE
        // Otherwise, if this is the late phase, we use the current frame's
        // transform.
        let world_pos = (world_from_local * vec4(local_pos, 1.0)).xyz;
        let view_pos = position_world_to_view(world_pos);
        let ndc_pos = position_world_to_ndc(world_pos);
#endif  // EARLY_PHASE

        let uv_pos = ndc_to_uv(ndc_pos.xy);

        // Update the AABB and maximum view-space depth.
        aabb = vec4(min(aabb.xy, uv_pos), max(aabb.zw, uv_pos));
        max_depth_view = max(max_depth_view, view_pos.z);
    }

    // Clip to the near plane to avoid the NDC depth becoming negative.
#ifdef EARLY_PHASE
    max_depth_view = min(-previous_view_uniforms.clip_from_view[3][2], max_depth_view);
#else   // EARLY_PHASE
    max_depth_view = min(-view.clip_from_view[3][2], max_depth_view);
#endif  // EARLY_PHASE

    // Figure out the depth of the occluder, and compare it to our own depth.

    let aabb_pixel_size = occlusion_culling::get_aabb_size_in_pixels(aabb, depth_pyramid);
    let occluder_depth_ndc =
        occlusion_culling::get_occluder_depth(aabb, aabb_pixel_size, depth_pyramid);

#ifdef EARLY_PHASE
    let max_depth_ndc = prev_view_z_to_depth_ndc(max_depth_view);
#else   // EARLY_PHASE
    let max_depth_ndc = view_z_to_depth_ndc(max_depth_view);
#endif

    // Are we culled out?
    if (max_depth_ndc < occluder_depth_ndc) {
#ifdef EARLY_PHASE
        // If this is the early phase, we need to make a note of this mesh so
        // that we examine it again in the late phase, so that we handle the
        // case in which a mesh that was invisible last frame became visible in
        // this frame.
        let output_work_item_index = atomicAdd(&late_preprocess_work_item_indirect_parameters[
            push_constants.late_preprocess_work_item_indirect_offset].work_item_count, 1u);
        if (output_work_item_index % 64u == 0u) {
            // Our workgroup size is 64, and the indirect parameters for the
            // late mesh preprocessing phase are counted in workgroups, so if
            // we're the first thread in this workgroup, bump the workgroup
            // count.
            atomicAdd(&late_preprocess_work_item_indirect_parameters[
                push_constants.late_preprocess_work_item_indirect_offset].dispatch_x, 1u);
        }

        // Enqueue a work item for the late prepass phase.
        late_preprocess_work_items[output_work_item_index].input_index = input_index;
        late_preprocess_work_items[output_work_item_index].output_or_indirect_parameters_index =
            indirect_parameters_index;
#endif  // EARLY_PHASE
        // This mesh is culled. Skip it.
        return;
    }
#endif  // OCCLUSION_CULLING

    // Calculate inverse transpose.
    let local_from_world_transpose = transpose(maths::inverse_affine3(transpose(
        world_from_local_affine_transpose)));

    // Pack inverse transpose.
    let local_from_world_transpose_a = mat2x4<f32>(
        vec4<f32>(local_from_world_transpose[0].xyz, local_from_world_transpose[1].x),
        vec4<f32>(local_from_world_transpose[1].yz, local_from_world_transpose[2].xy));
    let local_from_world_transpose_b = local_from_world_transpose[2].z;

    // Figure out the output index. In indirect mode, this involves bumping the
    // instance index in the indirect parameters metadata, which
    // `build_indirect_params.wgsl` will use to generate the actual indirect
    // parameters. Otherwise, this index was directly supplied to us.
#ifdef INDIRECT
#ifdef LATE_PHASE
    let batch_output_index = atomicLoad(
        &indirect_parameters_gpu_metadata[indirect_parameters_index].early_instance_count
    ) + atomicAdd(
        &indirect_parameters_gpu_metadata[indirect_parameters_index].late_instance_count,
        1u
    );
#else   // LATE_PHASE
    let batch_output_index = atomicAdd(
        &indirect_parameters_gpu_metadata[indirect_parameters_index].early_instance_count,
        1u
    );
#endif  // LATE_PHASE

    let mesh_output_index =
        indirect_parameters_cpu_metadata[indirect_parameters_index].base_output_index +
        batch_output_index;

#endif  // INDIRECT

    // Write the output.
    output[mesh_output_index].world_from_local = world_from_local_affine_transpose;
    output[mesh_output_index].previous_world_from_local =
        previous_world_from_local_affine_transpose;
    output[mesh_output_index].local_from_world_transpose_a = local_from_world_transpose_a;
    output[mesh_output_index].local_from_world_transpose_b = local_from_world_transpose_b;
    output[mesh_output_index].flags = current_input[input_index].flags;
    output[mesh_output_index].lightmap_uv_rect = current_input[input_index].lightmap_uv_rect;
    output[mesh_output_index].first_vertex_index = current_input[input_index].first_vertex_index;
    output[mesh_output_index].current_skin_index = current_input[input_index].current_skin_index;
    output[mesh_output_index].material_and_lightmap_bind_group_slot =
        current_input[input_index].material_and_lightmap_bind_group_slot;
    output[mesh_output_index].tag = current_input[input_index].tag;
}
