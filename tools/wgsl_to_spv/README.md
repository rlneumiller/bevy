# wgsl_to_spv

A small tool to mash Bevy flavored WGSL modules together with naga_oil and spit out SPIR-V.

Why: Debugging naga_oil imports and naga validation errors can be a pain. This tool lets you check if your shader actually compiles and validates against Vulkan standards without having to run the full Bevy engine.

## Build it

```bash
cargo build -p wgsl_to_spv --release
```

## Run it

```bash
./target/release/wgsl_to_spv path/to/shader.wgsl out.spv --def MY_FLAG=true
```

## What it does

- Resolves Imports: Uses naga_oil to handle #import and #define_import_path exactly like Bevy does.

- Shader Defs: Pass --def NAME=VALUE to toggle conditional logic (booleans and ints supported).

- Validation Dumps: If composition fails, it dumps everything to tools/wgsl_to_spv/module_dumps/ so you can actually see what the final WGSL looked like before it broke.

- SPIR-V Output: Generates a .spv file for use with spirv-val or spirv-dis.
Example: Debugging Bevy PBR
If you're hacking on mesh_preprocess.wgsl and want to see why Naga is complaining:

```bash
./target/release/wgsl_to_spv crates/bevy_pbr/src/render/mesh_preprocess.wgsl /tmp/mesh.spv --def OCCLUSION_CULLING=true --def EARLY_PHASE=true
```

Then check it with the standard Khronos tools:

```bash
spirv-val --target-env vulkan1.2 /tmp/mesh.spv
spirv-dis /tmp/mesh.spv -o /tmp/mesh.spvasm
```

## Missing tokens

If you find a token that isn't being resolved correctly, just add it to the shader_defs map in `src/main.rs`.

The tool is hardcoded to use the local workspace versions of `naga` and `naga_oil`
