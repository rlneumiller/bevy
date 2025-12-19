use std::{
    collections::{HashMap, HashSet},
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::exit,
};

use bevy_shader::{Shader, ShaderImport};
use naga::back::spv::{self, WriterFlags};
use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga_oil::compose::{Composer, ShaderDefValue};
use walkdir::WalkDir;

/// Helper to compose WGSL using the repository's composer and emit SPIR-V
/// Emits helpful diagnostics on failure.
/// Usage: wgsl_to_spv <input.wgsl> [output.spv] [--def NAME=VALUE]
fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);

    let src = match args.next() {
        Some(src) => src,
        None => {
            eprintln!("Usage: wgsl_to_spv <input.wgsl> [output.spv] [--def NAME=VALUE ...]");
            exit(2);
        }
    };

    let out = args.next().unwrap_or_else(|| "out.spv".to_string());

    // Collect any shader defs provided as `--def NAME=VALUE` after the paths.
    let mut shader_defs_cli: HashMap<String, ShaderDefValue> = HashMap::new();

    for arg in args {
        if let Some(def) = arg.strip_prefix("--def") {
            let def = def.trim_start_matches('=');
            if let Some((k, v)) = def.split_once('=') {
                let v = v.trim();
                let val = if v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("false") {
                    ShaderDefValue::Bool(v.eq_ignore_ascii_case("true"))
                } else if let Ok(n) = v.parse::<i32>() {
                    ShaderDefValue::Int(n)
                } else if let Ok(n) = v.parse::<u32>() {
                    ShaderDefValue::UInt(n)
                } else {
                    // Fallback to Bool true for lone flags
                    ShaderDefValue::Bool(true)
                };
                shader_defs_cli.insert(k.to_string(), val);
            }
        }
    }

    let src_text = fs::read_to_string(&src)?;
    let root_shader = Shader::from_wgsl(src_text, src.clone());

    let workspace_root = env::current_dir()?;
    let module_index = build_module_index()?;

    let mut composer = Composer::non_validating().with_capabilities(Capabilities::all());

    let mut added_module_paths: Vec<PathBuf> = Vec::new();
    let mut visited_custom_modules = HashSet::new();
    let mut visited_asset_paths = HashSet::new();

    for import in root_shader.imports() {
        let result = match import {
            ShaderImport::Custom(name) => add_custom_module(
                &mut composer,
                name.trim(),
                &module_index,
                &workspace_root,
                &mut visited_custom_modules,
                &mut visited_asset_paths,
                &mut added_module_paths,
            ),
            ShaderImport::AssetPath(path) => add_asset_module(
                &mut composer,
                path.trim(),
                &module_index,
                &workspace_root,
                &mut visited_custom_modules,
                &mut visited_asset_paths,
                &mut added_module_paths,
            ),
        };

        if let Err(err) = result {
            eprintln!("Warning: {err}");
        }
    }

    // Confirm that entry shader is registered
    composer.add_composable_module((&root_shader).into())?;

    // Merge the provided shader defs from CLI with sensible defaults for Bevy
    let mut shader_defs: HashMap<String, ShaderDefValue> = shader_defs_cli;
    shader_defs
        .entry("AVAILABLE_STORAGE_BUFFER_BINDINGS".to_string())
        .or_insert_with(|| ShaderDefValue::UInt(8));
    shader_defs
        .entry("MAX_DIRECTIONAL_LIGHTS".to_string())
        .or_insert_with(|| ShaderDefValue::UInt(4));
    // Common Bevy defaults to help the composer expand interpolation tokens (#{...})
    shader_defs
        .entry("MAX_CASCADES_PER_LIGHT".to_string())
        .or_insert_with(|| ShaderDefValue::UInt(4));
    shader_defs
        .entry("MATERIAL_BIND_GROUP".to_string())
        .or_insert_with(|| ShaderDefValue::UInt(2));
    shader_defs
        .entry("PER_OBJECT_BUFFER_BATCH_SIZE".to_string())
        .or_insert_with(|| ShaderDefValue::UInt(256));
    shader_defs
        .entry("MAX_MORPH_WEIGHTS".to_string())
        .or_insert_with(|| ShaderDefValue::UInt(256));
    shader_defs
        .entry("MAX_VIEW_REFLECTION_PROBES".to_string())
        .or_insert_with(|| ShaderDefValue::UInt(8));

    // Try to make a naga module; if it fails, write diagnostics to disk to aid debugging.
    let naga_module_result = composer.make_naga_module(naga_oil::compose::NagaModuleDescriptor {
        shader_defs: shader_defs.clone(),
        ..(&root_shader).into()
    });

    let naga_module = match naga_module_result {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Composer error: {e}");

            // On error, dump entry shader and added modules
            let dump_dir = Path::new("tools/wgsl_to_spv/module_dumps");
            let _ = fs::create_dir_all(dump_dir);

            let _ = fs::write(dump_dir.join("entry.wgsl"), root_shader.source.as_str());

            for p in &added_module_paths {
                if let Ok(text) = fs::read_to_string(p) {
                    let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("module");
                    let _ = fs::write(dump_dir.join(name), text);
                }
            }

            // Attempt fallback composed WGSL using a simple inliner and write it for debugging
            if let Ok(composed) = compose_wgsl_simple(root_shader.source.as_str(), &module_index) {
                let _ = fs::write("tools/wgsl_to_spv/composed_mesh_preprocess.wgsl", &composed);
                eprintln!("Wrote fallback composed WGSL to tools/wgsl_to_spv/composed_mesh_preprocess.wgsl");
            }

            return Err(Box::new(e));
        }
    };

    // Validate and write SPIR-V
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let info = validator.validate(&naga_module)?;

    let options = spv::Options {
        lang_version: (1, 3),
        flags: WriterFlags::empty(),
        ..Default::default()
    };

    let mut writer = spv::Writer::new(&options)?;
    let mut words = Vec::new();

    writer.write(&naga_module, &info, None, &None, &mut words)?;

    let mut bytes = Vec::with_capacity(words.len() * 4);

    for w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }

    fs::write(&out, &bytes)?;
    println!("Wrote SPIR-V to {out}");
    Ok(())
}

fn build_module_index() -> Result<HashMap<String, PathBuf>, Box<dyn Error>> {
    let mut index = HashMap::new();

    for root in ["crates", "assets"] {
        for entry in WalkDir::new(root)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|entry| entry.path().is_file())
        {
            let path = entry.path();

            if path.extension().and_then(|ext| ext.to_str()) != Some("wgsl") {
                continue;
            }

            let contents = fs::read_to_string(path)?;

            for line in contents.lines() {
                if let Some(module_name) = line.trim_start().strip_prefix("#define_import_path ") {
                    let module_name = module_name.trim().to_string();
                    index
                        .entry(module_name)
                        .or_insert_with(|| path.to_path_buf());
                }
            }
        }
    }
    Ok(index)
}

fn add_custom_module(
    composer: &mut Composer,
    module_name: &str,
    module_index: &HashMap<String, PathBuf>,
    workspace_root: &Path,
    visited_custom_modules: &mut HashSet<String>,
    visited_asset_paths: &mut HashSet<PathBuf>,
    added_module_paths: &mut Vec<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    if !visited_custom_modules.insert(module_name.to_string()) {
        return Ok(());
    }

    if composer.contains_module(module_name) {
        return Ok(());
    }

    let module_path = module_index
        .get(module_name)
        .ok_or_else(|| format!("could not find module {module_name}"))?;
    let source = fs::read_to_string(module_path)?;
    let shader = Shader::from_wgsl(source, module_path.to_string_lossy().to_string());

    for import in shader.imports() {
        match import {
            ShaderImport::Custom(name) => {
                add_custom_module(
                    composer,
                    name.trim(),
                    module_index,
                    workspace_root,
                    visited_custom_modules,
                    visited_asset_paths,
                    added_module_paths,
                )?;
            }
            ShaderImport::AssetPath(path) => {
                add_asset_module(
                    composer,
                    path.trim(),
                    module_index,
                    workspace_root,
                    visited_custom_modules,
                    visited_asset_paths,
                    added_module_paths,
                )?;
            }
        }
    }

    composer.add_composable_module((&shader).into())?;
    added_module_paths.push(module_path.to_path_buf());
    Ok(())
}

fn add_asset_module(
    composer: &mut Composer,
    asset_path: &str,
    module_index: &HashMap<String, PathBuf>,
    workspace_root: &Path,
    visited_custom_modules: &mut HashSet<String>,
    visited_asset_paths: &mut HashSet<PathBuf>,
    added_module_paths: &mut Vec<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    let resolved = resolve_asset_path(workspace_root, asset_path)?;
    let canonical = resolved.canonicalize()?;
    if !visited_asset_paths.insert(canonical.clone()) {
        return Ok(());
    }

    let source = fs::read_to_string(&canonical)?;
    let shader = Shader::from_wgsl(source, canonical.to_string_lossy().to_string());

    for import in shader.imports() {
        match import {
            ShaderImport::Custom(name) => {
                add_custom_module(
                    composer,
                    name.trim(),
                    module_index,
                    workspace_root,
                    visited_custom_modules,
                    visited_asset_paths,
                    added_module_paths,
                )?;
            }
            ShaderImport::AssetPath(path) => {
                add_asset_module(
                    composer,
                    path.trim(),
                    module_index,
                    workspace_root,
                    visited_custom_modules,
                    visited_asset_paths,
                    added_module_paths,
                )?;
            }
        }
    }

    composer.add_composable_module((&shader).into())?;
    added_module_paths.push(canonical);
    Ok(())
}

fn resolve_asset_path(root: &Path, asset_path: &str) -> Result<PathBuf, Box<dyn Error>> {
    let candidate = PathBuf::from(asset_path);
    if candidate.is_absolute() && candidate.exists() {
        return Ok(candidate);
    }

    let candidate = root.join(asset_path);
    if candidate.exists() {
        return Ok(candidate);
    }

    let candidate = root.join("assets").join(asset_path);
    if candidate.exists() {
        return Ok(candidate);
    }

    Err(format!("could not resolve asset import path {asset_path}").into())
}

/// Fallback inline function to replace `#import` tokens by inlining the
/// file found via `#define_import_path`. This is only intended as an aid for debugging;
/// Refer to the composer for correct usage.
fn compose_wgsl_simple(
    entry_text: &str,
    module_index: &HashMap<String, PathBuf>,
) -> Result<String, Box<dyn Error>> {
    fn inline(
        src: &str,
        out: &mut String,
        module_index: &HashMap<String, PathBuf>,
        visited: &mut HashSet<String>,
    ) -> Result<(), Box<dyn Error>> {
        for line in src.lines() {
            let trimmed = line.trim_start();
            if let Some(rest) = trimmed.strip_prefix("#import ") {
                let import_token = rest
                    .trim()
                    .trim_end_matches(';')
                    .split_whitespace()
                    .next()
                    .unwrap();
                if import_token.starts_with('"') {
                    let path = import_token.trim_matches('"');
                    let text = fs::read_to_string(path)
                        .or_else(|_| fs::read_to_string(Path::new("assets").join(path)))?;
                    inline(&text, out, module_index, visited)?;
                } else {
                    let token = import_token;
                    let module_name = if token.contains('{') {
                        token
                            .split('{')
                            .next()
                            .unwrap()
                            .trim_end_matches("::")
                            .to_string()
                    } else if token.contains("::") {
                        token.split("::").take(2).collect::<Vec<_>>().join("::")
                    } else {
                        token.to_string()
                    };
                    if visited.contains(&module_name) {
                        continue;
                    }
                    visited.insert(module_name.clone());
                    if let Some(p) = module_index.get(&module_name) {
                        let text = fs::read_to_string(p)?;
                        // remove #define_import_path lines
                        let filtered = text
                            .lines()
                            .filter(|l| !l.trim_start().starts_with("#define_import_path"))
                            .collect::<Vec<_>>()
                            .join("\n");
                        inline(&filtered, out, module_index, visited)?;
                    }
                }
            } else {
                out.push_str(line);
                out.push('\n');
            }
        }
        Ok(())
    }

    let mut out = String::new();
    let mut visited = HashSet::new();
    inline(entry_text, &mut out, module_index, &mut visited)?;
    Ok(out)
}
