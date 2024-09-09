use std::{collections::HashMap, fs};
use std::path::Path;

use anyhow::{anyhow, Result};
use tracing::{debug, info, warn};

// This is kind of overkill

const SHADER_DIR: &str = "shaders";
const COMPILED_DIR: &str = "compiled";

fn compile_shader(source: &str, filename: &str) -> Result<Vec<u8>> {
    use shaderc::{Compiler, ShaderKind};

    let shader_kind = if filename.contains("vert") {
        ShaderKind::Vertex
    } else if filename.contains("frag") {
        ShaderKind::Fragment
    } else if filename.contains("geom") {
        ShaderKind::Geometry
    } else {
        return Err(anyhow!("Unknown shader type for file: {}", filename));
    };

    let compiler = Compiler::new()
        .ok_or(anyhow!("Failed to create shader compiler"))?;

    let binary_result = compiler.compile_into_spirv(
        source, shader_kind, "shader.glsl", "main", None)?;
    
    if binary_result.get_num_warnings() > 0 {
        warn!("Shader compiler warnings: {}", binary_result.get_warning_messages());
    }

    Ok(binary_result.as_binary_u8().to_vec())
}

struct SourceFile {
    source: String,
    filename: String,
    hash: u64,
}

fn all_source_files() -> Result<Vec<SourceFile>> {
    let mut files = Vec::new();

    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(crate::RESOURCES_DIR)
        .join(SHADER_DIR);

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            continue;
        }

        let path = entry.path();
        let filename = path
            .file_name().unwrap()
            .to_str().unwrap()
            .to_string();
        let source = fs::read_to_string(&path)?;

        let hash = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            source.hash(&mut hasher);
            hasher.finish()
        };

        files.push(SourceFile { source, filename, hash });
    }

    Ok(files)
}

struct CompiledFile {
    binary: Vec<u8>,
    original_filename: String,
    hash: u64,
}

fn all_compiled_files() -> Result<Vec<CompiledFile>> {
    let mut files = Vec::new();

    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(crate::RESOURCES_DIR)
        .join(SHADER_DIR)
        .join(COMPILED_DIR);

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let filename = path.file_name().unwrap().to_str().unwrap().to_string();
        let binary = fs::read(&path)?;

        let mut dot_split = filename.split('.').collect::<Vec<_>>();
        dot_split.pop(); // remove the file extension

        let hash = dot_split
            .pop()
            .ok_or(anyhow!("No hash in compiled filename: {}", filename))?
            .parse()?;

        // all before the last dot
        let original_filename = dot_split.join(".");
        
        files.push(CompiledFile { binary, original_filename, hash });
    }

    Ok(files)
}

fn compile_shaders_if_needed(source_files: &Vec<SourceFile>, compiled_files: &mut Vec<CompiledFile>) -> Result<usize> {
    use std::fs;
    use std::path::Path;

    let mut recompiled = 0;

    for source_file in source_files {
        let compiled_file = compiled_files.iter()
            .find(|cf| cf.original_filename == source_file.filename);

        if let Some(compiled_file) = compiled_file {
            if compiled_file.hash == source_file.hash {
                continue;
            } else {
                let old_file_path = Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join(crate::RESOURCES_DIR)
                    .join(SHADER_DIR)
                    .join(COMPILED_DIR)
                    .join(&format!("{}.{}.spv", compiled_file.original_filename, compiled_file.hash));

                debug!("Removing old compiled shader: {}", old_file_path.display());
                fs::remove_file(&old_file_path)?;
            }
        }

        let binary = compile_shader(&source_file.source, &source_file.filename)?;

        let compiled_filename = format!("{}.{}.spv", source_file.filename, source_file.hash);
        let compiled_path = Path::new(crate::RESOURCES_DIR)
            .join(SHADER_DIR)
            .join(COMPILED_DIR)
            .join(&compiled_filename);

        fs::write(&compiled_path, &binary)?;

        let compiled_file = CompiledFile {
            binary,
            original_filename: source_file.filename.clone(),
            hash: source_file.hash,
        };

        compiled_files.retain(|cf| cf.original_filename != source_file.filename);
        compiled_files.push(compiled_file);

        info!("Recompiled shader: {}", source_file.filename);

        recompiled += 1;
    }

    Ok(recompiled)
}

// we may never need geometry shaders, but it's nice to have just in case
#[derive(Debug)]
pub struct ShaderData {
    pub vertex: Vec<u8>,
    pub geometry: Option<Vec<u8>>,
    pub fragment: Vec<u8>,
}

// This function could be optimized, but shader count is low so performance is not a concern
fn group_shaders(compiled_files: &Vec<CompiledFile>) -> Result<Vec<(String, ShaderData)>> {
    let mut shader_groups = Vec::new();

    let mut compiled_files = compiled_files.iter().collect::<Vec<_>>();
    compiled_files.sort_by(|a, b| a.original_filename.cmp(&b.original_filename));

    while let Some(file) = compiled_files.first() {
        let identifier = file.original_filename.split('.').next().unwrap();

        // find all shaders with the same identifier
        let shaders = compiled_files
            .iter()
            .take_while(|f| f.original_filename.starts_with(identifier))
            .collect::<Vec<_>>();

        let vertex = shaders.iter()
            .find(|f| f.original_filename.contains("vert"))
            .ok_or(anyhow!("No vertex shader found for identifier: {}", identifier))?;

        let fragment = shaders.iter()
            .find(|f| f.original_filename.contains("frag"))
            .ok_or(anyhow!("No fragment shader found for identifier: {}", identifier))?;

        let geometry = shaders.iter()
            .find(|f| f.original_filename.contains("geom"));

        let shader_data = ShaderData {
            vertex: vertex.binary.clone(),
            geometry: geometry.map(|g| g.binary.clone()),
            fragment: fragment.binary.clone(),
        };

        shader_groups.push((identifier.to_string(), shader_data));

        compiled_files = compiled_files[shaders.len()..].to_vec();
    }

    Ok(shader_groups)
}

pub fn load_shaders() -> Result<HashMap<String, ShaderData>> {
    // info!("Loading shaders...");
    // let source_files = all_source_files()?;
    // let mut compiled_files = all_compiled_files()?;

    // // If we're in debug mode, recompile shaders if needed
    // #[cfg(debug_assertions)]
    // {
    //     let recompiled_count = compile_shaders_if_needed(&source_files, &mut compiled_files)?;
    //     info!("{} shaders loaded ({} recompiled)", compiled_files.len(), recompiled_count);
    // }

    // let shader_map = group_shaders(&compiled_files)?
    //     .into_iter()
    //     .collect::<HashMap<_, _>>();

    // Ok(shader_map)

    Ok(HashMap::new())
}