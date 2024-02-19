use std::{
    fmt,
    fs::File,
    io::{self, Read, Write},
    path::Path,
    process::{Command, ExitStatus},
};
use tempfile::{NamedTempFile, TempPath};

pub struct TempAsmFile {
    path: TempPath,
}

pub struct TempBinFile {
    path: TempPath,
}

impl TempAsmFile {
    pub fn create(asm_code: impl fmt::Display) -> io::Result<Self> {
        let mut asm_file = NamedTempFile::new()?;
        write!(asm_file, "{}", asm_code)?;
        let asm_path = asm_file.into_temp_path();
        Ok(Self { path: asm_path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn assembled(&self) -> io::Result<TempBinFile> {
        TempBinFile::from_asm_file(self.path())
    }
}

impl TempBinFile {
    pub fn from_asm_file(asm_path: impl AsRef<Path>) -> io::Result<Self> {
        let bin_path = NamedTempFile::new()?.into_temp_path();
        let status = assemble_source_file(asm_path, &bin_path)?;
        if status.success() {
            Ok(Self { path: bin_path })
        } else {
            Err(io::Error::new(io::ErrorKind::Other, status.to_string()))
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn read(&self) -> io::Result<Vec<u8>> {
        let mut bin_file = File::open(self.path())?;
        let mut buf = Vec::new();
        bin_file.read_to_end(&mut buf)?;
        Ok(buf)
    }
}

pub fn no_error<T>(expr: Result<T, impl fmt::Display>) -> T {
    match expr {
        Ok(expr) => expr,
        Err(error) => panic!("Error: {}", error),
    }
}

pub fn assemble_program(program: impl fmt::Display) -> io::Result<Vec<u8>> {
    TempAsmFile::create(program)?.assembled()?.read()
}

fn assemble_source_file(
    asm_path: impl AsRef<Path>,
    bin_path: impl AsRef<Path>,
) -> io::Result<ExitStatus> {
    Command::new("nasm")
        .args(["-f", "bin"])
        .arg(asm_path.as_ref())
        .arg("-o")
        .arg(bin_path.as_ref())
        .status()
}

#[cfg(test)]
mod test {
    use super::*;
    use std::fs::read_to_string;

    #[test]
    fn should_create_asm_file() {
        let temp_asm_file = no_error(TempAsmFile::create(""));
        assert!(temp_asm_file.path().exists());
    }

    #[test]
    fn asm_file_should_contain_input_text() {
        let input_text = "some\ntext";
        let temp_asm_file = no_error(TempAsmFile::create(input_text));
        let file_text = no_error(read_to_string(temp_asm_file.path()));
        assert_eq!(&file_text, input_text);
    }

    #[test]
    fn should_delete_asm_file_when_dropped() {
        let temp_asm_file = no_error(TempAsmFile::create(""));
        let path = temp_asm_file.path.to_path_buf();
        drop(temp_asm_file);
        assert!(!path.exists());
    }

    #[test]
    fn should_assemble_empty_asm_file() {
        let temp_bin_file = no_error(TempAsmFile::create("").unwrap().assembled());
        assert!(temp_bin_file.path().exists());
    }

    #[test]
    fn should_delete_bin_file_when_dropped() {
        let temp_bin_file = no_error(TempAsmFile::create("").unwrap().assembled());
        let path = temp_bin_file.path.to_path_buf();
        drop(temp_bin_file);
        assert!(!path.exists());
    }

    #[test]
    fn should_read_empty_bin_file() {
        let temp_bin_file = no_error(TempAsmFile::create("").unwrap().assembled());
        assert!(temp_bin_file.read().unwrap().is_empty());
    }

    #[test]
    fn should_read_single_two_byte_instruction_bin_file() {
        let temp_bin_file = no_error(TempAsmFile::create("bits 16\nmov ax, bx"))
            .assembled()
            .unwrap();
        assert_eq!(temp_bin_file.read().unwrap().len(), 2);
    }

    #[test]
    fn should_assemble_instructions() {
        assert_eq!(
            no_error(assemble_program("bits 16\nmov ax, bx\nmov cx, dx")).len(),
            4
        );
    }
}
