use std::{
    cmp::Ordering,
    fmt,
    ops::{Shl, Shr},
};
use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum DecodeError<'a> {
    #[error("unknown opcode in instruction byte {byte:08b}")]
    UnknownOpcode { byte: u8 },
    #[error("unsupported mode bits in instruction byte {byte:08b}")]
    UnsupportedMode { byte: u8 },
    #[error("unsupported memory address expression bits in instruction byte {byte:08b}")]
    UnsupportedMemoryAddressExpression { byte: u8 },
    #[error("insufficient bytes for instruction: {bytes:?}")]
    MissingInstructionBytes { bytes: &'a [u8] },
    #[error("insufficient bytes for memory address displacement: {bytes:?}")]
    MissingDisplacementBytes { bytes: &'a [u8] },
    #[error("invalid bits in instruction: {bytes:?}")]
    InvalidInstruction { bytes: &'a [u8] },
}

#[derive(Error, Debug, PartialEq, Eq)]
#[error("invalid instruction at IP={ip}: {error}")]
pub struct ProgramDecodeError<'a> {
    error: DecodeError<'a>,
    ip: usize,
}

pub type DecodeResult<'a, T> = Result<T, DecodeError<'a>>;

pub struct Program {
    instructions: Vec<Instruction>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instruction {
    op: Operation,
    source: Source,
    dest: Destination,
    size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    Mov,
    Add,
    Adc,
    Sub,
    Sbb,
    Cmp,
    And,
    Or,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    None,
    Immediate(Immediate),
    Register(Register),
    Memory(MemoryAddressExpression),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Destination {
    None,
    Register(Register),
    Memory(MemoryAddressExpression),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Register {
    Full(WordRegister),
    Half(ByteRegister),
    Segment(SegmentRegister),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WordRegister {
    /// Primary accumulator.
    AX,
    /// Base, accumulator.
    BX,
    /// Counter, accumulator.
    CX,
    /// Accumulater, extended accumulator.
    DX,
    /// Stack pointer.
    SP,
    /// Base pointer.
    BP,
    /// Source index.
    SI,
    /// Destination index.
    DI,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteRegister {
    /// Lower half of AX.
    AL,
    /// Upper half of AX.
    AH,
    /// Lower half of BX.
    BL,
    /// Upper half of BX.
    BH,
    /// Lower half of CX.
    CL,
    /// Upper half of CX.
    CH,
    /// Lower half of DX.
    DL,
    /// Upper half of DX.
    DH,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentRegister {
    ES,
    CS,
    SS,
    DS,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryAddressExpression {
    Direct(u16),
    BXPlusSI,
    BXPlusDI,
    BPPlusSI,
    BPPlusDI,
    SI,
    DI,
    BX,
    BXPlusSIPlus(u16),
    BXPlusDIPlus(u16),
    BPPlusSIPlus(u16),
    BPPlusDIPlus(u16),
    SIPlus(u16),
    DIPlus(u16),
    BPPlus(u16),
    BXPlus(u16),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Immediate {
    Word(u16),
    Byte(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Opcode {
    MovRegMemToFromReg,
    MovMemToFromAccumReg,
    MovImmToRegMem,
    MovImmToReg,
    MovRegMemToFromSegReg,
    ArithmeticImmWithMemReg,
    ArithmeticRegMemWithReg(ArithmeticOpcode),
    ArithmeticImmWithAccumReg(ArithmeticOpcode),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArithmeticOpcode {
    Add,
    Adc,
    Sub,
    Sbb,
    Cmp,
    And,
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    SourceInRegField,
    DestInRegField,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SignBit {
    Set,
    NotSet,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OperandSize {
    Word,
    Byte,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Register,
    Memory(MemoryMode),
}

#[allow(clippy::enum_variant_names)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MemoryMode {
    NoDisplacement,
    ByteDisplacement,
    WordDisplacement,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DecodedRegisterOrMemory {
    reg_or_mem: RegisterOrMemory,
    disp_size: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum RegisterOrMemory {
    Register(Register),
    Memory(MemoryAddressExpression),
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "bits 16{}",
            self.instructions
                .iter()
                .fold(String::new(), |acc, instr| format!("{}\n{}", acc, instr)),
        )
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self {
                op,
                source: Source::None,
                dest: Destination::None,
                ..
            } => write!(f, "{}", op),
            Self {
                op,
                source: Source::None,
                dest,
                ..
            } => write!(f, "{} {}", op, dest),
            Self {
                op,
                source,
                dest: Destination::None,
                ..
            } => write!(f, "{} {}", op, source),
            Self {
                op,
                source: Source::Immediate(imm),
                dest: Destination::Memory(mem_addr_expr),
                ..
            } => write!(
                f,
                "{} {}, {} {}",
                op,
                mem_addr_expr,
                match imm {
                    Immediate::Word(_) => "word",
                    Immediate::Byte(_) => "byte",
                },
                imm
            ),
            Self {
                op, source, dest, ..
            } => write!(f, "{} {}, {}", op, dest, source),
        }
    }
}

impl From<Register> for Source {
    fn from(reg: Register) -> Self {
        Self::Register(reg)
    }
}

impl From<WordRegister> for Source {
    fn from(reg: WordRegister) -> Self {
        Register::Full(reg).into()
    }
}

impl From<ByteRegister> for Source {
    fn from(reg: ByteRegister) -> Self {
        Register::Half(reg).into()
    }
}

impl From<SegmentRegister> for Source {
    fn from(reg: SegmentRegister) -> Self {
        Register::Segment(reg).into()
    }
}

impl From<MemoryAddressExpression> for Source {
    fn from(mem_addr_expr: MemoryAddressExpression) -> Self {
        Self::Memory(mem_addr_expr)
    }
}

impl From<RegisterOrMemory> for Source {
    fn from(reg_or_mem: RegisterOrMemory) -> Self {
        match reg_or_mem {
            RegisterOrMemory::Register(reg) => reg.into(),
            RegisterOrMemory::Memory(mem_addr_expr) => mem_addr_expr.into(),
        }
    }
}

impl From<Immediate> for Source {
    fn from(imm: Immediate) -> Self {
        Self::Immediate(imm)
    }
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => Ok(()),
            Self::Immediate(imm) => imm.fmt(f),
            Self::Register(reg) => reg.fmt(f),
            Self::Memory(mem_addr_expr) => mem_addr_expr.fmt(f),
        }
    }
}

impl From<Register> for Destination {
    fn from(reg: Register) -> Self {
        Self::Register(reg)
    }
}

impl From<WordRegister> for Destination {
    fn from(reg: WordRegister) -> Self {
        Register::Full(reg).into()
    }
}

impl From<ByteRegister> for Destination {
    fn from(reg: ByteRegister) -> Self {
        Register::Half(reg).into()
    }
}

impl From<SegmentRegister> for Destination {
    fn from(reg: SegmentRegister) -> Self {
        Register::Segment(reg).into()
    }
}

impl From<MemoryAddressExpression> for Destination {
    fn from(mem_addr_expr: MemoryAddressExpression) -> Self {
        Self::Memory(mem_addr_expr)
    }
}

impl From<RegisterOrMemory> for Destination {
    fn from(reg_or_mem: RegisterOrMemory) -> Self {
        match reg_or_mem {
            RegisterOrMemory::Register(reg) => reg.into(),
            RegisterOrMemory::Memory(mem_addr_expr) => mem_addr_expr.into(),
        }
    }
}

impl fmt::Display for Destination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => Ok(()),
            Self::Register(reg) => reg.fmt(f),
            Self::Memory(mem_addr_expr) => mem_addr_expr.fmt(f),
        }
    }
}

impl From<ArithmeticOpcode> for Operation {
    fn from(opcode: ArithmeticOpcode) -> Self {
        match opcode {
            ArithmeticOpcode::Add => Self::Add,
            ArithmeticOpcode::Adc => Self::Adc,
            ArithmeticOpcode::Sub => Self::Sub,
            ArithmeticOpcode::Sbb => Self::Sbb,
            ArithmeticOpcode::Cmp => Self::Cmp,
            ArithmeticOpcode::And => Self::And,
            ArithmeticOpcode::Or => Self::Or,
        }
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Mov => "mov",
                Self::Add => "add",
                Self::Adc => "adc",
                Self::Sub => "sub",
                Self::Sbb => "sbb",
                Self::Cmp => "cmp",
                Self::And => "and",
                Self::Or => "or",
            }
        )
    }
}

impl Register {
    fn main_accumulator(operand_size: OperandSize) -> Self {
        match operand_size {
            OperandSize::Word => Self::Full(WordRegister::AX),
            OperandSize::Byte => Self::Half(ByteRegister::AL),
        }
    }
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full(reg) => reg.fmt(f),
            Self::Half(reg) => reg.fmt(f),
            Self::Segment(reg) => reg.fmt(f),
        }
    }
}

impl fmt::Display for WordRegister {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::AX => "ax",
                Self::BX => "bx",
                Self::CX => "cx",
                Self::DX => "dx",
                Self::SP => "sp",
                Self::BP => "bp",
                Self::SI => "si",
                Self::DI => "di",
            },
        )
    }
}

impl fmt::Display for ByteRegister {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::AL => "al",
                Self::AH => "ah",
                Self::BL => "bl",
                Self::BH => "bh",
                Self::CL => "cl",
                Self::CH => "ch",
                Self::DL => "dl",
                Self::DH => "dh",
            },
        )
    }
}

impl fmt::Display for SegmentRegister {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::ES => "es",
                Self::CS => "cs",
                Self::SS => "ss",
                Self::DS => "ds",
            },
        )
    }
}

impl MemoryAddressExpression {
    fn displacement_size(&self, mode: MemoryMode) -> usize {
        match mode {
            MemoryMode::NoDisplacement => {
                if let Self::Direct(_) = self {
                    2
                } else {
                    0
                }
            }
            MemoryMode::ByteDisplacement => 1,
            MemoryMode::WordDisplacement => 2,
        }
    }
}

impl fmt::Display for MemoryAddressExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn fmt_disp(disp: &u16) -> impl fmt::Display {
            // Use signed interpretation for formatting
            let disp = *disp as i16;
            match disp.cmp(&0) {
                Ordering::Greater => format!(" + {}", disp),
                // Cast to 32-bit before negating to avoid overflow edge case
                Ordering::Less => format!(" - {}", -(disp as i32)),
                Ordering::Equal => String::new(),
            }
        }

        match self {
            Self::Direct(addr) => write!(f, "[{}]", addr),
            Self::BXPlusSI => write!(f, "[bx + si]"),
            Self::BXPlusDI => write!(f, "[bx + di]"),
            Self::BPPlusSI => write!(f, "[bp + si]"),
            Self::BPPlusDI => write!(f, "[bp + di]"),
            Self::SI => write!(f, "[si]"),
            Self::DI => write!(f, "[di]"),
            Self::BX => write!(f, "[bx]"),
            Self::BXPlusSIPlus(disp) => write!(f, "[bx + si{}]", fmt_disp(disp)),
            Self::BXPlusDIPlus(disp) => write!(f, "[bx + di{}]", fmt_disp(disp)),
            Self::BPPlusSIPlus(disp) => write!(f, "[bp + si{}]", fmt_disp(disp)),
            Self::BPPlusDIPlus(disp) => write!(f, "[bp + di{}]", fmt_disp(disp)),
            Self::SIPlus(disp) => write!(f, "[si{}]", fmt_disp(disp)),
            Self::DIPlus(disp) => write!(f, "[di{}]", fmt_disp(disp)),
            Self::BPPlus(disp) => write!(f, "[bp{}]", fmt_disp(disp)),
            Self::BXPlus(disp) => write!(f, "[bx{}]", fmt_disp(disp)),
        }
    }
}

impl fmt::Display for Immediate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Word(imm) => write!(f, "{}", imm),
            Self::Byte(imm) => write!(f, "{}", imm),
        }
    }
}

impl DecodedRegisterOrMemory {
    fn register(reg: Register) -> Self {
        Self {
            reg_or_mem: RegisterOrMemory::Register(reg),
            disp_size: 0,
        }
    }

    fn memory(mem_addr_expr: MemoryAddressExpression, mode: MemoryMode) -> Self {
        Self {
            reg_or_mem: RegisterOrMemory::Memory(mem_addr_expr),
            disp_size: mem_addr_expr.displacement_size(mode),
        }
    }
}

pub fn decode_program(bytes: &[u8]) -> Result<Program, ProgramDecodeError> {
    let mut instructions = Vec::with_capacity(bytes.len() / 2);
    let mut ip = 0;
    while ip < bytes.len() {
        let instr =
            decode_instruction(&bytes[ip..]).map_err(|error| ProgramDecodeError { error, ip })?;
        ip += instr.size;
        instructions.push(instr);
    }
    Ok(Program { instructions })
}

pub fn decode_instruction(bytes: &[u8]) -> DecodeResult<Instruction> {
    if bytes.is_empty() {
        return Err(DecodeError::MissingInstructionBytes { bytes });
    }
    let opcode = decode_opcode(bytes[0])?;
    match opcode {
        Opcode::MovRegMemToFromReg => {
            decode_instr_for_reg_mem_to_from_reg_opcode(Operation::Mov, bytes)
        }
        Opcode::MovMemToFromAccumReg => {
            decode_instr_for_mem_to_from_accum_reg_opcode(Operation::Mov, bytes)
        }
        Opcode::MovImmToRegMem => {
            if bytes[1] & 0b00111000 != 0 {
                return Err(DecodeError::InvalidInstruction { bytes: &bytes[..2] });
            }
            decode_instr_for_imm_to_reg_mem_opcode(Operation::Mov, bytes)
        }
        Opcode::MovImmToReg => decode_instr_for_imm_to_reg_opcode(Operation::Mov, bytes),
        Opcode::MovRegMemToFromSegReg => {
            if bytes[1] & 0b00100000 != 0 {
                return Err(DecodeError::InvalidInstruction { bytes: &bytes[..2] });
            }
            decode_instr_for_reg_mem_to_from_seg_reg_opcode(Operation::Mov, bytes)
        }
        Opcode::ArithmeticRegMemWithReg(opcode) => {
            decode_instr_for_reg_mem_to_from_reg_opcode(opcode.into(), bytes)
        }
        Opcode::ArithmeticImmWithMemReg => {
            decode_instr_for_arithmetic_imm_with_mem_reg_opcode(bytes)
        }
        Opcode::ArithmeticImmWithAccumReg(opcode) => {
            decode_instr_for_arithmetic_imm_with_accum_reg_opcode(opcode, bytes)
        }
    }
}

fn decode_instr_for_reg_mem_to_from_reg_opcode(
    op: Operation,
    bytes: &[u8],
) -> DecodeResult<Instruction> {
    if bytes.len() < 2 {
        return Err(DecodeError::MissingInstructionBytes { bytes });
    }
    let direction = decode_direction(bytes[0]);
    let operand_size = decode_operand_size(bytes[0]);
    let reg = decode_first_register(operand_size, bytes[1]);
    let DecodedRegisterOrMemory {
        reg_or_mem,
        disp_size,
    } = decode_register_or_memory(operand_size, &bytes[1..])?;
    let (source, dest) = assign_reg_and_reg_mem_to_source_and_dest(direction, reg, reg_or_mem);
    Ok(Instruction {
        op,
        source,
        dest,
        size: 2 + disp_size,
    })
}

fn decode_instr_for_mem_to_from_accum_reg_opcode(
    op: Operation,
    bytes: &[u8],
) -> DecodeResult<Instruction> {
    if bytes.len() < 3 {
        return Err(DecodeError::MissingInstructionBytes { bytes });
    }
    let direction = decode_direction(bytes[0]);
    let operand_size = decode_operand_size(bytes[0]);
    let reg = Register::main_accumulator(operand_size);
    let mem_addr = decode_address(&bytes[1..])?;
    let (source, dest) = match direction {
        Direction::SourceInRegField => (mem_addr.into(), reg.into()), // d = 0
        Direction::DestInRegField => (reg.into(), mem_addr.into()),   // d = 1
    };
    Ok(Instruction {
        op,
        source,
        dest,
        size: 3,
    })
}

fn decode_instr_for_imm_to_reg_mem_opcode(
    op: Operation,
    bytes: &[u8],
) -> DecodeResult<Instruction> {
    if bytes.len() < 3 {
        return Err(DecodeError::MissingInstructionBytes { bytes });
    }
    let operand_size = decode_operand_size(bytes[0]);
    let DecodedRegisterOrMemory {
        reg_or_mem,
        disp_size,
    } = decode_register_or_memory(operand_size, &bytes[1..])?;
    let imm = decode_immediate(operand_size, &bytes[2 + disp_size..])?;
    let size = 2
        + disp_size
        + match operand_size {
            OperandSize::Word => 2,
            OperandSize::Byte => 1,
        };
    Ok(Instruction {
        op,
        source: imm.into(),
        dest: reg_or_mem.into(),
        size,
    })
}

fn decode_instr_for_imm_to_reg_opcode(op: Operation, bytes: &[u8]) -> DecodeResult<Instruction> {
    fn decode_imm_to_reg_operand_size(byte1: u8) -> OperandSize {
        if byte1 & 0b00001000 == 0b00000000 {
            OperandSize::Byte
        } else {
            OperandSize::Word
        }
    }
    if bytes.len() < 2 {
        return Err(DecodeError::MissingInstructionBytes { bytes });
    }
    let operand_size = decode_imm_to_reg_operand_size(bytes[0]);
    let reg = decode_last_register(operand_size, bytes[0]);
    let imm = decode_immediate(operand_size, &bytes[1..])?;
    let size = match operand_size {
        OperandSize::Word => 3,
        OperandSize::Byte => 2,
    };
    Ok(Instruction {
        op,
        source: imm.into(),
        dest: reg.into(),
        size,
    })
}

fn decode_instr_for_reg_mem_to_from_seg_reg_opcode(
    op: Operation,
    bytes: &[u8],
) -> DecodeResult<Instruction> {
    if bytes.len() < 2 {
        return Err(DecodeError::MissingInstructionBytes { bytes });
    }
    let direction = decode_direction(bytes[0]);
    let DecodedRegisterOrMemory {
        reg_or_mem,
        disp_size,
    } = decode_register_or_memory(OperandSize::Word, &bytes[1..])?;
    let seg_reg = Register::Segment(decode_segment_register(bytes[1]));
    let (source, dest) = assign_reg_and_reg_mem_to_source_and_dest(direction, seg_reg, reg_or_mem);
    Ok(Instruction {
        op,
        source,
        dest,
        size: 2 + disp_size,
    })
}

fn decode_instr_for_arithmetic_imm_with_mem_reg_opcode(bytes: &[u8]) -> DecodeResult<Instruction> {
    if bytes.len() < 3 {
        return Err(DecodeError::MissingInstructionBytes { bytes });
    }
    let sign_bit = decode_sign_bit(bytes[0]);
    let operand_size = decode_operand_size(bytes[0]);
    let DecodedRegisterOrMemory {
        reg_or_mem,
        disp_size,
    } = decode_register_or_memory(operand_size, &bytes[1..])?;
    let imm = decode_immediate_with_sign_bit(sign_bit, operand_size, &bytes[2 + disp_size..])?;
    let opcode = decode_arithmetic_opcode(bytes[1])?;
    let size = match (sign_bit, operand_size) {
        (SignBit::NotSet, OperandSize::Word) => 2 + disp_size + 2,
        _ => 2 + disp_size + 1,
    };
    Ok(Instruction {
        op: opcode.into(),
        source: imm.into(),
        dest: reg_or_mem.into(),
        size,
    })
}

fn decode_instr_for_arithmetic_imm_with_accum_reg_opcode(
    opcode: ArithmeticOpcode,
    bytes: &[u8],
) -> DecodeResult<Instruction> {
    if bytes.len() < 2 {
        return Err(DecodeError::MissingInstructionBytes { bytes });
    }
    let operand_size = decode_operand_size(bytes[0]);
    let imm = decode_immediate(operand_size, &bytes[1..])?;
    let reg = Register::main_accumulator(operand_size);
    let size = match operand_size {
        OperandSize::Word => 3,
        OperandSize::Byte => 2,
    };
    Ok(Instruction {
        op: opcode.into(),
        source: imm.into(),
        dest: reg.into(),
        size,
    })
}

fn decode_register_or_memory(
    operand_size: OperandSize,
    bytes: &[u8],
) -> DecodeResult<DecodedRegisterOrMemory> {
    if bytes.is_empty() {
        return Err(DecodeError::MissingInstructionBytes { bytes });
    }
    let mode = decode_mode(bytes[0]);
    Ok(match mode {
        Mode::Register => {
            DecodedRegisterOrMemory::register(decode_last_register(operand_size, bytes[0]))
        }
        Mode::Memory(mem_mode) => DecodedRegisterOrMemory::memory(
            decode_memory_address_expression(mem_mode, bytes[0], &bytes[1..])?,
            mem_mode,
        ),
    })
}

fn assign_reg_and_reg_mem_to_source_and_dest(
    direction: Direction,
    reg: Register,
    reg_or_mem: RegisterOrMemory,
) -> (Source, Destination) {
    match direction {
        Direction::SourceInRegField => (reg.into(), reg_or_mem.into()),
        Direction::DestInRegField => (reg_or_mem.into(), reg.into()),
    }
}

fn decode_opcode(byte1: u8) -> DecodeResult<'static, Opcode> {
    match byte1 & 0b11111110 {
        0b11000110 => Ok(Opcode::MovImmToRegMem),
        _ => match byte1 & 0b11111101 {
            0b10001100 => Ok(Opcode::MovRegMemToFromSegReg),
            _ => match byte1 & 0b11111100 {
                0b10001000 => Ok(Opcode::MovRegMemToFromReg),
                0b10100000 => Ok(Opcode::MovMemToFromAccumReg),
                0b10000000 => Ok(Opcode::ArithmeticImmWithMemReg),
                _ => match byte1 & 0b11000110 {
                    0b00000100 => Ok(Opcode::ArithmeticImmWithAccumReg(decode_arithmetic_opcode(
                        byte1,
                    )?)),
                    _ => match byte1 & 0b11000100 {
                        0b00000000 => Ok(Opcode::ArithmeticRegMemWithReg(
                            decode_arithmetic_opcode(byte1)?,
                        )),
                        _ => match byte1 & 0b11110000 {
                            0b10110000 => Ok(Opcode::MovImmToReg),
                            _ => Err(DecodeError::UnknownOpcode { byte: byte1 }),
                        },
                    },
                },
            },
        },
    }
}

fn decode_arithmetic_opcode(byte: u8) -> DecodeResult<'static, ArithmeticOpcode> {
    match byte & 0b00111000 {
        0b00000000 => Ok(ArithmeticOpcode::Add),
        0b00010000 => Ok(ArithmeticOpcode::Adc),
        0b00101000 => Ok(ArithmeticOpcode::Sub),
        0b00011000 => Ok(ArithmeticOpcode::Sbb),
        0b00111000 => Ok(ArithmeticOpcode::Cmp),
        0b00100000 => Ok(ArithmeticOpcode::And),
        0b00001000 => Ok(ArithmeticOpcode::Or),
        _ => Err(DecodeError::UnknownOpcode { byte }),
    }
}

fn decode_direction(byte1: u8) -> Direction {
    if byte1 & 0b00000010 == 0b00000000 {
        Direction::SourceInRegField
    } else {
        Direction::DestInRegField
    }
}

fn decode_sign_bit(byte1: u8) -> SignBit {
    if byte1 & 0b00000010 == 0b00000000 {
        SignBit::NotSet
    } else {
        SignBit::Set
    }
}

fn decode_operand_size(byte1: u8) -> OperandSize {
    if byte1 & 0b00000001 == 0b00000000 {
        OperandSize::Byte
    } else {
        OperandSize::Word
    }
}

fn decode_mode(byte2: u8) -> Mode {
    match byte2 & 0b11000000 {
        0b11000000 => Mode::Register,
        0b00000000 => Mode::Memory(MemoryMode::NoDisplacement),
        0b01000000 => Mode::Memory(MemoryMode::ByteDisplacement),
        0b10000000 => Mode::Memory(MemoryMode::WordDisplacement),
        _ => unreachable!(),
    }
}

fn decode_first_register(operand_size: OperandSize, byte: u8) -> Register {
    match operand_size {
        OperandSize::Word => Register::Full(decode_first_word_register(byte)),
        OperandSize::Byte => Register::Half(decode_first_byte_register(byte)),
    }
}

fn decode_last_register(operand_size: OperandSize, byte: u8) -> Register {
    match operand_size {
        OperandSize::Word => Register::Full(decode_last_word_register(byte)),
        OperandSize::Byte => Register::Half(decode_last_byte_register(byte)),
    }
}

fn decode_first_word_register(byte2: u8) -> WordRegister {
    decode_word_register(shift_mask_register_byte(byte2, 3))
}

fn decode_last_word_register(byte2: u8) -> WordRegister {
    decode_word_register(shift_mask_register_byte(byte2, 0))
}

fn decode_first_byte_register(byte2: u8) -> ByteRegister {
    decode_byte_register(shift_mask_register_byte(byte2, 3))
}

fn decode_last_byte_register(byte2: u8) -> ByteRegister {
    decode_byte_register(shift_mask_register_byte(byte2, 0))
}

fn decode_segment_register(byte2: u8) -> SegmentRegister {
    match byte2 & 0b00011000 {
        0b00000000 => SegmentRegister::ES,
        0b00001000 => SegmentRegister::CS,
        0b00010000 => SegmentRegister::SS,
        0b00011000 => SegmentRegister::DS,
        _ => unreachable!(),
    }
}

fn shift_mask_register_byte(byte: u8, rshift: usize) -> u8 {
    byte.shr(rshift) & 0b00000111
}

fn decode_word_register(masked_shifted_byte: u8) -> WordRegister {
    match masked_shifted_byte {
        0b00000000 => WordRegister::AX,
        0b00000001 => WordRegister::CX,
        0b00000010 => WordRegister::DX,
        0b00000011 => WordRegister::BX,
        0b00000100 => WordRegister::SP,
        0b00000101 => WordRegister::BP,
        0b00000110 => WordRegister::SI,
        0b00000111 => WordRegister::DI,
        _ => unreachable!(),
    }
}

fn decode_byte_register(masked_shifted_byte: u8) -> ByteRegister {
    match masked_shifted_byte {
        0b00000000 => ByteRegister::AL,
        0b00000001 => ByteRegister::CL,
        0b00000010 => ByteRegister::DL,
        0b00000011 => ByteRegister::BL,
        0b00000100 => ByteRegister::AH,
        0b00000101 => ByteRegister::CH,
        0b00000110 => ByteRegister::DH,
        0b00000111 => ByteRegister::BH,
        _ => unreachable!(),
    }
}

fn decode_memory_address_expression(
    mode: MemoryMode,
    byte2: u8,
    disp_bytes: &[u8],
) -> DecodeResult<MemoryAddressExpression> {
    fn decode_decode_memory_address_expression_with_disp(
        masked_byte: u8,
        disp: u16,
    ) -> MemoryAddressExpression {
        match masked_byte {
            0b00000000 => MemoryAddressExpression::BXPlusSIPlus(disp),
            0b00000001 => MemoryAddressExpression::BXPlusDIPlus(disp),
            0b00000010 => MemoryAddressExpression::BPPlusSIPlus(disp),
            0b00000011 => MemoryAddressExpression::BPPlusDIPlus(disp),
            0b00000100 => MemoryAddressExpression::SIPlus(disp),
            0b00000101 => MemoryAddressExpression::DIPlus(disp),
            0b00000110 => MemoryAddressExpression::BPPlus(disp),
            0b00000111 => MemoryAddressExpression::BXPlus(disp),
            _ => unreachable!(),
        }
    }

    let masked_byte = byte2 & 0b00000111;
    Ok(match mode {
        MemoryMode::NoDisplacement => match masked_byte {
            0b00000110 => decode_address(disp_bytes)?,
            0b00000000 => MemoryAddressExpression::BXPlusSI,
            0b00000001 => MemoryAddressExpression::BXPlusDI,
            0b00000010 => MemoryAddressExpression::BPPlusSI,
            0b00000011 => MemoryAddressExpression::BPPlusDI,
            0b00000100 => MemoryAddressExpression::SI,
            0b00000101 => MemoryAddressExpression::DI,
            0b00000111 => MemoryAddressExpression::BX,
            _ => unreachable!(),
        },
        MemoryMode::ByteDisplacement => {
            if disp_bytes.is_empty() {
                return Err(DecodeError::MissingDisplacementBytes { bytes: disp_bytes });
            }
            let disp = sign_extend_to_16_bit(disp_bytes[0]);
            decode_decode_memory_address_expression_with_disp(masked_byte, disp)
        }
        MemoryMode::WordDisplacement => {
            if disp_bytes.len() < 2 {
                return Err(DecodeError::MissingDisplacementBytes { bytes: disp_bytes });
            }
            // Use first byte as low byte and second as high byte
            let disp = combine_low_and_high_bytes(disp_bytes[0], disp_bytes[1]);
            decode_decode_memory_address_expression_with_disp(masked_byte, disp)
        }
    })
}

fn decode_address(bytes: &[u8]) -> DecodeResult<MemoryAddressExpression> {
    if bytes.len() < 2 {
        return Err(DecodeError::MissingInstructionBytes { bytes });
    }
    let addr = combine_low_and_high_bytes(bytes[0], bytes[1]);
    Ok(MemoryAddressExpression::Direct(addr))
}

fn decode_immediate(operand_size: OperandSize, bytes: &[u8]) -> DecodeResult<Immediate> {
    decode_immediate_with_sign_bit(SignBit::NotSet, operand_size, bytes)
}

fn decode_immediate_with_sign_bit(
    sign_bit: SignBit,
    operand_size: OperandSize,
    bytes: &[u8],
) -> DecodeResult<Immediate> {
    Ok(match (sign_bit, operand_size) {
        (SignBit::NotSet, OperandSize::Word) => {
            if bytes.len() < 2 {
                return Err(DecodeError::MissingInstructionBytes { bytes });
            }
            Immediate::Word(combine_low_and_high_bytes(bytes[0], bytes[1]))
        }
        (SignBit::Set, OperandSize::Word) => {
            if bytes.is_empty() {
                return Err(DecodeError::MissingInstructionBytes { bytes });
            }
            Immediate::Word(sign_extend_to_16_bit(bytes[0]))
        }
        (_, OperandSize::Byte) => {
            if bytes.is_empty() {
                return Err(DecodeError::MissingInstructionBytes { bytes });
            }
            Immediate::Byte(bytes[0])
        }
    })
}

fn sign_extend_to_16_bit(byte: u8) -> u16 {
    ((byte as u16).shl(8) as i16).shr(8) as u16
}

fn combine_low_and_high_bytes(low_byte: u8, high_byte: u8) -> u16 {
    low_byte as u16 + (high_byte as u16).shl(8)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::testutil::no_error;

    #[test]
    fn should_decode_mov_opcodes() {
        assert_eq!(
            no_error(decode_opcode(0b10001000)),
            Opcode::MovRegMemToFromReg
        );
        assert_eq!(
            no_error(decode_opcode(0b10001001)),
            Opcode::MovRegMemToFromReg
        );
        assert_eq!(
            no_error(decode_opcode(0b10001010)),
            Opcode::MovRegMemToFromReg
        );
        assert_eq!(
            no_error(decode_opcode(0b10001011)),
            Opcode::MovRegMemToFromReg
        );
        assert_eq!(
            no_error(decode_opcode(0b10100000)),
            Opcode::MovMemToFromAccumReg
        );
        assert_eq!(
            no_error(decode_opcode(0b10100011)),
            Opcode::MovMemToFromAccumReg
        );
        assert_eq!(no_error(decode_opcode(0b11000110)), Opcode::MovImmToRegMem);
        assert_eq!(no_error(decode_opcode(0b11000111)), Opcode::MovImmToRegMem);
        assert_eq!(no_error(decode_opcode(0b10110000)), Opcode::MovImmToReg);
        assert_eq!(no_error(decode_opcode(0b10111010)), Opcode::MovImmToReg);
        assert_eq!(
            no_error(decode_opcode(0b10001100)),
            Opcode::MovRegMemToFromSegReg
        );
        assert_eq!(
            no_error(decode_opcode(0b10001110)),
            Opcode::MovRegMemToFromSegReg
        );
    }

    #[test]
    fn should_decode_arithmetic_opcodes() {
        assert_eq!(
            no_error(decode_opcode(0b10000000)),
            Opcode::ArithmeticImmWithMemReg
        );
        assert_eq!(
            no_error(decode_opcode(0b10000011)),
            Opcode::ArithmeticImmWithMemReg
        );

        assert_eq!(
            no_error(decode_opcode(0b00000000)),
            Opcode::ArithmeticRegMemWithReg(ArithmeticOpcode::Add)
        );
        assert_eq!(
            no_error(decode_opcode(0b00000011)),
            Opcode::ArithmeticRegMemWithReg(ArithmeticOpcode::Add)
        );
        assert_eq!(
            no_error(decode_opcode(0b00000100)),
            Opcode::ArithmeticImmWithAccumReg(ArithmeticOpcode::Add)
        );
        assert_eq!(
            no_error(decode_opcode(0b00000101)),
            Opcode::ArithmeticImmWithAccumReg(ArithmeticOpcode::Add)
        );

        assert_eq!(
            no_error(decode_opcode(0b00010000)),
            Opcode::ArithmeticRegMemWithReg(ArithmeticOpcode::Adc)
        );
        assert_eq!(
            no_error(decode_opcode(0b00010100)),
            Opcode::ArithmeticImmWithAccumReg(ArithmeticOpcode::Adc)
        );

        assert_eq!(
            no_error(decode_opcode(0b00101000)),
            Opcode::ArithmeticRegMemWithReg(ArithmeticOpcode::Sub)
        );
        assert_eq!(
            no_error(decode_opcode(0b00101100)),
            Opcode::ArithmeticImmWithAccumReg(ArithmeticOpcode::Sub)
        );

        assert_eq!(
            no_error(decode_opcode(0b00011000)),
            Opcode::ArithmeticRegMemWithReg(ArithmeticOpcode::Sbb)
        );
        assert_eq!(
            no_error(decode_opcode(0b00011100)),
            Opcode::ArithmeticImmWithAccumReg(ArithmeticOpcode::Sbb)
        );

        assert_eq!(
            no_error(decode_opcode(0b00111000)),
            Opcode::ArithmeticRegMemWithReg(ArithmeticOpcode::Cmp)
        );
        assert_eq!(
            no_error(decode_opcode(0b00111100)),
            Opcode::ArithmeticImmWithAccumReg(ArithmeticOpcode::Cmp)
        );

        assert_eq!(
            no_error(decode_opcode(0b00100000)),
            Opcode::ArithmeticRegMemWithReg(ArithmeticOpcode::And)
        );
        assert_eq!(
            no_error(decode_opcode(0b00100100)),
            Opcode::ArithmeticImmWithAccumReg(ArithmeticOpcode::And)
        );

        assert_eq!(
            no_error(decode_opcode(0b00001000)),
            Opcode::ArithmeticRegMemWithReg(ArithmeticOpcode::Or)
        );
        assert_eq!(
            no_error(decode_opcode(0b00001100)),
            Opcode::ArithmeticImmWithAccumReg(ArithmeticOpcode::Or)
        );
    }

    #[test]
    fn should_decode_direction_bit() {
        assert_eq!(decode_direction(0b00000000), Direction::SourceInRegField);
        assert_eq!(decode_direction(0b00000010), Direction::DestInRegField);
        assert_eq!(decode_direction(0b00010000), Direction::SourceInRegField);
        assert_eq!(decode_direction(0b00000011), Direction::DestInRegField);
    }

    #[test]
    fn should_decode_operand_size() {
        assert_eq!(decode_operand_size(0b00000000), OperandSize::Byte);
        assert_eq!(decode_operand_size(0b00000001), OperandSize::Word);
        assert_eq!(decode_operand_size(0b00000010), OperandSize::Byte);
        assert_eq!(decode_operand_size(0b01000001), OperandSize::Word);
    }

    #[test]
    fn should_decode_register_mode() {
        assert_eq!(decode_mode(0b11000000), Mode::Register);
        assert_eq!(decode_mode(0b11001000), Mode::Register);
    }

    #[test]
    fn should_decode_memory_modes() {
        assert_eq!(
            decode_mode(0b00000000),
            Mode::Memory(MemoryMode::NoDisplacement)
        );
        assert_eq!(
            decode_mode(0b01000000),
            Mode::Memory(MemoryMode::ByteDisplacement)
        );
        assert_eq!(
            decode_mode(0b10000000),
            Mode::Memory(MemoryMode::WordDisplacement)
        );
    }

    #[test]
    fn should_decode_first_register_word_size() {
        assert_eq!(decode_first_word_register(0b00000000), WordRegister::AX);
        assert_eq!(decode_first_word_register(0b01000100), WordRegister::AX);
        assert_eq!(decode_first_word_register(0b00001000), WordRegister::CX);
        assert_eq!(decode_first_word_register(0b00010000), WordRegister::DX);
        assert_eq!(decode_first_word_register(0b00011000), WordRegister::BX);
        assert_eq!(decode_first_word_register(0b00100000), WordRegister::SP);
        assert_eq!(decode_first_word_register(0b00101000), WordRegister::BP);
        assert_eq!(decode_first_word_register(0b00110000), WordRegister::SI);
        assert_eq!(decode_first_word_register(0b00111000), WordRegister::DI);
    }

    #[test]
    fn should_decode_first_register_byte_size() {
        assert_eq!(decode_first_byte_register(0b00000000), ByteRegister::AL);
        assert_eq!(decode_first_byte_register(0b01000100), ByteRegister::AL);
        assert_eq!(decode_first_byte_register(0b00001000), ByteRegister::CL);
        assert_eq!(decode_first_byte_register(0b00010000), ByteRegister::DL);
        assert_eq!(decode_first_byte_register(0b00011000), ByteRegister::BL);
        assert_eq!(decode_first_byte_register(0b00100000), ByteRegister::AH);
        assert_eq!(decode_first_byte_register(0b00101000), ByteRegister::CH);
        assert_eq!(decode_first_byte_register(0b00110000), ByteRegister::DH);
        assert_eq!(decode_first_byte_register(0b00111000), ByteRegister::BH);
    }

    #[test]
    fn should_decode_second_register_word_size() {
        assert_eq!(decode_last_word_register(0b00000000), WordRegister::AX);
        assert_eq!(decode_last_word_register(0b01001000), WordRegister::AX);
        assert_eq!(decode_last_word_register(0b00000001), WordRegister::CX);
        assert_eq!(decode_last_word_register(0b00000010), WordRegister::DX);
        assert_eq!(decode_last_word_register(0b00000011), WordRegister::BX);
        assert_eq!(decode_last_word_register(0b00000100), WordRegister::SP);
        assert_eq!(decode_last_word_register(0b00000101), WordRegister::BP);
        assert_eq!(decode_last_word_register(0b00000110), WordRegister::SI);
        assert_eq!(decode_last_word_register(0b00000111), WordRegister::DI);
    }

    #[test]
    fn should_decode_segment_register() {
        assert_eq!(decode_segment_register(0b00000000), SegmentRegister::ES);
        assert_eq!(decode_segment_register(0b00100100), SegmentRegister::ES);
        assert_eq!(decode_segment_register(0b00001000), SegmentRegister::CS);
        assert_eq!(decode_segment_register(0b00010000), SegmentRegister::SS);
        assert_eq!(decode_segment_register(0b00011000), SegmentRegister::DS);
    }

    #[test]
    fn should_decode_direct_memory_address_expressions() {
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::NoDisplacement,
                0b00000110,
                &[0x00, 0x00]
            )),
            MemoryAddressExpression::Direct(0x0000)
        );
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::NoDisplacement,
                0b00000110,
                &[0xcd, 0xab]
            )),
            MemoryAddressExpression::Direct(0xabcd)
        );
    }

    #[test]
    fn should_decode_memory_address_expressions_no_disp() {
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::NoDisplacement,
                0b00000000,
                &[]
            )),
            MemoryAddressExpression::BXPlusSI
        );
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::NoDisplacement,
                0b00000001,
                &[]
            )),
            MemoryAddressExpression::BXPlusDI
        );
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::NoDisplacement,
                0b00000010,
                &[]
            )),
            MemoryAddressExpression::BPPlusSI
        );
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::NoDisplacement,
                0b00000011,
                &[]
            )),
            MemoryAddressExpression::BPPlusDI
        );
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::NoDisplacement,
                0b00000100,
                &[]
            )),
            MemoryAddressExpression::SI
        );
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::NoDisplacement,
                0b00000101,
                &[]
            )),
            MemoryAddressExpression::DI
        );
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::NoDisplacement,
                0b00000111,
                &[]
            )),
            MemoryAddressExpression::BX
        );
    }

    #[test]
    fn should_decode_memory_address_expressions_byte_disp() {
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::ByteDisplacement,
                0b00000000,
                &[0]
            )),
            MemoryAddressExpression::BXPlusSIPlus(0)
        );
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::ByteDisplacement,
                0b00000001,
                &[1]
            )),
            MemoryAddressExpression::BXPlusDIPlus(1)
        );
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::ByteDisplacement,
                0b00000010,
                &[2]
            )),
            MemoryAddressExpression::BPPlusSIPlus(2)
        );
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::ByteDisplacement,
                0b00000011,
                &[3]
            )),
            MemoryAddressExpression::BPPlusDIPlus(3)
        );
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::ByteDisplacement,
                0b00000100,
                &[4]
            )),
            MemoryAddressExpression::SIPlus(4)
        );
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::ByteDisplacement,
                0b00000101,
                &[5]
            )),
            MemoryAddressExpression::DIPlus(5)
        );
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::ByteDisplacement,
                0b00000110,
                &[6]
            )),
            MemoryAddressExpression::BPPlus(6)
        );
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::ByteDisplacement,
                0b00000111,
                &[7]
            )),
            MemoryAddressExpression::BXPlus(7)
        );
    }

    #[test]
    fn should_decode_memory_address_expressions_word_disp() {
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::WordDisplacement,
                0b00000000,
                &[0x00, 0x01]
            )),
            MemoryAddressExpression::BXPlusSIPlus(0x0100)
        );
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::WordDisplacement,
                0b00000001,
                &[0x02, 0x01]
            )),
            MemoryAddressExpression::BXPlusDIPlus(0x0102)
        );
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::WordDisplacement,
                0b00000010,
                &[0xf0, 0x0f]
            )),
            MemoryAddressExpression::BPPlusSIPlus(0x0ff0)
        );
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::WordDisplacement,
                0b00000011,
                &[0x34, 0x12]
            )),
            MemoryAddressExpression::BPPlusDIPlus(0x1234)
        );
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::WordDisplacement,
                0b00000100,
                &[0xff, 0xff]
            )),
            MemoryAddressExpression::SIPlus(0xffff)
        );
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::WordDisplacement,
                0b00000101,
                &[0x12, 0xf0]
            )),
            MemoryAddressExpression::DIPlus(0xf012)
        );
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::WordDisplacement,
                0b00000110,
                &[0xcd, 0xab]
            )),
            MemoryAddressExpression::BPPlus(0xabcd)
        );
        assert_eq!(
            no_error(decode_memory_address_expression(
                MemoryMode::WordDisplacement,
                0b00000111,
                &[0x01, 0xff]
            )),
            MemoryAddressExpression::BXPlus(0xff01)
        );
    }

    #[test]
    fn should_sign_extend_to_16_bit() {
        assert_eq!(sign_extend_to_16_bit(0b00000000), 0b0000000000000000);
        assert_eq!(sign_extend_to_16_bit(0b01010101), 0b0000000001010101);
        assert_eq!(sign_extend_to_16_bit(0b10101010), 0b1111111110101010);
        assert_eq!(sign_extend_to_16_bit(0b11101111), 0b1111111111101111);
    }

    #[test]
    fn should_decode_mov_reg_reg_instructions() {
        assert_eq!(
            no_error(decode_instruction(&[0b10001011, 0b11000011])),
            Instruction {
                op: Operation::Mov,
                source: WordRegister::BX.into(),
                dest: WordRegister::AX.into(),
                size: 2
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10001001, 0b11000011])),
            Instruction {
                op: Operation::Mov,
                source: WordRegister::AX.into(),
                dest: WordRegister::BX.into(),
                size: 2
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10001010, 0b11111010])),
            Instruction {
                op: Operation::Mov,
                source: ByteRegister::DL.into(),
                dest: ByteRegister::BH.into(),
                size: 2
            }
        );
    }

    #[test]
    fn should_decode_mov_reg_mem_instructions() {
        assert_eq!(
            no_error(decode_instruction(&[0b10001011, 0b00011000])),
            Instruction {
                op: Operation::Mov,
                source: MemoryAddressExpression::BXPlusSI.into(),
                dest: WordRegister::BX.into(),
                size: 2
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10001011, 0b00000100])),
            Instruction {
                op: Operation::Mov,
                source: MemoryAddressExpression::SI.into(),
                dest: WordRegister::AX.into(),
                size: 2
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10001010, 0b00000100])),
            Instruction {
                op: Operation::Mov,
                source: MemoryAddressExpression::SI.into(),
                dest: ByteRegister::AL.into(),
                size: 2
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10001001, 0b00000100])),
            Instruction {
                op: Operation::Mov,
                source: WordRegister::AX.into(),
                dest: MemoryAddressExpression::SI.into(),
                size: 2
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10001011, 0b01000110, 3])),
            Instruction {
                op: Operation::Mov,
                source: MemoryAddressExpression::BPPlus(3).into(),
                dest: WordRegister::AX.into(),
                size: 3
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10001011, 0b10000110, 0xcd, 0xab])),
            Instruction {
                op: Operation::Mov,
                source: MemoryAddressExpression::BPPlus(0xabcd).into(),
                dest: WordRegister::AX.into(),
                size: 4
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10001011, 0b00000110, 0xcd, 0xab])),
            Instruction {
                op: Operation::Mov,
                source: MemoryAddressExpression::Direct(0xabcd).into(),
                dest: WordRegister::AX.into(),
                size: 4
            }
        );
    }

    #[test]
    fn should_decode_mov_mem_accum_reg_instructions() {
        assert_eq!(
            no_error(decode_instruction(&[0b10100001, 0xcd, 0xab])),
            Instruction {
                op: Operation::Mov,
                source: MemoryAddressExpression::Direct(0xabcd).into(),
                dest: WordRegister::AX.into(),
                size: 3
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10100011, 0xcd, 0xab])),
            Instruction {
                op: Operation::Mov,
                source: WordRegister::AX.into(),
                dest: MemoryAddressExpression::Direct(0xabcd).into(),
                size: 3
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10100000, 0xcd, 0xab])),
            Instruction {
                op: Operation::Mov,
                source: MemoryAddressExpression::Direct(0xabcd).into(),
                dest: ByteRegister::AL.into(),
                size: 3
            }
        );
    }

    #[test]
    fn should_decode_mov_imm_mem_instructions() {
        assert_eq!(
            no_error(decode_instruction(&[0b11000111, 0b00000100, 0xcd, 0xab])),
            Instruction {
                op: Operation::Mov,
                source: Immediate::Word(0xabcd).into(),
                dest: MemoryAddressExpression::SI.into(),
                size: 4
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b11000110, 0b00000100, 0xcd])),
            Instruction {
                op: Operation::Mov,
                source: Immediate::Byte(0xcd).into(),
                dest: MemoryAddressExpression::SI.into(),
                size: 3
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[
                0b11000111, 0b01000111, 0x56, 0xcd, 0xab
            ])),
            Instruction {
                op: Operation::Mov,
                source: Immediate::Word(0xabcd).into(),
                dest: MemoryAddressExpression::BXPlus(0x56).into(),
                size: 5
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b11000110, 0b01000111, 0x56, 0xab])),
            Instruction {
                op: Operation::Mov,
                source: Immediate::Byte(0xab).into(),
                dest: MemoryAddressExpression::BXPlus(0x56).into(),
                size: 4
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[
                0b11000111, 0b10000110, 0xef, 0xcd, 0xab, 0x89
            ])),
            Instruction {
                op: Operation::Mov,
                source: Immediate::Word(0x89ab).into(),
                dest: MemoryAddressExpression::BPPlus(0xcdef).into(),
                size: 6
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[
                0b11000110, 0b10000110, 0xef, 0xcd, 0xab
            ])),
            Instruction {
                op: Operation::Mov,
                source: Immediate::Byte(0xab).into(),
                dest: MemoryAddressExpression::BPPlus(0xcdef).into(),
                size: 5
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[
                0b11000111, 0b00000110, 0xef, 0xcd, 0xab, 0x89
            ])),
            Instruction {
                op: Operation::Mov,
                source: Immediate::Word(0x89ab).into(),
                dest: MemoryAddressExpression::Direct(0xcdef).into(),
                size: 6
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[
                0b11000110, 0b00000110, 0xef, 0xcd, 0xab
            ])),
            Instruction {
                op: Operation::Mov,
                source: Immediate::Byte(0xab).into(),
                dest: MemoryAddressExpression::Direct(0xcdef).into(),
                size: 5
            }
        );
    }

    #[test]
    fn should_decode_mov_imm_reg_instructions() {
        assert_eq!(
            no_error(decode_instruction(&[0b11000111, 0b11000100, 0xcd, 0xab])),
            Instruction {
                op: Operation::Mov,
                source: Immediate::Word(0xabcd).into(),
                dest: WordRegister::SP.into(),
                size: 4
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b11000110, 0b11000110, 0xab])),
            Instruction {
                op: Operation::Mov,
                source: Immediate::Byte(0xab).into(),
                dest: ByteRegister::DH.into(),
                size: 3
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10111000, 0xcd, 0xab])),
            Instruction {
                op: Operation::Mov,
                source: Immediate::Word(0xabcd).into(),
                dest: WordRegister::AX.into(),
                size: 3
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10110011, 0xab])),
            Instruction {
                op: Operation::Mov,
                source: Immediate::Byte(0xab).into(),
                dest: ByteRegister::BL.into(),
                size: 2
            }
        );
    }

    #[test]
    fn should_decode_mov_reg_seg_reg_instructions() {
        assert_eq!(
            no_error(decode_instruction(&[0b10001110, 0b11000100])),
            Instruction {
                op: Operation::Mov,
                source: WordRegister::SP.into(),
                dest: SegmentRegister::ES.into(),
                size: 2
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10001100, 0b11000100])),
            Instruction {
                op: Operation::Mov,
                source: SegmentRegister::ES.into(),
                dest: WordRegister::SP.into(),
                size: 2
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10001110, 0b11011100])),
            Instruction {
                op: Operation::Mov,
                source: WordRegister::SP.into(),
                dest: SegmentRegister::DS.into(),
                size: 2
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10001110, 0b11011001])),
            Instruction {
                op: Operation::Mov,
                source: WordRegister::CX.into(),
                dest: SegmentRegister::DS.into(),
                size: 2
            }
        );
    }

    #[test]
    fn should_decode_mov_mem_seg_reg_instructions() {
        assert_eq!(
            no_error(decode_instruction(&[0b10001110, 0b00010111])),
            Instruction {
                op: Operation::Mov,
                source: MemoryAddressExpression::BX.into(),
                dest: SegmentRegister::SS.into(),
                size: 2
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10001110, 0b01010101, 0x7])),
            Instruction {
                op: Operation::Mov,
                source: MemoryAddressExpression::DIPlus(0x7).into(),
                dest: SegmentRegister::SS.into(),
                size: 3
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10001110, 0b10010101, 0xcd, 0xab])),
            Instruction {
                op: Operation::Mov,
                source: MemoryAddressExpression::DIPlus(0xabcd).into(),
                dest: SegmentRegister::SS.into(),
                size: 4
            }
        );
    }

    #[test]
    fn should_decode_add_reg_mem_reg_instructions() {
        should_decode_arithmetic_reg_mem_reg_instructions(0b00000000, Operation::Add);
    }

    #[test]
    fn should_decode_adc_reg_mem_reg_instructions() {
        should_decode_arithmetic_reg_mem_reg_instructions(0b00010000, Operation::Adc);
    }

    #[test]
    fn should_decode_sub_reg_mem_reg_instructions() {
        should_decode_arithmetic_reg_mem_reg_instructions(0b00101000, Operation::Sub);
    }

    #[test]
    fn should_decode_sbb_reg_mem_reg_instructions() {
        should_decode_arithmetic_reg_mem_reg_instructions(0b00011000, Operation::Sbb);
    }

    #[test]
    fn should_decode_cmp_reg_mem_reg_instructions() {
        should_decode_arithmetic_reg_mem_reg_instructions(0b00111000, Operation::Cmp);
    }

    #[test]
    fn should_decode_and_reg_mem_reg_instructions() {
        should_decode_arithmetic_reg_mem_reg_instructions(0b00100000, Operation::And);
    }

    #[test]
    fn should_decode_or_reg_mem_reg_instructions() {
        should_decode_arithmetic_reg_mem_reg_instructions(0b00001000, Operation::Or);
    }

    #[allow(clippy::identity_op)]
    fn should_decode_arithmetic_reg_mem_reg_instructions(arithmetic_byte: u8, op: Operation) {
        assert_eq!(
            no_error(decode_instruction(&[
                0b00000001 | arithmetic_byte,
                0b11010111
            ])),
            Instruction {
                op,
                source: WordRegister::DX.into(),
                dest: WordRegister::DI.into(),
                size: 2
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[
                0b00000011 | arithmetic_byte,
                0b11010111
            ])),
            Instruction {
                op,
                source: WordRegister::DI.into(),
                dest: WordRegister::DX.into(),
                size: 2
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[
                0b00000000 | arithmetic_byte,
                0b11010111
            ])),
            Instruction {
                op,
                source: ByteRegister::DL.into(),
                dest: ByteRegister::BH.into(),
                size: 2
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[
                0b00000001 | arithmetic_byte,
                0b00010101
            ])),
            Instruction {
                op,
                source: WordRegister::DX.into(),
                dest: MemoryAddressExpression::DI.into(),
                size: 2
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[
                0b00000001 | arithmetic_byte,
                0b01010101,
                0x7
            ])),
            Instruction {
                op,
                source: WordRegister::DX.into(),
                dest: MemoryAddressExpression::DIPlus(0x7).into(),
                size: 3
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[
                0b00000001 | arithmetic_byte,
                0b10010101,
                0xcd,
                0xab
            ])),
            Instruction {
                op,
                source: WordRegister::DX.into(),
                dest: MemoryAddressExpression::DIPlus(0xabcd).into(),
                size: 4
            }
        );
    }

    #[test]
    fn should_decode_add_imm_reg_mem_instructions() {
        should_decode_arithmetic_imm_reg_mem_instructions(0b00000000, Operation::Add);
    }

    #[test]
    fn should_decode_adc_imm_reg_mem_instructions() {
        should_decode_arithmetic_imm_reg_mem_instructions(0b00010000, Operation::Adc);
    }

    #[test]
    fn should_decode_sub_imm_reg_mem_instructions() {
        should_decode_arithmetic_imm_reg_mem_instructions(0b00101000, Operation::Sub);
    }

    #[test]
    fn should_decode_sbb_imm_reg_mem_instructions() {
        should_decode_arithmetic_imm_reg_mem_instructions(0b00011000, Operation::Sbb);
    }

    #[test]
    fn should_decode_cmp_imm_reg_mem_instructions() {
        should_decode_arithmetic_imm_reg_mem_instructions(0b00111000, Operation::Cmp);
    }

    #[test]
    fn should_decode_and_imm_reg_mem_instructions() {
        should_decode_arithmetic_imm_reg_mem_instructions(0b00100000, Operation::And);
    }

    #[test]
    fn should_decode_or_imm_reg_mem_instructions() {
        should_decode_arithmetic_imm_reg_mem_instructions(0b00001000, Operation::Or);
    }

    fn should_decode_arithmetic_imm_reg_mem_instructions(arithmetic_byte: u8, op: Operation) {
        assert_eq!(
            no_error(decode_instruction(&[
                0b10000001,
                0b11000111 | arithmetic_byte,
                0xcd,
                0xab
            ])),
            Instruction {
                op,
                source: Immediate::Word(0xabcd).into(),
                dest: WordRegister::DI.into(),
                size: 4
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[
                0b10000011,
                0b11000111 | arithmetic_byte,
                0xab
            ])),
            Instruction {
                op,
                source: Immediate::Word(sign_extend_to_16_bit(0xab)).into(),
                dest: WordRegister::DI.into(),
                size: 3
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[
                0b10000000,
                0b11000111 | arithmetic_byte,
                0xab
            ])),
            Instruction {
                op,
                source: Immediate::Byte(0xab).into(),
                dest: ByteRegister::BH.into(),
                size: 3
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[
                0b10000010,
                0b11000111 | arithmetic_byte,
                0xab
            ])),
            Instruction {
                op,
                source: Immediate::Byte(0xab).into(),
                dest: ByteRegister::BH.into(),
                size: 3
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[
                0b10000001,
                0b10000111 | arithmetic_byte,
                0xef,
                0xcd,
                0xcd,
                0xab
            ])),
            Instruction {
                op,
                source: Immediate::Word(0xabcd).into(),
                dest: MemoryAddressExpression::BXPlus(0xcdef).into(),
                size: 6
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[
                0b10000011,
                0b10000111 | arithmetic_byte,
                0xef,
                0xcd,
                0xab
            ])),
            Instruction {
                op,
                source: Immediate::Word(sign_extend_to_16_bit(0xab)).into(),
                dest: MemoryAddressExpression::BXPlus(0xcdef).into(),
                size: 5
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[
                0b10000000,
                0b10000111 | arithmetic_byte,
                0xef,
                0xcd,
                0xab
            ])),
            Instruction {
                op,
                source: Immediate::Byte(0xab).into(),
                dest: MemoryAddressExpression::BXPlus(0xcdef).into(),
                size: 5
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[
                0b10000010,
                0b10000111 | arithmetic_byte,
                0xef,
                0xcd,
                0xab
            ])),
            Instruction {
                op,
                source: Immediate::Byte(0xab).into(),
                dest: MemoryAddressExpression::BXPlus(0xcdef).into(),
                size: 5
            }
        );
    }

    #[test]
    fn should_decode_add_imm_with_accum_reg_instructions() {
        should_decode_arithmetic_imm_with_accum_reg_instructions(0b00000000, Operation::Add);
    }

    #[test]
    fn should_decode_adc_imm_with_accum_reg_instructions() {
        should_decode_arithmetic_imm_with_accum_reg_instructions(0b00010000, Operation::Adc);
    }

    #[test]
    fn should_decode_sbb_imm_with_accum_reg_instructions() {
        should_decode_arithmetic_imm_with_accum_reg_instructions(0b00011000, Operation::Sbb);
    }

    #[test]
    fn should_decode_sub_imm_with_accum_reg_instructions() {
        should_decode_arithmetic_imm_with_accum_reg_instructions(0b00101000, Operation::Sub);
    }

    #[test]
    fn should_decode_cmp_imm_with_accum_reg_instructions() {
        should_decode_arithmetic_imm_with_accum_reg_instructions(0b00111000, Operation::Cmp);
    }

    #[test]
    fn should_decode_and_imm_with_accum_reg_instructions() {
        should_decode_arithmetic_imm_with_accum_reg_instructions(0b00100000, Operation::And);
    }

    #[test]
    fn should_decode_or_imm_with_accum_reg_instructions() {
        should_decode_arithmetic_imm_with_accum_reg_instructions(0b00001000, Operation::Or);
    }

    fn should_decode_arithmetic_imm_with_accum_reg_instructions(
        arithmetic_byte: u8,
        op: Operation,
    ) {
        assert_eq!(
            no_error(decode_instruction(&[
                0b00000101 | arithmetic_byte,
                0xcd,
                0xab
            ])),
            Instruction {
                op,
                source: Immediate::Word(0xabcd).into(),
                dest: WordRegister::AX.into(),
                size: 3
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b00000100 | arithmetic_byte, 0x7])),
            Instruction {
                op,
                source: Immediate::Byte(0x7).into(),
                dest: ByteRegister::AL.into(),
                size: 2
            }
        );
    }

    #[test]
    fn should_fail_decoding_no_instruction() {
        assert_eq!(
            decode_instruction(&[]).unwrap_err(),
            DecodeError::MissingInstructionBytes { bytes: &[] }
        );
    }

    #[test]
    fn should_fail_decoding_single_byte_instruction() {
        let bytes = &[0b10001011];
        assert_eq!(
            decode_instruction(bytes).unwrap_err(),
            DecodeError::MissingInstructionBytes { bytes }
        );
    }

    #[test]
    fn should_display_word_register() {
        assert_eq!(&WordRegister::AX.to_string(), "ax");
        assert_eq!(&WordRegister::BX.to_string(), "bx");
        assert_eq!(&WordRegister::CX.to_string(), "cx");
        assert_eq!(&WordRegister::DX.to_string(), "dx");
        assert_eq!(&WordRegister::SP.to_string(), "sp");
        assert_eq!(&WordRegister::BP.to_string(), "bp");
        assert_eq!(&WordRegister::SI.to_string(), "si");
        assert_eq!(&WordRegister::DI.to_string(), "di");
    }

    #[test]
    fn should_display_byte_register() {
        assert_eq!(&ByteRegister::AL.to_string(), "al");
        assert_eq!(&ByteRegister::AH.to_string(), "ah");
        assert_eq!(&ByteRegister::BL.to_string(), "bl");
        assert_eq!(&ByteRegister::BH.to_string(), "bh");
        assert_eq!(&ByteRegister::CL.to_string(), "cl");
        assert_eq!(&ByteRegister::CH.to_string(), "ch");
        assert_eq!(&ByteRegister::DL.to_string(), "dl");
        assert_eq!(&ByteRegister::DH.to_string(), "dh");
    }

    #[test]
    fn should_display_segment_register() {
        assert_eq!(&SegmentRegister::ES.to_string(), "es");
        assert_eq!(&SegmentRegister::CS.to_string(), "cs");
        assert_eq!(&SegmentRegister::SS.to_string(), "ss");
        assert_eq!(&SegmentRegister::DS.to_string(), "ds");
    }

    #[test]
    fn should_display_direct_memory_address_expressions() {
        assert_eq!(&MemoryAddressExpression::Direct(42).to_string(), "[42]");
        assert_eq!(&MemoryAddressExpression::Direct(4096).to_string(), "[4096]");
    }

    #[test]
    fn should_display_memory_address_expressions_no_disp() {
        assert_eq!(&MemoryAddressExpression::BXPlusSI.to_string(), "[bx + si]");
        assert_eq!(&MemoryAddressExpression::BXPlusDI.to_string(), "[bx + di]");
        assert_eq!(&MemoryAddressExpression::BPPlusSI.to_string(), "[bp + si]");
        assert_eq!(&MemoryAddressExpression::BPPlusDI.to_string(), "[bp + di]");
        assert_eq!(&MemoryAddressExpression::SI.to_string(), "[si]");
        assert_eq!(&MemoryAddressExpression::DI.to_string(), "[di]");
        assert_eq!(&MemoryAddressExpression::BX.to_string(), "[bx]");
    }

    #[test]
    fn should_display_memory_address_expressions_disp() {
        assert_eq!(
            &MemoryAddressExpression::BXPlusSIPlus(0).to_string(),
            "[bx + si]"
        );
        assert_eq!(
            &MemoryAddressExpression::BXPlusDIPlus(u16::MAX).to_string(),
            "[bx + di - 1]"
        );
        assert_eq!(
            &MemoryAddressExpression::BPPlusSIPlus(128).to_string(),
            "[bp + si + 128]"
        );
        assert_eq!(
            &MemoryAddressExpression::BPPlusDIPlus(32767).to_string(),
            "[bp + di + 32767]"
        );
        assert_eq!(
            &MemoryAddressExpression::SIPlus(16).to_string(),
            "[si + 16]"
        );
        assert_eq!(&MemoryAddressExpression::DIPlus(1).to_string(), "[di + 1]");
        assert_eq!(&MemoryAddressExpression::BPPlus(2).to_string(), "[bp + 2]");
        assert_eq!(&MemoryAddressExpression::BXPlus(3).to_string(), "[bx + 3]");
    }

    #[test]
    fn should_display_mov_instructions() {
        assert_eq!(
            &Instruction {
                op: Operation::Mov,
                source: WordRegister::BX.into(),
                dest: WordRegister::AX.into(),
                size: 0
            }
            .to_string(),
            "mov ax, bx"
        );
        assert_eq!(
            &Instruction {
                op: Operation::Mov,
                source: WordRegister::DX.into(),
                dest: WordRegister::CX.into(),
                size: 0
            }
            .to_string(),
            "mov cx, dx"
        );
        assert_eq!(
            &Instruction {
                op: Operation::Mov,
                source: ByteRegister::BL.into(),
                dest: ByteRegister::AH.into(),
                size: 0
            }
            .to_string(),
            "mov ah, bl"
        );
        assert_eq!(
            &Instruction {
                op: Operation::Mov,
                source: ByteRegister::CH.into(),
                dest: ByteRegister::DL.into(),
                size: 0
            }
            .to_string(),
            "mov dl, ch"
        );
        assert_eq!(
            &Instruction {
                op: Operation::Mov,
                source: ByteRegister::AH.into(),
                dest: MemoryAddressExpression::BPPlusSIPlus(3).into(),
                size: 0
            }
            .to_string(),
            "mov [bp + si + 3], ah"
        );
        assert_eq!(
            &Instruction {
                op: Operation::Mov,
                source: MemoryAddressExpression::Direct(256).into(),
                dest: WordRegister::AX.into(),
                size: 0
            }
            .to_string(),
            "mov ax, [256]"
        );
        assert_eq!(
            &Instruction {
                op: Operation::Mov,
                source: Immediate::Word(256).into(),
                dest: MemoryAddressExpression::BPPlusSI.into(),
                size: 0
            }
            .to_string(),
            "mov [bp + si], word 256"
        );
        assert_eq!(
            &Instruction {
                op: Operation::Mov,
                source: Immediate::Byte(7).into(),
                dest: MemoryAddressExpression::DIPlus(9).into(),
                size: 0
            }
            .to_string(),
            "mov [di + 9], byte 7"
        );
        assert_eq!(
            &Instruction {
                op: Operation::Mov,
                source: Immediate::Word(420).into(),
                dest: WordRegister::AX.into(),
                size: 0
            }
            .to_string(),
            "mov ax, 420"
        );
        assert_eq!(
            &Instruction {
                op: Operation::Mov,
                source: Immediate::Byte(42).into(),
                dest: ByteRegister::CL.into(),
                size: 0
            }
            .to_string(),
            "mov cl, 42"
        );
    }

    #[test]
    fn should_display_arithmetic_instructions() {
        assert_eq!(
            &Instruction {
                op: Operation::Add,
                source: Immediate::Word(420).into(),
                dest: WordRegister::BP.into(),
                size: 0
            }
            .to_string(),
            "add bp, 420"
        );
        assert_eq!(
            &Instruction {
                op: Operation::Add,
                source: Immediate::Byte(7).into(),
                dest: MemoryAddressExpression::DIPlus(9).into(),
                size: 0
            }
            .to_string(),
            "add [di + 9], byte 7"
        );
        assert_eq!(
            &Instruction {
                op: Operation::Adc,
                source: Immediate::Word(420).into(),
                dest: WordRegister::BP.into(),
                size: 0
            }
            .to_string(),
            "adc bp, 420"
        );
        assert_eq!(
            &Instruction {
                op: Operation::Adc,
                source: Immediate::Byte(7).into(),
                dest: MemoryAddressExpression::DIPlus(9).into(),
                size: 0
            }
            .to_string(),
            "adc [di + 9], byte 7"
        );
        assert_eq!(
            &Instruction {
                op: Operation::Sub,
                source: Immediate::Word(420).into(),
                dest: WordRegister::BP.into(),
                size: 0
            }
            .to_string(),
            "sub bp, 420"
        );
        assert_eq!(
            &Instruction {
                op: Operation::Sub,
                source: Immediate::Byte(7).into(),
                dest: MemoryAddressExpression::DIPlus(9).into(),
                size: 0
            }
            .to_string(),
            "sub [di + 9], byte 7"
        );
        assert_eq!(
            &Instruction {
                op: Operation::Sbb,
                source: Immediate::Word(420).into(),
                dest: WordRegister::BP.into(),
                size: 0
            }
            .to_string(),
            "sbb bp, 420"
        );
        assert_eq!(
            &Instruction {
                op: Operation::Sbb,
                source: Immediate::Byte(7).into(),
                dest: MemoryAddressExpression::DIPlus(9).into(),
                size: 0
            }
            .to_string(),
            "sbb [di + 9], byte 7"
        );
        assert_eq!(
            &Instruction {
                op: Operation::Cmp,
                source: Immediate::Word(420).into(),
                dest: WordRegister::BP.into(),
                size: 0
            }
            .to_string(),
            "cmp bp, 420"
        );
        assert_eq!(
            &Instruction {
                op: Operation::Cmp,
                source: Immediate::Byte(7).into(),
                dest: MemoryAddressExpression::DIPlus(9).into(),
                size: 0
            }
            .to_string(),
            "cmp [di + 9], byte 7"
        );
        assert_eq!(
            &Instruction {
                op: Operation::And,
                source: Immediate::Word(420).into(),
                dest: WordRegister::BP.into(),
                size: 0
            }
            .to_string(),
            "and bp, 420"
        );
        assert_eq!(
            &Instruction {
                op: Operation::And,
                source: Immediate::Byte(7).into(),
                dest: MemoryAddressExpression::DIPlus(9).into(),
                size: 0
            }
            .to_string(),
            "and [di + 9], byte 7"
        );
        assert_eq!(
            &Instruction {
                op: Operation::Or,
                source: Immediate::Word(420).into(),
                dest: WordRegister::BP.into(),
                size: 0
            }
            .to_string(),
            "or bp, 420"
        );
        assert_eq!(
            &Instruction {
                op: Operation::Or,
                source: Immediate::Byte(7).into(),
                dest: MemoryAddressExpression::DIPlus(9).into(),
                size: 0
            }
            .to_string(),
            "or [di + 9], byte 7"
        );
    }
}
