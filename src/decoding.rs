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
pub struct DecodedInstruction {
    instr: Instruction,
    size: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Instruction {
    MovRegRegWord(MovRegRegWord),
    MovRegRegByte(MovRegRegByte),
    MovRegMem(MovRegMem),
    MovMemAccumReg(MovMemAccumReg),
    MovImmMem(MovImmMem),
    MovImmReg(MovImmReg),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MovRegRegWord {
    direction: Direction,
    reg1: WordRegister,
    reg2: WordRegister,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MovRegRegByte {
    direction: Direction,
    reg1: ByteRegister,
    reg2: ByteRegister,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MovRegMem {
    direction: Direction,
    reg: Register,
    mem_addr_expr: MemoryAddressExpression,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MovMemAccumReg {
    operand_size: OperandSize,
    direction: Direction,
    mem_addr: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MovImmMem {
    imm: Immediate,
    mem_addr_expr: MemoryAddressExpression,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MovImmReg {
    imm: u16,
    reg: Register,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opcode {
    MovRegMemToFromReg,
    MovMemToAccumReg,
    MovImmToRegMem,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    FromTo,
    ToFrom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandSize {
    Word,
    Byte,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Register,
    Memory(MemoryMode),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryMode {
    NoDisplacement,
    ByteDisplacement,
    WordDisplacement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Register {
    Full(WordRegister),
    Half(ByteRegister),
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
        write!(
            f,
            "{}",
            match self {
                Self::MovRegRegWord(instr) => {
                    instr.to_string()
                }
                Self::MovRegRegByte(instr) => {
                    instr.to_string()
                }
                Self::MovRegMem(instr) => {
                    instr.to_string()
                }
                Self::MovMemAccumReg(instr) => {
                    instr.to_string()
                }
                Self::MovImmMem(instr) => {
                    instr.to_string()
                }
                Self::MovImmReg(instr) => {
                    instr.to_string()
                }
            }
        )
    }
}

impl MovRegRegWord {
    const SIZE: usize = 2;
}

impl fmt::Display for MovRegRegWord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (first, second) = match self.direction {
            Direction::ToFrom => (self.reg1, self.reg2),
            Direction::FromTo => (self.reg2, self.reg1),
        };
        write!(f, "mov {}, {}", first, second)
    }
}

impl MovRegRegByte {
    const SIZE: usize = 2;
}

impl fmt::Display for MovRegRegByte {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (first, second) = match self.direction {
            Direction::ToFrom => (self.reg1, self.reg2),
            Direction::FromTo => (self.reg2, self.reg1),
        };
        write!(f, "mov {}, {}", first, second)
    }
}

impl fmt::Display for MovRegMem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.direction {
            Direction::ToFrom => write!(f, "mov {}, {}", self.reg, self.mem_addr_expr),
            Direction::FromTo => write!(f, "mov {}, {}", self.mem_addr_expr, self.reg),
        }
    }
}

impl fmt::Display for MovMemAccumReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let reg = match self.operand_size {
            OperandSize::Word => Register::Full(WordRegister::AX),
            OperandSize::Byte => Register::Half(ByteRegister::AL),
        };
        let addr = MemoryAddressExpression::Direct(self.mem_addr);
        match self.direction {
            Direction::ToFrom => write!(f, "mov {}, {}", reg, addr),
            Direction::FromTo => write!(f, "mov {}, {}", addr, reg),
        }
    }
}

impl fmt::Display for MovImmMem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mov {}, {}", self.mem_addr_expr, self.imm)
    }
}

impl fmt::Display for MovImmReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mov {}, {}", self.reg, self.imm)
    }
}

impl Direction {
    fn flipped(&self) -> Self {
        match self {
            Self::ToFrom => Self::FromTo,
            Self::FromTo => Self::ToFrom,
        }
    }
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full(reg) => reg.fmt(f),
            Self::Half(reg) => reg.fmt(f),
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
            Self::Word(imm) => write!(f, "word {}", imm),
            Self::Byte(imm) => write!(f, "byte {}", imm),
        }
    }
}

pub fn decode_program(bytes: &[u8]) -> Result<Program, ProgramDecodeError> {
    let mut instructions = Vec::with_capacity(bytes.len() / 2);
    let mut ip = 0;
    while ip < bytes.len() {
        let DecodedInstruction { instr, size } =
            decode_instruction(&bytes[ip..]).map_err(|error| ProgramDecodeError { error, ip })?;
        ip += size;
        instructions.push(instr);
    }
    Ok(Program { instructions })
}

pub fn decode_instruction(bytes: &[u8]) -> DecodeResult<DecodedInstruction> {
    if bytes.is_empty() {
        return Err(DecodeError::MissingInstructionBytes { bytes });
    }
    let opcode = decode_opcode(bytes[0])?;
    let (instr, size) = match opcode {
        Opcode::MovRegMemToFromReg => {
            if bytes.len() < 2 {
                return Err(DecodeError::MissingInstructionBytes { bytes });
            }
            let direction = decode_direction(bytes[0]);
            let operand_size = decode_operand_size(bytes[0]);
            let mode = decode_mode(bytes[1]);
            match mode {
                Mode::Register => match operand_size {
                    OperandSize::Word => (
                        Instruction::MovRegRegWord(MovRegRegWord {
                            direction,
                            reg1: decode_first_word_register(bytes[1]),
                            reg2: decode_second_word_register(bytes[1]),
                        }),
                        MovRegRegWord::SIZE,
                    ),
                    OperandSize::Byte => (
                        Instruction::MovRegRegByte(MovRegRegByte {
                            direction,
                            reg1: decode_first_byte_register(bytes[1]),
                            reg2: decode_second_byte_register(bytes[1]),
                        }),
                        MovRegRegByte::SIZE,
                    ),
                },
                Mode::Memory(mem_mode) => {
                    let mem_addr_expr =
                        decode_memory_address_expression(mem_mode, bytes[1], &bytes[2..])?;
                    (
                        Instruction::MovRegMem(MovRegMem {
                            direction,
                            reg: decode_first_register(operand_size, bytes[1]),
                            mem_addr_expr,
                        }),
                        2 + mem_addr_expr.displacement_size(mem_mode),
                    )
                }
            }
        }
        Opcode::MovMemToAccumReg => {
            if bytes.len() < 3 {
                return Err(DecodeError::MissingInstructionBytes { bytes });
            }
            let direction = decode_direction(bytes[0]).flipped();
            let operand_size = decode_operand_size(bytes[0]);
            (
                Instruction::MovMemAccumReg(MovMemAccumReg {
                    operand_size,
                    direction,
                    mem_addr: combine_low_and_high_bytes(bytes[1], bytes[2]),
                }),
                3,
            )
        }
        Opcode::MovImmToRegMem => {
            if bytes.len() < 3 {
                return Err(DecodeError::MissingInstructionBytes { bytes });
            }
            let operand_size = decode_operand_size(bytes[0]);
            let mode = decode_mode(bytes[1]);
            match mode {
                Mode::Register => todo!(),
                Mode::Memory(mem_mode) => {
                    let mem_addr_expr =
                        decode_memory_address_expression(mem_mode, bytes[1], &bytes[2..])?;
                    let disp_size = mem_addr_expr.displacement_size(mem_mode);
                    let (imm, data_size) = match operand_size {
                        OperandSize::Word => {
                            if bytes.len() < 2 + disp_size + 2 {
                                return Err(DecodeError::MissingInstructionBytes { bytes });
                            }
                            (
                                Immediate::Word(combine_low_and_high_bytes(
                                    bytes[2 + disp_size],
                                    bytes[2 + disp_size + 1],
                                )),
                                2,
                            )
                        }
                        OperandSize::Byte => {
                            if bytes.len() < 2 + disp_size + 1 {
                                return Err(DecodeError::MissingInstructionBytes { bytes });
                            }
                            (Immediate::Byte(bytes[2 + disp_size]), 1)
                        }
                    };
                    (
                        Instruction::MovImmMem(MovImmMem { imm, mem_addr_expr }),
                        2 + disp_size + data_size,
                    )
                }
            }
        }
    };
    Ok(DecodedInstruction { instr, size })
}

fn decode_opcode(byte1: u8) -> DecodeResult<'static, Opcode> {
    match byte1 & 0b11111110 {
        0b11000110 => Ok(Opcode::MovImmToRegMem),
        _ => match byte1 & 0b11111100 {
            0b10001000 => Ok(Opcode::MovRegMemToFromReg),
            0b10100000 => Ok(Opcode::MovMemToAccumReg),
            _ => Err(DecodeError::UnknownOpcode { byte: byte1 }),
        },
    }
}

fn decode_direction(byte1: u8) -> Direction {
    if byte1 & 0b00000010 == 0b00000000 {
        Direction::FromTo
    } else {
        Direction::ToFrom
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

fn decode_first_register(operand_size: OperandSize, byte2: u8) -> Register {
    match operand_size {
        OperandSize::Word => Register::Full(decode_first_word_register(byte2)),
        OperandSize::Byte => Register::Half(decode_first_byte_register(byte2)),
    }
}

fn decode_first_word_register(byte2: u8) -> WordRegister {
    decode_word_register(shift_mask_register_byte(byte2, 3))
}

fn decode_second_word_register(byte2: u8) -> WordRegister {
    decode_word_register(shift_mask_register_byte(byte2, 0))
}

fn decode_first_byte_register(byte2: u8) -> ByteRegister {
    decode_byte_register(shift_mask_register_byte(byte2, 3))
}

fn decode_second_byte_register(byte2: u8) -> ByteRegister {
    decode_byte_register(shift_mask_register_byte(byte2, 0))
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
            0b00000110 => {
                if disp_bytes.len() < 2 {
                    return Err(DecodeError::MissingDisplacementBytes { bytes: disp_bytes });
                }
                let addr = combine_low_and_high_bytes(disp_bytes[0], disp_bytes[1]);
                MemoryAddressExpression::Direct(addr)
            }
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

fn sign_extend_to_16_bit(byte: u8) -> u16 {
    ((byte as u16).shl(8) as i16).shr(8) as u16
}

fn combine_low_and_high_bytes(low_byte: u8, high_byte: u8) -> u16 {
    low_byte as u16 + (high_byte as u16).shl(8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::no_error;

    #[test]
    fn should_decode_mov_reg_mem_to_from_reg_opcode() {
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
    }

    #[test]
    fn should_decode_mov_mem_to_accum_reg_opcode() {
        assert_eq!(
            no_error(decode_opcode(0b10100000)),
            Opcode::MovMemToAccumReg
        );
        assert_eq!(
            no_error(decode_opcode(0b10100011)),
            Opcode::MovMemToAccumReg
        );
    }

    #[test]
    fn should_decode_mov_imm_to_reg_mem_opcode() {
        assert_eq!(no_error(decode_opcode(0b11000110)), Opcode::MovImmToRegMem);
        assert_eq!(no_error(decode_opcode(0b11000111)), Opcode::MovImmToRegMem);
    }

    #[test]
    fn should_fail_decoding_unknown_opcode() {
        let byte1 = 0b00000000;
        let result = decode_opcode(byte1);
        assert_eq!(
            result.unwrap_err(),
            DecodeError::UnknownOpcode { byte: byte1 }
        );
    }

    #[test]
    fn should_decode_direction_bit() {
        assert_eq!(decode_direction(0b00000000), Direction::FromTo);
        assert_eq!(decode_direction(0b00000010), Direction::ToFrom);
        assert_eq!(decode_direction(0b00010000), Direction::FromTo);
        assert_eq!(decode_direction(0b00000011), Direction::ToFrom);
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
        assert_eq!(decode_second_word_register(0b00000000), WordRegister::AX);
        assert_eq!(decode_second_word_register(0b01001000), WordRegister::AX);
        assert_eq!(decode_second_word_register(0b00000001), WordRegister::CX);
        assert_eq!(decode_second_word_register(0b00000010), WordRegister::DX);
        assert_eq!(decode_second_word_register(0b00000011), WordRegister::BX);
        assert_eq!(decode_second_word_register(0b00000100), WordRegister::SP);
        assert_eq!(decode_second_word_register(0b00000101), WordRegister::BP);
        assert_eq!(decode_second_word_register(0b00000110), WordRegister::SI);
        assert_eq!(decode_second_word_register(0b00000111), WordRegister::DI);
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
    fn should_decode_mov_reg_reg_word_instructions() {
        assert_eq!(
            no_error(decode_instruction(&[0b10001011, 0b11000011])),
            DecodedInstruction {
                instr: Instruction::MovRegRegWord(MovRegRegWord {
                    direction: Direction::ToFrom,
                    reg1: WordRegister::AX,
                    reg2: WordRegister::BX
                }),
                size: 2
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10001001, 0b11000011])),
            DecodedInstruction {
                instr: Instruction::MovRegRegWord(MovRegRegWord {
                    direction: Direction::FromTo,
                    reg1: WordRegister::AX,
                    reg2: WordRegister::BX
                }),
                size: 2
            }
        );
    }

    #[test]
    fn should_decode_mov_reg_reg_byte_instructions() {
        assert_eq!(
            no_error(decode_instruction(&[0b10001010, 0b11111010])),
            DecodedInstruction {
                instr: Instruction::MovRegRegByte(MovRegRegByte {
                    direction: Direction::ToFrom,
                    reg1: ByteRegister::BH,
                    reg2: ByteRegister::DL
                }),
                size: 2
            }
        );
    }

    #[test]
    fn should_decode_mov_reg_mem_instructions() {
        assert_eq!(
            no_error(decode_instruction(&[0b10001011, 0b00011000])),
            DecodedInstruction {
                instr: Instruction::MovRegMem(MovRegMem {
                    direction: Direction::ToFrom,
                    reg: Register::Full(WordRegister::BX),
                    mem_addr_expr: MemoryAddressExpression::BXPlusSI
                }),
                size: 2
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10001011, 0b00000100])),
            DecodedInstruction {
                instr: Instruction::MovRegMem(MovRegMem {
                    direction: Direction::ToFrom,
                    reg: Register::Full(WordRegister::AX),
                    mem_addr_expr: MemoryAddressExpression::SI
                }),
                size: 2
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10001010, 0b00000100])),
            DecodedInstruction {
                instr: Instruction::MovRegMem(MovRegMem {
                    direction: Direction::ToFrom,
                    reg: Register::Half(ByteRegister::AL),
                    mem_addr_expr: MemoryAddressExpression::SI
                }),
                size: 2
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10001001, 0b00000100])),
            DecodedInstruction {
                instr: Instruction::MovRegMem(MovRegMem {
                    direction: Direction::FromTo,
                    reg: Register::Full(WordRegister::AX),
                    mem_addr_expr: MemoryAddressExpression::SI
                }),
                size: 2
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10001011, 0b01000110, 3])),
            DecodedInstruction {
                instr: Instruction::MovRegMem(MovRegMem {
                    direction: Direction::ToFrom,
                    reg: Register::Full(WordRegister::AX),
                    mem_addr_expr: MemoryAddressExpression::BPPlus(3)
                }),
                size: 3
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10001011, 0b10000110, 0xcd, 0xab])),
            DecodedInstruction {
                instr: Instruction::MovRegMem(MovRegMem {
                    direction: Direction::ToFrom,
                    reg: Register::Full(WordRegister::AX),
                    mem_addr_expr: MemoryAddressExpression::BPPlus(0xabcd)
                }),
                size: 4
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10001011, 0b00000110, 0xcd, 0xab])),
            DecodedInstruction {
                instr: Instruction::MovRegMem(MovRegMem {
                    direction: Direction::ToFrom,
                    reg: Register::Full(WordRegister::AX),
                    mem_addr_expr: MemoryAddressExpression::Direct(0xabcd)
                }),
                size: 4
            }
        );
    }

    #[test]
    fn should_decode_mov_mem_accum_reg_instructions() {
        assert_eq!(
            no_error(decode_instruction(&[0b10100001, 0xcd, 0xab])),
            DecodedInstruction {
                instr: Instruction::MovMemAccumReg(MovMemAccumReg {
                    operand_size: OperandSize::Word,
                    direction: Direction::ToFrom,
                    mem_addr: 0xabcd
                }),
                size: 3
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10100011, 0xcd, 0xab])),
            DecodedInstruction {
                instr: Instruction::MovMemAccumReg(MovMemAccumReg {
                    operand_size: OperandSize::Word,
                    direction: Direction::FromTo,
                    mem_addr: 0xabcd
                }),
                size: 3
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b10100000, 0xcd, 0xab])),
            DecodedInstruction {
                instr: Instruction::MovMemAccumReg(MovMemAccumReg {
                    operand_size: OperandSize::Byte,
                    direction: Direction::ToFrom,
                    mem_addr: 0xabcd
                }),
                size: 3
            }
        );
    }

    #[test]
    fn should_decode_mov_imm_mem_instructions() {
        assert_eq!(
            no_error(decode_instruction(&[0b11000111, 0b00000100, 0xcd, 0xab])),
            DecodedInstruction {
                instr: Instruction::MovImmMem(MovImmMem {
                    imm: Immediate::Word(0xabcd),
                    mem_addr_expr: MemoryAddressExpression::SI
                }),
                size: 4
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b11000110, 0b00000100, 0xcd])),
            DecodedInstruction {
                instr: Instruction::MovImmMem(MovImmMem {
                    imm: Immediate::Byte(0xcd),
                    mem_addr_expr: MemoryAddressExpression::SI
                }),
                size: 3
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[
                0b11000111, 0b01000111, 0x56, 0xcd, 0xab
            ])),
            DecodedInstruction {
                instr: Instruction::MovImmMem(MovImmMem {
                    imm: Immediate::Word(0xabcd),
                    mem_addr_expr: MemoryAddressExpression::BXPlus(0x56)
                }),
                size: 5
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[0b11000110, 0b01000111, 0x56, 0xab])),
            DecodedInstruction {
                instr: Instruction::MovImmMem(MovImmMem {
                    imm: Immediate::Byte(0xab),
                    mem_addr_expr: MemoryAddressExpression::BXPlus(0x56)
                }),
                size: 4
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[
                0b11000111, 0b10000110, 0xef, 0xcd, 0xab, 0x89
            ])),
            DecodedInstruction {
                instr: Instruction::MovImmMem(MovImmMem {
                    imm: Immediate::Word(0x89ab),
                    mem_addr_expr: MemoryAddressExpression::BPPlus(0xcdef)
                }),
                size: 6
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[
                0b11000110, 0b10000110, 0xef, 0xcd, 0xab
            ])),
            DecodedInstruction {
                instr: Instruction::MovImmMem(MovImmMem {
                    imm: Immediate::Byte(0xab),
                    mem_addr_expr: MemoryAddressExpression::BPPlus(0xcdef)
                }),
                size: 5
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[
                0b11000111, 0b00000110, 0xef, 0xcd, 0xab, 0x89
            ])),
            DecodedInstruction {
                instr: Instruction::MovImmMem(MovImmMem {
                    imm: Immediate::Word(0x89ab),
                    mem_addr_expr: MemoryAddressExpression::Direct(0xcdef)
                }),
                size: 6
            }
        );
        assert_eq!(
            no_error(decode_instruction(&[
                0b11000110, 0b00000110, 0xef, 0xcd, 0xab
            ])),
            DecodedInstruction {
                instr: Instruction::MovImmMem(MovImmMem {
                    imm: Immediate::Byte(0xab),
                    mem_addr_expr: MemoryAddressExpression::Direct(0xcdef)
                }),
                size: 5
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
    fn should_fail_decoding_instruction_with_unknown_opcode() {
        let bytes = &[0b00000011, 0b11000011];
        assert_eq!(
            decode_instruction(bytes).unwrap_err(),
            DecodeError::UnknownOpcode { byte: bytes[0] }
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
    fn should_display_mov_reg_reg_instructions() {
        assert_eq!(
            &Instruction::MovRegRegWord(MovRegRegWord {
                direction: Direction::ToFrom,
                reg1: WordRegister::AX,
                reg2: WordRegister::BX
            })
            .to_string(),
            "mov ax, bx"
        );
        assert_eq!(
            &Instruction::MovRegRegWord(MovRegRegWord {
                direction: Direction::ToFrom,
                reg1: WordRegister::CX,
                reg2: WordRegister::DX
            })
            .to_string(),
            "mov cx, dx"
        );
        assert_eq!(
            &Instruction::MovRegRegWord(MovRegRegWord {
                direction: Direction::FromTo,
                reg1: WordRegister::AX,
                reg2: WordRegister::BX
            })
            .to_string(),
            "mov bx, ax"
        );
        assert_eq!(
            &Instruction::MovRegRegByte(MovRegRegByte {
                direction: Direction::ToFrom,
                reg1: ByteRegister::AH,
                reg2: ByteRegister::BL,
            })
            .to_string(),
            "mov ah, bl"
        );
        assert_eq!(
            &Instruction::MovRegRegByte(MovRegRegByte {
                direction: Direction::ToFrom,
                reg1: ByteRegister::DL,
                reg2: ByteRegister::CH,
            })
            .to_string(),
            "mov dl, ch"
        );
        assert_eq!(
            &Instruction::MovRegRegByte(MovRegRegByte {
                direction: Direction::FromTo,
                reg1: ByteRegister::AH,
                reg2: ByteRegister::BL,
            })
            .to_string(),
            "mov bl, ah"
        );
    }

    #[test]
    fn should_display_mov_reg_mem_instructions() {
        assert_eq!(
            &Instruction::MovRegMem(MovRegMem {
                direction: Direction::FromTo,
                reg: Register::Half(ByteRegister::AH),
                mem_addr_expr: MemoryAddressExpression::BPPlusSIPlus(3),
            })
            .to_string(),
            "mov [bp + si + 3], ah"
        );
        assert_eq!(
            &Instruction::MovMemAccumReg(MovMemAccumReg {
                operand_size: OperandSize::Word,
                direction: Direction::ToFrom,
                mem_addr: 256
            })
            .to_string(),
            "mov ax, [256]"
        );
    }

    #[test]
    fn should_display_mov_imm_mem_instructions() {
        assert_eq!(
            &Instruction::MovImmMem(MovImmMem {
                imm: Immediate::Word(256),
                mem_addr_expr: MemoryAddressExpression::BPPlusSI,
            })
            .to_string(),
            "mov [bp + si], word 256"
        );
        assert_eq!(
            &Instruction::MovImmMem(MovImmMem {
                imm: Immediate::Byte(7),
                mem_addr_expr: MemoryAddressExpression::DIPlus(9),
            })
            .to_string(),
            "mov [di + 9], byte 7"
        );
    }

    #[test]
    fn should_display_mov_imm_reg_instructions() {
        assert_eq!(
            &Instruction::MovImmReg(MovImmReg {
                imm: 420,
                reg: Register::Full(WordRegister::AX),
            })
            .to_string(),
            "mov ax, 420"
        );
        assert_eq!(
            &Instruction::MovImmReg(MovImmReg {
                imm: 42,
                reg: Register::Half(ByteRegister::CL),
            })
            .to_string(),
            "mov cl, 42"
        );
    }
}
