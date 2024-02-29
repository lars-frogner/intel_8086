use core::fmt;
use intel_8086::{
    decoding,
    testutil::{self, no_error},
};

fn test_decoding(program: impl fmt::Display) {
    let assembled = no_error(testutil::assemble_program(&program));
    let decoded = no_error(decoding::decode_program(&assembled));
    println!("{}", &decoded);
    let reassembled = no_error(testutil::assemble_program(&decoded));
    for idx in 0..usize::min(assembled.len(), reassembled.len()) {
        if reassembled[idx] != assembled[idx] {
            panic!(
                "Byte {} of reassembled program ({:08b}) differs from that of assembled program ({:08b})\n\
                 ---------- Input program ----------\n{}\n\
                 --------- Decoded program ---------\n{}",
                idx, reassembled[idx], assembled[idx], program, decoded
            );
        }
    }
    assert_eq!(reassembled, assembled);
}

#[test]
fn test_decoding_single_mov_reg_reg_word_instruction() {
    test_decoding("bits 16\nmov cx, bx");
}

#[test]
fn test_decoding_single_mov_reg_reg_byte_instruction() {
    test_decoding("bits 16\nmov al, dh");
}

#[test]
fn test_decoding_multiple_mov_reg_reg_word_instructions() {
    test_decoding(
        "\
        bits 16\n\
        mov cx, bx\n\
        mov ch, ah\n\
        mov dx, bx\n\
        mov si, bx\n\
        mov bx, di\n\
        mov al, cl\n\
        mov ch, ch\n\
        mov bx, ax\n\
        mov bx, si\n\
        mov sp, di\n\
        mov bp, ax",
    );
}

#[test]
fn test_decoding_mov_reg_mem_no_disp_instructions() {
    test_decoding(
        "\
        bits 16\n\
        mov al, [bx + si]\n\
        mov ch, [bx + di]\n\
        mov bx, [bp + si]\n\
        mov dx, [bp + di]\n\
        mov cx, [si]\n\
        mov bh, [di]\n\
        mov dx, [bx]\n\
        mov [bx + si], al\n\
        mov [bx + di], ch\n\
        mov [bp + si], bx\n\
        mov [bp + di], dx\n\
        mov [si], cx\n\
        mov [di], bh\n\
        mov [bx], dx",
    );
}

#[test]
fn test_decoding_mov_reg_mem_byte_disp_instructions() {
    test_decoding(
        "\
        bits 16\n\
        mov al, [bx + si + 1]\n\
        mov ch, [bx + di + 2]\n\
        mov bx, [bp + si + 3]\n\
        mov dx, [bp + di + 4]\n\
        mov cx, [si + 5]\n\
        mov bh, [di + 6]\n\
        mov dx, [bx + 127]\n\
        mov bx, [bp]\n\
        mov [bx + si + 1], al\n\
        mov [bx + di + 2], ch\n\
        mov [bp + si + 3], bx\n\
        mov [bp + di + 4], dx\n\
        mov [si + 5], cx\n\
        mov [di + 6], bh\n\
        mov [bx + 127], dx\n\
        mov [bp], bx",
    );
}

#[test]
fn test_decoding_mov_reg_mem_word_disp_instructions() {
    test_decoding(
        "\
        bits 16\n\
        mov al, [bx + si + 128]\n\
        mov ch, [bx + di + 200]\n\
        mov bx, [bp + si + 300]\n\
        mov dx, [bp + di + 400]\n\
        mov cx, [si + 500]\n\
        mov bh, [di + 600]\n\
        mov dx, [bx + 10000]\n\
        mov bx, [bp + 32767]\n\
        mov [bx + si + 128], al\n\
        mov [bx + di + 200], ch\n\
        mov [bp + si + 300], bx\n\
        mov [bp + di + 400], dx\n\
        mov [si + 500], cx\n\
        mov [di + 600], bh\n\
        mov [bx + 10000], dx\n\
        mov [bp + 32767], bx",
    );
}

#[test]
fn test_decoding_mov_reg_mem_neg_disp_instructions() {
    test_decoding(
        "\
        bits 16\n\
        mov cx, [bp - 1]\n\
        mov bl, [bp - 128]\n\
        mov [bp - 129], ah\n\
        mov [bp - 32768], dx",
    );
}

#[test]
fn test_decoding_mov_reg_mem_direct_address_instructions() {
    test_decoding(
        "\
        bits 16\n\
        mov cx, [0]\n\
        mov bl, [255]\n\
        mov [65535], ah",
    );
}

#[test]
fn test_decoding_mov_mem_accum_reg_instructions() {
    test_decoding(
        "\
        bits 16\n\
        mov al, [0]\n\
        mov [256], ax",
    );
}

#[test]
fn test_decoding_mov_imm_mem_instructions() {
    test_decoding(
        "\
        bits 16\n\
        mov [0], byte 0\n\
        mov [3000], byte 1\n\
        mov [bx + si], byte 2\n\
        mov [bx + di], byte 4\n\
        mov [bp + si], byte 8\n\
        mov [bp + di], byte 16\n\
        mov [si], byte 32\n\
        mov [di], byte 64\n\
        mov [bx], byte 128\n\
        mov [bx + si + 1], word 256\n\
        mov [bx + di + 2], word 512\n\
        mov [bp + si + 3], word 1024\n\
        mov [bp + di + 4], word 2048\n\
        mov [si + 5], word 4096\n\
        mov [di + 6], word 8192\n\
        mov [bx + 127], byte -1\n\
        mov [bp], byte -2\n\
        mov [bx + si + 128], byte -4\n\
        mov [bx + di + 200], byte -8\n\
        mov [bp + si + 300], byte -16\n\
        mov [bp + di + 400], byte -32\n\
        mov [si + 500], byte -64\n\
        mov [di + 600], byte -128\n\
        mov [bx + 10000], word -256\n\
        mov [bp + 32767], word -512",
    );
}

#[test]
fn test_decoding_mov_imm_reg_instructions() {
    test_decoding(
        "\
        bits 16\n\
        mov al, 0\n\
        mov bx, 256\n\
        mov bp, 4096\n\
        mov ch, -16",
    )
}

#[test]
fn test_decoding_mov_reg_seg_reg_instructions() {
    test_decoding(
        "\
        bits 16\n\
        mov es, ax\n\
        mov cs, bx\n\
        mov ss, cx\n\
        mov ds, dx\n\
        mov sp, es\n\
        mov bp, cs\n\
        mov si, ss\n\
        mov di, ds",
    )
}

#[test]
fn test_decoding_mov_mem_seg_reg_instructions() {
    test_decoding(
        "\
        bits 16\n\
        mov es, [0]\n\
        mov cs, [bp]\n\
        mov ss, [bx + si]\n\
        mov ds, [si + 5]\n\
        mov [bp + di], es\n\
        mov [bp + di + 78], cs\n\
        mov [bp + di + 999], ss\n\
        mov [1234], ds",
    )
}

#[test]
fn test_decoding_add_mem_reg_reg_instructions() {
    test_decoding_arithmetic_mem_reg_reg_instructions("add");
}

#[test]
fn test_decoding_adc_mem_reg_reg_instructions() {
    test_decoding_arithmetic_mem_reg_reg_instructions("adc");
}

#[test]
fn test_decoding_sub_mem_reg_reg_instructions() {
    test_decoding_arithmetic_mem_reg_reg_instructions("sub");
}

#[test]
fn test_decoding_sbb_mem_reg_reg_instructions() {
    test_decoding_arithmetic_mem_reg_reg_instructions("sbb");
}

#[test]
fn test_decoding_cmp_mem_reg_reg_instructions() {
    test_decoding_arithmetic_mem_reg_reg_instructions("cmp");
}

#[test]
fn test_decoding_and_mem_reg_reg_instructions() {
    test_decoding_arithmetic_mem_reg_reg_instructions("and");
}

#[test]
fn test_decoding_or_mem_reg_reg_instructions() {
    test_decoding_arithmetic_mem_reg_reg_instructions("or");
}

fn test_decoding_arithmetic_mem_reg_reg_instructions(op: &str) {
    test_decoding(format!(
        "\
        bits 16\n\
        {0} ax, bx\n\
        {0} al, bh\n\
        {0} cx, [0]\n\
        {0} dh, [bp]\n\
        {0} [bp + di], sp\n\
        {0} [bp + di + 7], bp\n\
        {0} [bp + di + 512], cl",
        op
    ))
}

#[test]
fn test_decoding_add_imm_mem_reg_instructions() {
    test_decoding_arithmetic_imm_mem_reg_instructions("add");
}

#[test]
fn test_decoding_adc_imm_mem_reg_instructions() {
    test_decoding_arithmetic_imm_mem_reg_instructions("adc");
}

#[test]
fn test_decoding_sub_imm_mem_reg_instructions() {
    test_decoding_arithmetic_imm_mem_reg_instructions("sub");
}

#[test]
fn test_decoding_sbb_imm_mem_reg_instructions() {
    test_decoding_arithmetic_imm_mem_reg_instructions("sbb");
}

#[test]
fn test_decoding_cmp_imm_mem_reg_instructions() {
    test_decoding_arithmetic_imm_mem_reg_instructions("cmp");
}

#[test]
fn test_decoding_and_imm_mem_reg_instructions() {
    test_decoding_arithmetic_imm_mem_reg_instructions("and");
}

#[test]
fn test_decoding_or_imm_mem_reg_instructions() {
    test_decoding_arithmetic_imm_mem_reg_instructions("or");
}

fn test_decoding_arithmetic_imm_mem_reg_instructions(op: &str) {
    test_decoding(format!(
        "\
        bits 16\n\
        {0} bx, 3\n\
        {0} bh, 3\n\
        {0} bx, -3\n\
        {0} bh, -3\n\
        {0} bx, 4096\n\
        {0} bx, -4096\n\
        {0} byte [5], 3\n\
        {0} byte [bp], -3\n\
        {0} byte [di + 1], 3\n\
        {0} byte [bp + di + 512], -3\n\
        {0} word [5], 3\n\
        {0} word [bp], -3\n\
        {0} word [di + 1], 3\n\
        {0} word [bp + di + 512], -3\n\
        {0} word [5], 4096\n\
        {0} word [bp], -4096\n\
        {0} word [di + 1], 4096\n\
        {0} word [bp + di + 512], -4096",
        op
    ))
}

#[test]
fn test_decoding_add_imm_accum_reg_instructions() {
    test_decoding_arithmetic_imm_accum_reg_instructions("add");
}

#[test]
fn test_decoding_adc_imm_accum_reg_instructions() {
    test_decoding_arithmetic_imm_accum_reg_instructions("adc");
}

#[test]
fn test_decoding_sub_imm_accum_reg_instructions() {
    test_decoding_arithmetic_imm_accum_reg_instructions("sub");
}

#[test]
fn test_decoding_sbb_imm_accum_reg_instructions() {
    test_decoding_arithmetic_imm_accum_reg_instructions("sbb");
}

#[test]
fn test_decoding_cmp_imm_accum_reg_instructions() {
    test_decoding_arithmetic_imm_accum_reg_instructions("cmp");
}

#[test]
fn test_decoding_and_imm_accum_reg_instructions() {
    test_decoding_arithmetic_imm_accum_reg_instructions("and");
}

#[test]
fn test_decoding_or_imm_accum_reg_instructions() {
    test_decoding_arithmetic_imm_accum_reg_instructions("or");
}

fn test_decoding_arithmetic_imm_accum_reg_instructions(op: &str) {
    test_decoding(format!(
        "\
        bits 16\n\
        {0} al, 3\n\
        {0} ax, 4096",
        op
    ))
}

#[test]
fn test_decoding_mul_reg_mem_accum_reg_instructions() {
    test_decoding_arithmetic_reg_mem_accum_reg_instructions("mul");
}

#[test]
fn test_decoding_imul_reg_mem_accum_reg_instructions() {
    test_decoding_arithmetic_reg_mem_accum_reg_instructions("imul");
}

#[test]
fn test_decoding_div_reg_mem_accum_reg_instructions() {
    test_decoding_arithmetic_reg_mem_accum_reg_instructions("div");
}

#[test]
fn test_decoding_idiv_reg_mem_accum_reg_instructions() {
    test_decoding_arithmetic_reg_mem_accum_reg_instructions("idiv");
}

#[test]
fn test_decoding_not_reg_mem_accum_reg_instructions() {
    test_decoding_arithmetic_reg_mem_accum_reg_instructions("not");
}

fn test_decoding_arithmetic_reg_mem_accum_reg_instructions(op: &str) {
    test_decoding(format!(
        "\
        bits 16\n\
        {0} ax\n\
        {0} bx\n\
        {0} al\n\
        {0} bh\n\
        {0} byte [4]\n\
        {0} word [4]\n\
        {0} byte [bp]\n\
        {0} word [bp]\n\
        {0} byte [bp + 4]\n\
        {0} word [bp + 4]\n\
        {0} byte [bp + 4096]\n\
        {0} word [bp + 4096]",
        op
    ))
}

#[test]
fn test_decoding_shl_instructions() {
    test_decoding_shift_instructions("shl");
}

#[test]
fn test_decoding_shr_instructions() {
    test_decoding_shift_instructions("shr");
}

#[test]
fn test_decoding_sar_instructions() {
    test_decoding_shift_instructions("sar");
}

#[test]
fn test_decoding_rol_instructions() {
    test_decoding_shift_instructions("rol");
}

#[test]
fn test_decoding_ror_instructions() {
    test_decoding_shift_instructions("ror");
}

#[test]
fn test_decoding_rcl_instructions() {
    test_decoding_shift_instructions("rcl");
}

#[test]
fn test_decoding_rcr_instructions() {
    test_decoding_shift_instructions("rcr");
}

fn test_decoding_shift_instructions(op: &str) {
    test_decoding(format!(
        "\
        bits 16\n\
        {0} ax, 1\n\
        {0} bx, 1\n\
        {0} al, 1\n\
        {0} bh, 1\n\
        {0} ax, cl\n\
        {0} bx, cl\n\
        {0} al, cl\n\
        {0} bh, cl\n\
        {0} word [bp], 1\n\
        {0} word [bp + 4], 1\n\
        {0} word [bp + 4096], 1\n\
        {0} byte [bp], 1\n\
        {0} byte [bp + 4], 1\n\
        {0} byte [bp + 4096], 1\n\
        {0} word [bp], cl\n\
        {0} word [bp + 4], cl\n\
        {0} word [bp + 4096], cl\n\
        {0} byte [bp], cl\n\
        {0} byte [bp + 4], cl\n\
        {0} byte [bp + 4096], cl",
        op
    ))
}
