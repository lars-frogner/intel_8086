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
