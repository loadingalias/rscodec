pub(super) const SPAN: usize = 25;

const ASCII_ZEROS: u64 = 0x3030_3030_3030_3030;

#[inline(always)]
pub(super) const fn eight_digits(value: u64) -> u64 {
    let halves = (value / 10_000) | ((value % 10_000) << 32);
    let hundreds = ((halves * 10_486) >> 20) & 0x0000_007f_0000_007f;
    let pairs = hundreds | (halves - hundreds * 100) << 16;
    let tens = ((pairs * 103) >> 10) & 0x000f_000f_000f_000f;
    tens | (pairs - tens * 10) << 8
}

#[inline(always)]
pub(super) fn sixteen_digits(value: u64) -> (u64, u64) {
    cfg_select! {
        all(target_arch = "aarch64", target_feature = "neon") => super::aarch64::digits::sixteen(value),
        target_arch = "x86_64" => super::x86::digits::sixteen(value),
        _ => {
            const GROUP: u64 = 100_000_000;
            (eight_digits(value / GROUP), eight_digits(value % GROUP))
        }
    }
}

#[inline(always)]
pub(super) fn write(value: u64, negative: bool, dest: &mut [u8; SPAN]) -> usize {
    const GROUP: u64 = 100_000_000;
    dest[0] = b'-';
    let sign = usize::from(negative);
    let Some(dest) = dest[sign..].first_chunk_mut::<{ SPAN - 1 }>() else {
        return sign;
    };
    let (lead, after, count) = if value < GROUP {
        (eight_digits(value), [0; 2], 0)
    } else if value < GROUP * GROUP {
        let (high, low) = sixteen_digits(value);
        (high, [low, 0], 1)
    } else {
        let (high, low) = sixteen_digits(value % (GROUP * GROUP));
        (eight_digits(value / (GROUP * GROUP)), [high, low], 2)
    };
    let zeros = (lead.trailing_zeros() / 8).min(7) as usize;
    dest[..8].copy_from_slice(&((lead | ASCII_ZEROS) >> (8 * zeros)).to_le_bytes());
    let at = 8 - zeros;
    for (index, word) in after.into_iter().enumerate() {
        dest[at + 8 * index..at + 8 * index + 8].copy_from_slice(&(word | ASCII_ZEROS).to_le_bytes());
    }
    sign + at + 8 * count
}
