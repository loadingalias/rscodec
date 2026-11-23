//! The instruction lists of the secret code, as their owners publish them.
//!
//! - `DOIT`: Intel, "Data Operand Independent Timing Instructions", the table of the page as it
//!   was on 2026-11-23 (updated 2026-02-23), with each mnemonic once. Its `REP_` rows are the string
//!   forms with a `rep` prefix.
//! - `DIT`: Arm, A64 ISA XML `ISA_A64_xml_A_profile-2026-03` (the archive of the permalink
//!   `ISA_A64_xml_A_profile-2026-06.tar.gz`, SHA-256
//!   `63a01a1696483bbe2edfef9e0f0cd053d6c1c619ec0587876cb7a60bb344f354`): each base, Advanced SIMD,
//!   and floating-point mnemonic whose files each say "This instruction is a data-independent-time
//!   instruction", or say it through the instruction of their alias. `ldr`, `ldrsw`, and `mov` are in
//!   it too: only their forms that read no register lack the note (the literal loads, `movz`, and
//!   `movn`).
//! - `DIT_SVE`: from the same archive, each SVE and SVE2 mnemonic whose files each have the note.
//!   An SVE instruction has a `z` or `p` register. The SVE mnemonics with a form that lacks the note
//!   (`cpy`, `dup`, `fmov`, `index`, `mov`, `rev`, `trn1`, `trn2`, `uzp1`, `uzp2`, `zip1`, `zip2`)
//!   are not in it: the scan passes them only with no data operand.

/// Whether Intel lists `mnemonic`, in capitals, with data operand independent timing.
pub(crate) fn doit(mnemonic: &str) -> bool {
    DOIT.split_ascii_whitespace().any(|name| name == mnemonic)
}

/// Whether Arm lists `mnemonic`, in lowercase, as a data-independent-time instruction.
pub(crate) fn dit(mnemonic: &str) -> bool {
    DIT.split_ascii_whitespace().any(|name| name == mnemonic)
}

/// Whether Arm lists the SVE `mnemonic`, in lowercase, as a data-independent-time instruction.
pub(crate) fn dit_sve(mnemonic: &str) -> bool {
    DIT_SVE.split_ascii_whitespace().any(|name| name == mnemonic)
}

const DOIT: &str = "ADC ADCX ADD ADOX AESDEC AESDECLAST AESENC AESENCLAST AESIMC AESKEYGENASSIST AND ANDN BSWAP BT \
BTC BTR BTS CMOVB CMOVBE CMOVL CMOVLE CMOVNB CMOVNBE CMOVNL CMOVNLE CMOVNO CMOVNP CMOVNS CMOVNZ \
CMOVO CMOVP CMOVS CMOVZ CMP CQO DEC GF2P8AFFINEINVQB GF2P8AFFINEQB GF2P8MULB IMUL INC KADDB \
KADDD KADDQ KADDW KANDB KANDD KANDNB KANDND KANDNQ KANDNW KANDQ KANDW KMOVB KMOVD KMOVQ KMOVW \
KNOTB KNOTD KNOTQ KNOTW KORB KORD KORQ KORTESTB KORTESTD KORTESTQ KORTESTW KORW KSHIFTLB \
KSHIFTLD KSHIFTLQ KSHIFTLW KSHIFTRB KSHIFTRD KSHIFTRQ KSHIFTRW KTESTB KTESTD KTESTQ KTESTW \
KUNPCKBW KUNPCKDQ KUNPCKWD KXNORB KXNORD KXNORQ KXNORW KXORB KXORD KXORQ KXORW LDDQU LEA MOV \
MOVAPS MOVD MOVDDUP MOVDQ2Q MOVDQA MOVDQU MOVNTDQ MOVNTDQA MOVQ MOVSB MOVSD MOVSHDUP MOVSLDUP \
MOVSQ MOVSS MOVSW MOVSX MOVSXD MOVUPS MOVZX MPSADBW MUL MULX NEG NOT OR PABSB PABSD PABSW \
PACKSSDW PACKSSWB PACKUSDW PACKUSWB PADDB PADDD PADDQ PADDSB PADDSW PADDUSB PADDUSW PADDW \
PALIGNR PAND PANDN PAVGB PAVGW PBLENDVB PBLENDW PCLMULQDQ PCMPEQB PCMPEQD PCMPEQQ PCMPEQW \
PCMPGTB PCMPGTD PCMPGTQ PCMPGTW PEXTRB PEXTRD PEXTRQ PEXTRW PHADDD PHADDSW PHADDW PHMINPOSUW \
PHSUBD PHSUBSW PHSUBW PINSRB PINSRD PINSRQ PINSRW PMADDUBSW PMADDWD PMAXSB PMAXSW PMAXUB PMAXUD \
PMAXUW PMINSB PMINSW PMINUB PMINUD PMINUW PMOVMSKB PMOVSXBD PMOVSXBQ PMOVSXBW PMOVSXDQ PMOVSXWD \
PMOVSXWQ PMOVZXBD PMOVZXBQ PMOVZXBW PMOVZXDQ PMOVZXWD PMOVZXWQ PMULDQ PMULHRSW PMULHUW PMULHW \
PMULLD PMULLW PMULUDQ POP POR PSADBW PSHUFB PSHUFD PSHUFHW PSHUFLW PSIGNB PSIGND PSIGNW PSLLD \
PSLLDQ PSLLQ PSLLW PSRAD PSRAW PSRLD PSRLDQ PSRLQ PSRLW PSUBB PSUBD PSUBQ PSUBSB PSUBSW PSUBUSB \
PSUBUSW PSUBW PTEST PUNPCKHBW PUNPCKHDQ PUNPCKHQDQ PUNPCKHWD PUNPCKLBW PUNPCKLDQ PUNPCKLQDQ \
PUNPCKLWD PUSH PXOR REP_MOVSB REP_MOVSD REP_MOVSW REP_STOSB REP_STOSD REP_STOSW ROL ROR RORX SAR \
SBB SETB SETBE SETL SETLE SETNB SETNBE SETNL SETNLE SETNO SETNP SETNS SETNZ SETO SETP SETS SETZ \
SHA1MSG1 SHA1MSG2 SHA1NEXTE SHA1RNDS4 SHA256MSG1 SHA256MSG2 SHA256RNDS2 SHL SHLD SHLX SHR SHRD \
SHRX SUB TEST VAESDEC VAESDECLAST VAESENC VAESENCLAST VAESIMC VAESKEYGENASSIST VALIGND VALIGNQ \
VBLENDMPD VBLENDMPS VBROADCASTF128 VBROADCASTF32X2 VBROADCASTF32X4 VBROADCASTF32X8 \
VBROADCASTF64X2 VBROADCASTF64X4 VBROADCASTI128 VBROADCASTI32X2 VBROADCASTI32X4 VBROADCASTI32X8 \
VBROADCASTI64X2 VBROADCASTI64X4 VBROADCASTSD VBROADCASTSS VCOMPRESSPD VCOMPRESSPS VDBPSADBW \
VEXPANDPD VEXPANDPS VEXTRACTF128 VEXTRACTF32X4 VEXTRACTF32X8 VEXTRACTF64X2 VEXTRACTF64X4 \
VEXTRACTI128 VEXTRACTI32X4 VEXTRACTI32X8 VEXTRACTI64X2 VEXTRACTI64X4 VGF2P8AFFINEINVQB \
VGF2P8AFFINEQB VGF2P8MULB VINSERTF128 VINSERTF32X4 VINSERTF32X8 VINSERTF64X2 VINSERTF64X4 \
VINSERTI128 VINSERTI32X4 VINSERTI32X8 VINSERTI64X2 VINSERTI64X4 VLDDQU VMOVD VMOVDDUP VMOVDQA \
VMOVDQA32 VMOVDQA64 VMOVDQU VMOVDQU16 VMOVDQU32 VMOVDQU64 VMOVDQU8 VMOVNTDQ VMOVNTDQA VMOVQ \
VMOVSD VMOVSHDUP VMOVSLDUP VMPSADBW VPABSB VPABSD VPABSQ VPABSW VPACKSSDW VPACKSSWB VPACKUSDW \
VPACKUSWB VPADDB VPADDD VPADDQ VPADDSB VPADDSW VPADDUSB VPADDUSW VPADDW VPALIGNR VPAND VPANDD \
VPANDN VPANDND VPANDNQ VPANDQ VPAVGB VPAVGW VPBLENDD VPBLENDMB VPBLENDMD VPBLENDMQ VPBLENDMW \
VPBLENDVB VPBLENDW VPBROADCASTB VPBROADCASTD VPBROADCASTMB2Q VPBROADCASTMW2D VPBROADCASTQ \
VPBROADCASTW VPCLMULQDQ VPCMPB VPCMPEQB VPCMPEQD VPCMPEQQ VPCMPEQW VPCMPGTB VPCMPGTD VPCMPGTQ \
VPCMPGTW VPCMPQ VPCMPUB VPCMPUD VPCMPUQ VPCMPUW VPCMPW VPERM2F128 VPERM2I128 VPERMD VPERMI2D \
VPERMI2PD VPERMI2PS VPERMI2Q VPERMI2W VPERMILPD VPERMILPS VPERMPD VPERMPS VPERMQ VPERMT2D \
VPERMT2PD VPERMT2PS VPERMT2Q VPERMT2W VPERMW VPEXTRB VPEXTRD VPEXTRQ VPEXTRW VPHADDD VPHADDSW \
VPHADDW VPHMINPOSUW VPHSUBD VPHSUBSW VPHSUBW VPINSRB VPINSRD VPINSRQ VPINSRW VPLZCNTD VPLZCNTQ \
VPMADD52HUQ VPMADD52LUQ VPMADDUBSW VPMADDWD VPMAXSB VPMAXSQ VPMAXSW VPMAXUB VPMAXUD VPMAXUQ \
VPMAXUW VPMINSB VPMINSQ VPMINSW VPMINUB VPMINUD VPMINUQ VPMINUW VPMOVB2M VPMOVD2M VPMOVDB \
VPMOVDW VPMOVM2B VPMOVM2D VPMOVM2Q VPMOVM2W VPMOVMSKB VPMOVQ2M VPMOVQB VPMOVQD VPMOVQW VPMOVSDB \
VPMOVSDW VPMOVSQB VPMOVSQD VPMOVSQW VPMOVSWB VPMOVSXBD VPMOVSXBQ VPMOVSXBW VPMOVSXDQ VPMOVSXWD \
VPMOVSXWQ VPMOVUSDB VPMOVUSDW VPMOVUSQB VPMOVUSQD VPMOVUSQW VPMOVUSWB VPMOVW2M VPMOVWB VPMOVZXBD \
VPMOVZXBQ VPMOVZXBW VPMOVZXDQ VPMOVZXWD VPMOVZXWQ VPMULDQ VPMULHRSW VPMULHUW VPMULHW VPMULLD \
VPMULLQ VPMULLW VPMULUDQ VPOR VPORD VPORQ VPROLD VPROLQ VPROLVD VPROLVQ VPRORD VPRORQ VPRORVD \
VPRORVQ VPSADBW VPSHLDD VPSHLDQ VPSHLDVD VPSHLDVQ VPSHLDVW VPSHLDW VPSHRDD VPSHRDQ VPSHRDVD \
VPSHRDVQ VPSHRDVW VPSHRDW VPSHUFB VPSHUFD VPSHUFHW VPSHUFLW VPSIGNB VPSIGND VPSIGNW VPSLLD \
VPSLLDQ VPSLLQ VPSLLVD VPSLLVQ VPSLLVW VPSLLW VPSRAD VPSRAQ VPSRAVD VPSRAVQ VPSRAVW VPSRAW \
VPSRLD VPSRLDQ VPSRLQ VPSRLVD VPSRLVQ VPSRLVW VPSRLW VPSUBB VPSUBD VPSUBQ VPSUBSB VPSUBSW \
VPSUBUSB VPSUBUSW VPSUBW VPTERNLOGD VPTERNLOGQ VPTEST VPTESTMB VPTESTMD VPTESTMQ VPTESTMW \
VPTESTNMB VPTESTNMD VPTESTNMQ VPTESTNMW VPUNPCKHBW VPUNPCKHDQ VPUNPCKHQDQ VPUNPCKHWD VPUNPCKLBW \
VPUNPCKLDQ VPUNPCKLQDQ VPUNPCKLWD VPXOR VPXORD VPXORQ VSHUFF32X4 VSHUFF64X2 VSHUFI32X4 \
VSHUFI64X2 VZEROALL VZEROUPPER XOR XORPS";

const DIT: &str = "abs adc adcs add addhn addp adds addv aesd aese aesimc aesmc and ands asr asrv bcax bfc bfi bfm \
bfxil bic bics bif bit bsl ccmn ccmp cinc cinv cls clz cmeq cmge cmgt cmhi cmhs cmle cmlt cmn \
cmp cmtst cneg cnt crc32b crc32cb crc32ch crc32cw crc32cx crc32h crc32w crc32x csel cset csetm \
csinc csinv csneg ctz dup eon eor eor3 ext extr fcsel ins ld1 ld1r ld2 ld2r ld3 ld3r ld4 ld4r \
ldadd ldadda ldaddab ldaddah ldaddal ldaddalb ldaddalh ldaddb ldaddh ldaddl ldaddlb ldaddlh ldap \
ldap1 ldapp ldapr ldaprb ldaprh ldapur ldapurb ldapurh ldapursb ldapursh ldapursw ldar ldarb \
ldarh ldatxr ldaxp ldaxr ldaxrb ldaxrh ldclr ldclra ldclrab ldclrah ldclral ldclralb ldclralh \
ldclrb ldclrh ldclrl ldclrlb ldclrlh ldclrp ldclrpa ldclrpal ldclrpl ldeor ldeora ldeorab \
ldeorah ldeoral ldeoralb ldeoralh ldeorb ldeorh ldeorl ldeorlb ldeorlh ldiapp ldlar ldlarb \
ldlarh ldnp ldp ldpsw ldr ldraa ldrab ldrb ldrh ldrsb ldrsh ldrsw ldset ldseta ldsetab ldsetah \
ldsetal ldsetalb ldsetalh ldsetb ldseth ldsetl ldsetlb ldsetlh ldsetp ldsetpa ldsetpal ldsetpl \
ldsmax ldsmaxa ldsmaxab ldsmaxah ldsmaxal ldsmaxalb ldsmaxalh ldsmaxb ldsmaxh ldsmaxl ldsmaxlb \
ldsmaxlh ldsmin ldsmina ldsminab ldsminah ldsminal ldsminalb ldsminalh ldsminb ldsminh ldsminl \
ldsminlb ldsminlh ldtadd ldtadda ldtaddal ldtaddl ldtclr ldtclra ldtclral ldtclrl ldtnp ldtp \
ldtr ldtrb ldtrh ldtrsb ldtrsh ldtrsw ldtset ldtseta ldtsetal ldtsetl ldtxr ldumax ldumaxa \
ldumaxab ldumaxah ldumaxal ldumaxalb ldumaxalh ldumaxb ldumaxh ldumaxl ldumaxlb ldumaxlh ldumin \
ldumina lduminab lduminah lduminal lduminalb lduminalh lduminb lduminh lduminl lduminlb lduminlh \
ldur ldurb ldurh ldursb ldursh ldursw ldxp ldxr ldxrb ldxrh lsl lslv lsr lsrv luti2 luti4 madd \
mla mls mneg mov msub mul mvn neg negs ngc ngcs not orn orr pmul pmull raddhn rax1 rbit rev \
rev16 rev32 rev64 rmif ror rorv rshrn rsubhn saba sabal sabd sabdl sadalp saddl saddlp saddlv \
saddw sbc sbcs sbfiz sbfm sbfx sdot setf16 setf8 sha1c sha1h sha1m sha1p sha1su0 sha1su1 sha256h \
sha256h2 sha256su0 sha256su1 sha512h sha512h2 sha512su0 sha512su1 shadd shl shll shrn shsub sli \
sm3partw1 sm3partw2 sm3ss1 sm3tt1a sm3tt1b sm3tt2a sm3tt2b sm4e sm4ekey smaddl smax smaxp smaxv \
smin sminp sminv smlal smlsl smmla smnegl smov smsubl smulh smull sqdmulh sqrdmlah sqrdmulh sri \
srshl srshr srsra sshl sshll sshr ssra ssubl ssubw st1 st2 st3 st4 stadd staddb staddh staddl \
staddlb staddlh stclr stclrb stclrh stclrl stclrlb stclrlh steor steorb steorh steorl steorlb \
steorlh stilp stl1 stllr stllrb stllrh stlp stlr stlrb stlrh stltxr stlur stlurb stlurh stlxp \
stlxr stlxrb stlxrh stnp stp str strb strh stset stsetb stseth stsetl stsetlb stsetlh stsmax \
stsmaxb stsmaxh stsmaxl stsmaxlb stsmaxlh stsmin stsminb stsminh stsminl stsminlb stsminlh \
sttadd sttaddl sttclr sttclrl sttnp sttp sttr sttrb sttrh sttset sttsetl sttxr stumax stumaxb \
stumaxh stumaxl stumaxlb stumaxlh stumin stuminb stuminh stuminl stuminlb stuminlh stur sturb \
sturh stxp stxr stxrb stxrh sub subhn subs sudot swpp swppa swppal swppl sxtb sxth sxtl sxtw tbl \
tbx trn1 trn2 tst uaba uabal uabd uabdl uadalp uaddl uaddlp uaddlv uaddw ubfiz ubfm ubfx udot \
uhadd uhsub umaddl umax umaxp umaxv umin uminp uminv umlal umlsl ummla umnegl umov umsubl umulh \
umull urshl urshr ursra usdot ushl ushll ushr usmmla usra usubl usubw uxtb uxth uxtl uzp1 uzp2 \
xar xtn zip1 zip2";

const DIT_SVE: &str = "abs adclb adclt add addhnb addhnt addp addpl addqp addqv addsubp addvl adr aesd aesdimc aese \
aesemc aesimc aesmc and andqv ands andv asr asrd asrr bcax bdep bext bgrp bic bics bsl bsl1n \
bsl2n cadd cls clz cmla cmpeq cmpge cmpgt cmphi cmphs cmple cmplo cmpls cmplt cmpne cnot cnt \
ctermeq ctermne decb decd dech decw dupq eon eor eor3 eorbt eorqv eors eortb eorv ext extq incb \
incd inch incw insr ld1b ld1d ld1h ld1q ld1rb ld1rd ld1rh ld1rob ld1rod ld1roh ld1row ld1rqb \
ld1rqd ld1rqh ld1rqw ld1rsb ld1rsh ld1rsw ld1rw ld1sb ld1sh ld1sw ld1w ld2b ld2d ld2h ld2q ld2w \
ld3b ld3d ld3h ld3q ld3w ld4b ld4d ld4h ld4q ld4w ldnt1b ldnt1d ldnt1h ldnt1sb ldnt1sh ldnt1sw \
ldnt1w ldr lsl lslr lsr lsrr luti2 luti4 luti6 mad mla mls movprfx movs msb mul nand nands nbsl \
neg nor nors not nots orn orns orqv orr orrs orv pmov pmul pmullb pmullt psel ptest raddhnb \
raddhnt rax1 rbit revb revd revh revw rshrnb rshrnt rsubhnb rsubhnt saba sabal sabalb sabalt \
sabd sabdlb sabdlt sadalp saddlb saddlbt saddlt saddv saddwb saddwt sbclb sbclt sclamp sdot sel \
shadd shrnb shrnt shsub shsubr sli sm4e sm4ekey smax smaxp smaxqv smaxv smin sminp sminqv sminv \
smlalb smlalt smlslb smlslt smmla smulh smullb smullt sqdmulh sqrdmlah sqrdmulh sri srshl srshr \
srsra sshllb sshllt ssra ssublb ssublbt ssublt ssubltb ssubwb ssubwt st1b st1d st1h st1q st1w \
st2b st2d st2h st2q st2w st3b st3d st3h st3q st3w st4b st4d st4h st4q st4w stnt1b stnt1d stnt1h \
stnt1w str sub subhnb subhnt subp subr sudot sunpkhi sunpklo sxtb sxth sxtw tbl tblq tbx tbxq \
uaba uabal uabalb uabalt uabd uabdlb uabdlt uadalp uaddlb uaddlt uaddv uaddwb uaddwt uclamp udot \
uhadd uhsub uhsubr umax umaxp umaxqv umaxv umin uminp uminqv uminv umlalb umlalt umlslb umlslt \
ummla umulh umullb umullt urshl urshr ursra usdot ushllb ushllt usmmla usra usublb usublt usubwb \
usubwt uunpkhi uunpklo uxtb uxth uxtw uzpq1 uzpq2 xar zipq1 zipq2";
