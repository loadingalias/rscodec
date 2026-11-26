//! The scan of the disassembly. From each entry of the harness, it follows each direct call and
//! jump to the code that the entry runs. In that code it finds:
//!
//! - a load from a constant table with an index register: the form of a lookup table;
//! - an indirect jump or call whose target comes from the data, and in the code of a secret entry,
//!   an indirect call that the scan cannot follow;
//! - a call out of the binary, as to `memcpy@plt`;
//! - a load or store whose address comes from the data;
//! - in the code of a secret entry, an instruction with a data operand that is not on the DOIT list
//!   of Intel or the DIT list of Arm;
//! - in the code of a secret entry, a write of `PSTATE.DIT`.
//!
//! The data is each value of a load of memory other than the stack, constant data, and the
//! environment of a closure. It flows through the general, vector, and mask registers, the flags,
//! and the slots of the stack, over the control flow of each function, to a fixed point. A call
//! ends what the scan knows of the registers: each function of the closure has its own scan.
//!
//! On x86_64, a call to a function of another crate can load its target from a slot of the global
//! offset table (GOT). The relocation of the slot in `objdump -R` gives the function, so the scan
//! follows the call as a direct call.
//!
//! A compare with the padding `=` gives public flags: the padding gives the length of the bytes,
//! which is public.
//!
//! In the whole binary, each write that sets `PSTATE.DIT` must be followed by `sb`, or by
//! `dsb nsh` and `isb`, on each path before the next call or return. The scan reads the
//! output of `objdump -d`, of GNU binutils or of LLVM.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    env,
    path::{Path, PathBuf},
    process::Command,
};

use crate::{Failure, lists};

/// The prefix of the names of the entries of the harness.
const ENTRY: &str = "rscodec_ct::cases::entry::";

/// The entries of the harness.
const ENTRIES: usize = 19;

/// The code that the scan does not follow: the panics of `core`, which a length check reaches and
/// which do not return. Each is reported.
const PANICS: [&str; 3] = ["core::panicking::", "core::slice::index::", "core::option::"];

/// The code of the secret entries that handles only public data, and the reason.
const PUBLIC: [(&str, &str); 1] = [(
    "rscodec::cpu::",
    "the detection of the CPU reads only the state of the CPU",
)];

/// The instructions with no data operand: control, hints, barriers, and moves of an immediate.
const NO_DATA_X86: &str = "JMP CALL RET NOP ENDBR64 UD2 INT3 LFENCE";
const NO_DATA_A64: &str = "b bl ret cbz cbnz tbz tbnz nop hint bti paciasp autiasp pacibsp autibsp adr adrp movz \
                           movn movk movi mvni brk udf sb dsb isb";

/// The reductions of A64 across the lanes of a vector: the validity of a text, and the first bad
/// offset of a refusal, which are results of the call.
const REDUCTIONS: [&str; 9] = [
    "umaxv", "uminv", "smaxv", "sminv", "addv", "uaddlv", "saddlv", "umaxp", "uminp",
];

/// The instructions that set the flags from their operands, and the instructions that read them.
const SETS_FLAGS_X86: &str = "cmp test add sub and or xor adc sbb inc dec neg shl shr sar rol ror bt bts btr btc \
                              ptest vptest bsf bsr tzcnt lzcnt popcnt imul mul comiss comisd ucomiss ucomisd \
                              kortestb kortestw kortestd kortestq ktestb ktestw ktestd ktestq";
const SETS_FLAGS_A64: &str = "cmp cmn tst subs adds ands bics ccmp ccmn fcmp fcmpe adcs sbcs negs ngcs";
const READS_FLAGS_X86: &str = "adc sbb rcl rcr";
const READS_FLAGS_A64: &str = "csel csinc csinv csneg cset csetm cinc cinv cneg ccmp ccmn adc adcs sbc sbcs ngc ngcs";

/// One instruction of the listing.
#[derive(Debug)]
struct Instruction {
    address: u64,
    mnemonic: String,
    operands: String,
}

/// The instruction sets of the scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Isa {
    X86,
    A64,
}

/// Whether the code of `name` handles only public data.
fn public(name: &str) -> bool {
    PUBLIC.iter().any(|(prefix, _)| name.starts_with(prefix))
}

/// Scans the disassembly of `args[0]`, or of this binary, of the instruction set `args[1]`
/// (`x86_64` or `aarch64`), or of this host. With `args[2]`, it also prints the facts before each
/// instruction of the function of that name, for the review of a finding.
pub(crate) fn run(args: &[String]) -> Result<(), Failure> {
    let binary = match args.first() {
        Some(path) => PathBuf::from(path),
        None => env::current_exe().map_err(|error| Failure::Host(format!("the path of this binary: {error}")))?,
    };
    let isa = match args.get(1).map(String::as_str) {
        Some("x86_64") => Isa::X86,
        Some("aarch64") => Isa::A64,
        _ if cfg!(target_arch = "x86_64") => Isa::X86,
        _ => Isa::A64,
    };
    let mut command = Command::new("objdump");
    command.args(["-d", "--no-show-raw-insn", "-C"]);
    if isa == Isa::X86 {
        command.args(["-M", "intel"]);
    }
    let output = command
        .arg(&binary)
        .output()
        .map_err(|error| Failure::Host(format!("objdump: {error}")))?;
    if !output.status.success() {
        return Err(Failure::Host(format!(
            "objdump: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let slots = match isa {
        Isa::X86 => got(&binary)?,
        Isa::A64 => BTreeMap::new(),
    };
    let functions = parse(isa, &String::from_utf8_lossy(&output.stdout), &slots);
    let entries: Vec<&String> = functions.keys().filter(|name| name.starts_with(ENTRY)).collect();
    if entries.len() != ENTRIES {
        return Err(Failure::Host(format!(
            "found {} entries of the harness, not {ENTRIES}",
            entries.len()
        )));
    }
    if let Some(traced) = args.get(2)
        && let Some(function) = functions.get(traced)
    {
        trace(isa, traced, function);
    }
    let mut findings = BTreeSet::new();
    let mut panics = BTreeSet::new();
    let mut secret_code = BTreeSet::new();
    for &entry in &entries {
        let secret = entry.ends_with("_secret");
        let (code, external) = closure(isa, &functions, entry, &mut panics);
        let instructions: usize = code.iter().filter_map(|name| functions.get(name)).map(Vec::len).sum();
        println!("scan {entry}: {} functions, {instructions} instructions", code.len());
        // `memset` writes one byte value, and its time depends only on the length. Another call out
        // of the binary is a finding in the secret code: the scan cannot read its code.
        for (caller, callee) in external {
            if callee.starts_with("memset") || !secret {
                findings.insert(format!("note: {caller} calls {callee}, outside the code of the binary"));
            } else {
                findings.insert(format!("{caller} calls {callee}, outside the code of the binary"));
            }
        }
        for name in &code {
            let Some(function) = functions.get(name) else { continue };
            let secret = secret && !public(name);
            addresses(isa, name, function, secret, &mut findings);
            if secret {
                secret_code.extend(function.iter().map(|code| code.mnemonic.clone()));
                if isa == Isa::A64 && function.iter().any(writes_dit) {
                    findings.insert(format!("{name}: writes PSTATE.DIT in the code of a secret entry"));
                }
            }
        }
    }
    if isa == Isa::A64 {
        let sets = barriers(&functions, &mut findings);
        println!("scan: {sets} writes that set PSTATE.DIT, each followed by its barrier unless reported");
        if sets == 0 {
            findings.insert("no write that sets PSTATE.DIT: the barrier check found nothing to check".to_owned());
        }
    }
    for (prefix, reason) in PUBLIC {
        println!("scan: the code of {prefix} has no list check: {reason}");
    }
    for panic in &panics {
        println!("scan: the secret or public code can call {panic}, which does not return");
    }
    println!(
        "scan: the mnemonics of the secret code: {}",
        secret_code.into_iter().collect::<Vec<_>>().join(" ")
    );
    let (notes, findings): (BTreeSet<_>, BTreeSet<_>) =
        findings.into_iter().partition(|line| line.starts_with("note: "));
    for note in &notes {
        println!("scan {note}");
    }
    for finding in &findings {
        println!("scan finding: {finding}");
    }
    println!("scan: {} entries, {} findings", entries.len(), findings.len());
    if findings.is_empty() {
        Ok(())
    } else {
        Err(Failure::Check(format!("{} findings of the scan", findings.len())))
    }
}

/// The function address in each slot of the GOT of `binary`, from its relocations of the form
/// `R_X86_64_RELATIVE`. A binary with no dynamic relocations, which `objdump -R` refuses, has none.
fn got(binary: &Path) -> Result<BTreeMap<u64, u64>, Failure> {
    let output = Command::new("objdump")
        .arg("-R")
        .arg(binary)
        .output()
        .map_err(|error| Failure::Host(format!("objdump -R: {error}")))?;
    // A slot of a binary with no dynamic relocations stays unknown, so each call through it in the
    // secret code is a finding.
    let listing = String::from_utf8_lossy(&output.stdout);
    Ok(listing
        .lines()
        .filter_map(|line| {
            let mut words = line.split_whitespace();
            let slot = u64::from_str_radix(words.next()?, 16).ok()?;
            let addend = (words.next()? == "R_X86_64_RELATIVE")
                .then(|| words.next()?.strip_prefix("*ABS*+0x"))
                .flatten()?;
            Some((slot, u64::from_str_radix(addend, 16).ok()?))
        })
        .collect())
}

/// The address and the name of the function that a line of `objdump -d` starts, as
/// `000000000006be00 <name>:`.
fn header(line: &str) -> Option<(u64, &str)> {
    let (address, name) = line.strip_suffix(">:")?.split_once(" <")?;
    Some((u64::from_str_radix(address, 16).ok()?, name))
}

/// The functions of a listing of `objdump -d`, by name. A call or jump through a slot of `slots`
/// becomes a direct call or jump to the function that the slot holds.
fn parse(isa: Isa, listing: &str, slots: &BTreeMap<u64, u64>) -> BTreeMap<String, Vec<Instruction>> {
    let starts: BTreeMap<u64, String> = listing
        .lines()
        .filter_map(header)
        .map(|(address, name)| (address, symbol(name)))
        .collect();
    // `#` starts a comment of x86 and an immediate of A64, whose comments start with `//` (GNU) or
    // `;` (LLVM).
    let comment: &[&str] = match isa {
        Isa::X86 => &["#"],
        Isa::A64 => &["//", ";"],
    };
    let mut functions = BTreeMap::new();
    let mut current: Option<(String, Vec<Instruction>)> = None;
    for line in listing.lines() {
        if let Some((_, name)) = header(line) {
            if let Some((name, code)) = current.take() {
                functions.insert(name, code);
            }
            current = Some((symbol(name), Vec::new()));
            continue;
        }
        let Some((address, text)) = line.trim_start().split_once(':') else {
            continue;
        };
        let Ok(address) = u64::from_str_radix(address.trim_start_matches("0x"), 16) else {
            continue;
        };
        // A call through the GOT names its slot in the comment, as `# 4ff0 <memcpy@GLIBC_2.14>`
        // (GNU) or `# 0xd14b0 <...>` (LLVM), and the name of an import there.
        let slot_comment = text.split_once('#').map(|(_, comment)| comment.trim());
        let import = slot_comment
            .and_then(|comment| comment.find('<').map(|at| comment.get(at..).unwrap_or_default().trim()))
            .filter(|name| name.contains('@'));
        let held = slot_comment
            .and_then(|comment| comment.split_whitespace().next())
            .and_then(|slot| u64::from_str_radix(slot.trim_start_matches("0x"), 16).ok())
            .and_then(|slot| slots.get(&slot))
            .and_then(|target| Some((*target, starts.get(target)?)));
        let text = comment
            .iter()
            .fold(text, |text, comment| text.split(comment).next().unwrap_or_default())
            .trim();
        let (mnemonic, operands) = text.split_once(char::is_whitespace).unwrap_or((text, ""));
        if mnemonic.is_empty() || mnemonic == "..." {
            continue;
        }
        // The objdump of Apple writes the arrangement of A64 on the mnemonic, as `tbl.16b`.
        let mnemonic = match mnemonic.split_once('.') {
            Some((stem, _)) if isa == Isa::A64 && stem != "b" => stem,
            _ => mnemonic,
        };
        let through_got = isa == Isa::X86 && (mnemonic == "call" || mnemonic == "jmp") && operands.contains("[rip");
        let operands = match (import, held) {
            (Some(import), _) if through_got => format!("{} {import}", operands.trim()),
            (None, Some((target, name))) if through_got => format!("{target:x} <{name}>"),
            _ => operands.trim().to_owned(),
        };
        if let Some((_, code)) = &mut current {
            code.push(Instruction {
                address,
                mnemonic: mnemonic.to_ascii_lowercase(),
                operands,
            });
        }
    }
    if let Some((name, code)) = current {
        functions.insert(name, code);
    }
    functions
}

/// A demangled name without its hash and without the underscore of Mach-O.
fn symbol(name: &str) -> String {
    let name = name
        .strip_prefix('_')
        .filter(|rest| rest.contains("::"))
        .unwrap_or(name);
    let name = match name.rsplit_once("::h") {
        Some((stem, hash)) if hash.len() == 16 && hash.chars().all(|c| c.is_ascii_hexdigit()) => stem,
        _ => name,
    };
    name.to_owned()
}

/// The function that an operand names, as in `4a3b0 <name+0x10>`.
fn target(operands: &str) -> Option<String> {
    let start = operands.find('<')?;
    let end = operands.rfind('>')?;
    let name = operands.get(start + 1..end)?;
    let name = match name.rsplit_once('+') {
        Some((stem, offset)) if offset.starts_with("0x") => stem,
        _ => name,
    };
    Some(symbol(name))
}

/// Whether the instruction is a direct branch, a call, or a jump.
fn branches(isa: Isa, mnemonic: &str) -> bool {
    match isa {
        Isa::X86 => mnemonic == "call" || mnemonic.starts_with('j'),
        Isa::A64 => ["b", "bl", "cbz", "cbnz", "tbz", "tbnz"].contains(&mnemonic) || mnemonic.starts_with("b."),
    }
}

/// Whether the operands of a branch end with its address, as `4a3b0 <name>`.
fn direct(operands: &str) -> bool {
    let last = operands.rsplit(',').next().unwrap_or_default();
    last.split_once('<')
        .is_some_and(|(address, _)| u64::from_str_radix(address.trim().trim_start_matches("0x"), 16).is_ok())
}

/// The names of the functions that `entry` runs, by each direct call and jump to another function,
/// and each call of a caller out of the code of the binary.
fn closure(
    isa: Isa,
    functions: &BTreeMap<String, Vec<Instruction>>,
    entry: &str,
    panics: &mut BTreeSet<String>,
) -> (BTreeSet<String>, BTreeSet<(String, String)>) {
    let mut code = BTreeSet::from([entry.to_owned()]);
    let mut external = BTreeSet::new();
    let mut queue = VecDeque::from([entry.to_owned()]);
    while let Some(name) = queue.pop_front() {
        for instruction in functions.get(&name).into_iter().flatten() {
            if !branches(isa, &instruction.mnemonic) {
                continue;
            }
            let Some(callee) = target(&instruction.operands) else {
                continue;
            };
            if callee == name || code.contains(&callee) {
                continue;
            }
            if PANICS.iter().any(|prefix| callee.starts_with(prefix)) {
                panics.insert(callee);
                continue;
            }
            if callee.contains('@') || !functions.contains_key(&callee) {
                external.insert((name.clone(), callee));
                continue;
            }
            code.insert(callee.clone());
            queue.push_back(callee);
        }
    }
    (code, external)
}

/// The registers inside the first `[...]` of `operands`: the base, and the index if it has one.
fn address(isa: Isa, operands: &str) -> Option<(Option<String>, Option<String>)> {
    let inside = operands.get(operands.find('[')? + 1..operands.find(']')?)?;
    match isa {
        Isa::X86 => {
            let mut base = None;
            let mut index = None;
            for term in inside.split(['+', '-']).map(str::trim) {
                let (register, scaled) = match term.split_once('*') {
                    Some((register, _)) => (register.trim(), true),
                    None => (term, false),
                };
                if !register.starts_with(|c: char| c.is_ascii_alphabetic()) {
                    continue;
                }
                if scaled || base.is_some() {
                    index = Some(family(register));
                } else {
                    base = Some(family(register));
                }
            }
            Some((base, index))
        }
        Isa::A64 => {
            let mut parts = inside.split(',').map(str::trim);
            let base = parts.next().map(family);
            let index = parts
                .next()
                .filter(|part| part.starts_with(['x', 'w', 'z']))
                .map(family);
            Some((base, index))
        }
    }
}

/// The 64-bit register of a general register name: `eax` and `al` are `rax`, `w8` is `x8`.
fn family(register: &str) -> String {
    let register = register.trim().to_ascii_lowercase();
    let x86 = [
        ("rax", ["eax", "ax", "al"]),
        ("rbx", ["ebx", "bx", "bl"]),
        ("rcx", ["ecx", "cx", "cl"]),
        ("rdx", ["edx", "dx", "dl"]),
        ("rsi", ["esi", "si", "sil"]),
        ("rdi", ["edi", "di", "dil"]),
        ("rbp", ["ebp", "bp", "bpl"]),
        ("rsp", ["esp", "sp", "spl"]),
    ];
    for (wide, narrow) in x86 {
        if narrow.contains(&register.as_str()) {
            return wide.to_owned();
        }
    }
    if let Some(number) = register.strip_prefix('r')
        && let Some(digits) = number.strip_suffix(['d', 'w', 'b'])
        && digits.chars().all(|c| c.is_ascii_digit())
    {
        return format!("r{digits}");
    }
    if let Some(number) = register.strip_prefix('w')
        && number.chars().all(|c| c.is_ascii_digit())
    {
        return format!("x{number}");
    }
    register
}

/// Whether a register name is a general register of the scan.
fn general(isa: Isa, register: &str) -> bool {
    let family = family(register);
    let numbered = |prefix: char| {
        family
            .strip_prefix(prefix)
            .is_some_and(|digits| !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()))
    };
    match isa {
        Isa::X86 => ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp"].contains(&family.as_str()) || numbered('r'),
        Isa::A64 => numbered('x'),
    }
}

/// The vector, mask, or predicate register of a name: `v3` for `xmm3`, `zmm3`, `s3`, `v3.16b`, or
/// `z3.b`, and `k1` or `p1` for a mask or a predicate.
fn lanes(isa: Isa, register: &str) -> Option<String> {
    let register = register.trim().to_ascii_lowercase();
    let register = register.split(['.', '[', '/']).next()?;
    let number = |prefix: &str| {
        register
            .strip_prefix(prefix)
            .filter(|digits| !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()))
            .map(str::to_owned)
    };
    match isa {
        Isa::X86 => ["xmm", "ymm", "zmm"]
            .iter()
            .find_map(|prefix| number(prefix))
            .map(|digits| format!("v{digits}"))
            .or_else(|| number("k").map(|digits| format!("k{digits}"))),
        Isa::A64 => ["v", "q", "d", "s", "h", "b", "z"]
            .iter()
            .find_map(|prefix| number(prefix))
            .map(|digits| format!("v{digits}"))
            .or_else(|| number("p").map(|digits| format!("p{digits}"))),
    }
}

/// The operands outside the brackets of a memory operand, split at commas and spaces. The registers
/// of a list in braces, as `{ v0.16b, v1.16b }` or the mask `{k1}`, are operands too.
fn words(operands: &str) -> Vec<&str> {
    let mut words = Vec::new();
    let mut depth = 0_u32;
    let mut start = 0;
    for (at, c) in operands.char_indices() {
        match c {
            '[' => {
                if depth == 0 {
                    words.extend(operands.get(start..at));
                }
                depth += 1;
            }
            ']' => {
                depth = depth.saturating_sub(1);
                start = at + 1;
            }
            ',' | ' ' | '{' | '}' if depth == 0 => {
                words.extend(operands.get(start..at));
                start = at + 1;
            }
            _ => {}
        }
    }
    words.extend(operands.get(start..));
    words
        .into_iter()
        .map(str::trim)
        .filter(|word| !word.is_empty())
        .collect()
}

/// The facts of the scan at one point of a function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct State {
    /// The registers that hold the address of constant data, from `lea reg, [rip+...]` or `adrp`.
    constant: BTreeSet<String>,
    /// The general registers that hold data.
    data: BTreeSet<String>,
    /// The vector, mask, and predicate registers that hold no data.
    clean: BTreeSet<String>,
    /// Whether the flags come from data.
    flags: bool,
    /// The slots of the stack that hold data, by their operand.
    slots: BTreeSet<String>,
    /// The data of each value on the stack of x86 `push`, the last on top.
    pushed: Vec<bool>,
    /// The register that holds the environment of a closure, whose general loads give pointers.
    environment: Option<String>,
    /// The vector registers that a reduction across the lanes wrote last.
    reduced: BTreeSet<String>,
}

impl State {
    /// The facts at a join of two paths: data from either path, the facts of no data from both.
    fn join(&mut self, other: &Self) -> bool {
        let before = self.clone();
        self.constant = self.constant.intersection(&other.constant).cloned().collect();
        self.data.extend(other.data.iter().cloned());
        self.clean = self.clean.intersection(&other.clean).cloned().collect();
        self.flags |= other.flags;
        self.slots.extend(other.slots.iter().cloned());
        self.reduced = self.reduced.intersection(&other.reduced).cloned().collect();
        let depth = self.pushed.len().min(other.pushed.len());
        self.pushed.truncate(depth);
        for (mine, theirs) in self.pushed.iter_mut().zip(&other.pushed) {
            *mine |= *theirs;
        }
        if self.environment != other.environment {
            self.environment = None;
        }
        *self != before
    }

    /// Whether the register of `word` holds data. A word that is no register holds none.
    fn holds(&self, isa: Isa, word: &str) -> bool {
        if let Some(lanes) = lanes(isa, word) {
            return !self.clean.contains(&lanes);
        }
        self.data.contains(&family(word))
    }
}

/// Prints the facts before each instruction of `function`.
fn trace(isa: Isa, name: &str, function: &[Instruction]) {
    for (state, code) in states(isa, name, function, true).iter().zip(function) {
        let Some(state) = state else {
            println!(
                "trace {:x}: unreached {} {}",
                code.address, code.mnemonic, code.operands
            );
            continue;
        };
        let data: Vec<_> = state.data.iter().map(String::as_str).collect();
        println!(
            "trace {:x}: {} {} | data {} flags {}",
            code.address,
            code.mnemonic,
            code.operands,
            data.join(","),
            state.flags
        );
    }
}

/// Records each finding of the dataflow of a function. See the module for the rules.
fn addresses(isa: Isa, name: &str, function: &[Instruction], secret: bool, findings: &mut BTreeSet<String>) {
    for (state, code) in states(isa, name, function, secret).iter().zip(function) {
        if let Some(state) = state {
            step(isa, name, code, secret, &mut state.clone(), findings);
        }
    }
}

/// The facts before each instruction of `function` at the fixed point, or `None` for an
/// instruction that no path reaches.
fn states(isa: Isa, name: &str, function: &[Instruction], secret: bool) -> Vec<Option<State>> {
    let index_of: BTreeMap<u64, usize> = function
        .iter()
        .enumerate()
        .map(|(at, code)| (code.address, at))
        .collect();
    // The targets of the direct branches in the function.
    let targets: BTreeSet<usize> = function
        .iter()
        .filter(|code| branches(isa, &code.mnemonic) && code.mnemonic != "call" && code.mnemonic != "bl")
        .filter_map(|code| {
            let destination = code.operands.rsplit(',').next()?.split_whitespace().next()?;
            let destination = u64::from_str_radix(destination.trim_start_matches("0x"), 16).ok()?;
            index_of.get(&destination).copied()
        })
        .collect();
    // The instructions that only an indirect jump can reach: after an unconditional transfer, and
    // the target of no direct branch.
    let unreached: Vec<usize> = function
        .windows(2)
        .enumerate()
        .filter(|(_, pair)| {
            matches!(
                pair[0].mnemonic.as_str(),
                "jmp" | "b" | "ret" | "br" | "ud2" | "brk" | "udf"
            )
        })
        .map(|(at, _)| at + 1)
        .filter(|at| !targets.contains(at))
        .collect();
    // The successors of each instruction in the function.
    let successors = |at: usize| -> Vec<usize> {
        let Some(code) = function.get(at) else {
            return Vec::new();
        };
        let mnemonic = code.mnemonic.as_str();
        let mut next = Vec::new();
        if branches(isa, mnemonic) && mnemonic != "call" && mnemonic != "bl" {
            let destination = code.operands.rsplit(',').next().unwrap_or_default();
            let destination = destination
                .split_whitespace()
                .next()
                .unwrap_or_default()
                .trim_start_matches("0x");
            if let Ok(destination) = u64::from_str_radix(destination, 16)
                && let Some(&target) = index_of.get(&destination)
            {
                next.push(target);
            }
        }
        if !matches!(mnemonic, "jmp" | "b" | "ret" | "br" | "ud2" | "brk" | "udf") {
            next.push(at + 1);
        }
        // An indirect jump, as of a jump table, reaches each instruction after an unconditional
        // transfer: the start of each block that only a jump can reach.
        if (mnemonic == "jmp" && !direct(&code.operands)) || mnemonic == "br" {
            next.extend(unreached.iter().copied());
        }
        next
    };
    let mut start = State::default();
    if name.contains("{closure#") {
        start.environment = Some(match isa {
            Isa::X86 => "rdi".to_owned(),
            Isa::A64 => "x0".to_owned(),
        });
    }
    let mut states: Vec<Option<State>> = vec![None; function.len()];
    if let Some(first) = states.first_mut() {
        *first = Some(start);
    }
    let mut work = vec![0_usize];
    let mut ignored = BTreeSet::new();
    while let Some(at) = work.pop() {
        let (Some(Some(state)), Some(code)) = (states.get(at), function.get(at)) else {
            continue;
        };
        let mut state = state.clone();
        step(isa, name, code, secret, &mut state, &mut ignored);
        for next in successors(at) {
            let Some(slot) = states.get_mut(next) else { continue };
            let changed = match slot {
                Some(known) => known.join(&state),
                None => {
                    *slot = Some(state.clone());
                    true
                }
            };
            if changed {
                work.push(next);
            }
        }
    }
    states
}

/// Whether the secret code may run `mnemonic`, from the list of its owner or as an instruction with
/// no data operand by its kind.
fn listed(isa: Isa, mnemonic: &str, sve: bool) -> bool {
    match isa {
        Isa::X86 => {
            lists::doit(mnemonic)
                || mnemonic.starts_with('J')
                || NO_DATA_X86.split_ascii_whitespace().any(|no_data| no_data == mnemonic)
        }
        Isa::A64 if sve => lists::dit_sve(mnemonic),
        Isa::A64 => {
            // `uaddw2` is the form of `uaddw` for the upper half, in the same file of Arm.
            let stem = mnemonic.strip_suffix('2').unwrap_or(mnemonic);
            lists::dit(mnemonic)
                || lists::dit(stem)
                || mnemonic.starts_with("b.")
                || NO_DATA_A64.split_ascii_whitespace().any(|no_data| no_data == mnemonic)
        }
    }
}

/// The facts after one instruction, and its findings.
fn step(
    isa: Isa,
    name: &str,
    instruction: &Instruction,
    secret: bool,
    state: &mut State,
    findings: &mut BTreeSet<String>,
) {
    let stack = |base: &str| ["rsp", "rbp", "sp", "x29"].contains(&base);
    let mnemonic = instruction.mnemonic.as_str();
    let operands = instruction.operands.as_str();
    let imported = target(operands).is_some_and(|callee| callee.contains('@'));
    let indirect = match isa {
        Isa::X86 => (mnemonic == "jmp" || mnemonic == "call") && !direct(operands) && !imported,
        Isa::A64 => mnemonic.starts_with("br") || mnemonic.starts_with("blr"),
    };
    let lea = mnemonic == "lea";
    // A multi-byte `nop`, as `data16 cs nop WORD PTR [rax+rax*1+0x0]`, accesses no memory.
    if isa == Isa::X86 && doit_name(instruction) == "NOP" {
        return;
    }
    let memory = address(isa, operands);
    let slot = operands
        .find('[')
        .zip(operands.find(']'))
        .and_then(|(open, close)| operands.get(open..=close))
        .map(|text| text.replace(' ', ""));
    let all = words(operands);
    let first = all.first().copied().unwrap_or_default();
    let stores = match isa {
        Isa::X86 => operands
            .trim_start()
            .split(',')
            .next()
            .is_some_and(|operand| operand.contains('[')),
        Isa::A64 => mnemonic.starts_with("st"),
    };
    if indirect {
        let target_data = all.iter().any(|word| state.holds(isa, word))
            || memory
                .as_ref()
                .is_some_and(|(base, index)| base.iter().chain(index).any(|register| state.data.contains(register)));
        if target_data {
            findings.insert(format!(
                "{name}: an indirect {mnemonic} to a target from the data ({operands})"
            ));
        } else if secret && (mnemonic == "call" || mnemonic.starts_with("blr")) {
            findings.insert(format!(
                "{name}: an indirect {mnemonic} that the scan cannot follow ({operands})"
            ));
        } else {
            findings.insert(format!(
                "note: {name}: an indirect {mnemonic} with no data in its target"
            ));
        }
    }
    let mut loaded = false;
    if let Some((base, index)) = &memory {
        // The address of a table can be in either register: GNU writes `[rcx+rax*1]` for the
        // table in `rax` and the index in `rcx`.
        let constant_base = base
            .iter()
            .chain(index)
            .any(|register| state.constant.contains(register) || register == "rip");
        let absolute = isa == Isa::X86 && base.is_none() && index.is_some();
        let from_data = base.iter().chain(index).any(|register| state.data.contains(register));
        if lea {
            // `lea` computes an address and reads no memory.
        } else if from_data {
            findings.insert(format!("{name}: an address from the data, {mnemonic} {operands}"));
        } else if index.is_some() && (constant_base || absolute) {
            findings.insert(format!(
                "note: {name}: a load from a constant table with no data in its index"
            ));
        }
        let environment = base.is_some() && *base == state.environment && general(isa, first);
        loaded = !stores
            && if lea {
                from_data
            } else if base.as_deref().is_some_and(stack) {
                slot.as_ref().is_some_and(|slot| state.slots.contains(slot))
            } else {
                !constant_base && !environment && !["prfm", "nop"].contains(&mnemonic)
            };
    }
    // The destination and the sources. A store writes no register. x86 writes its first operand,
    // and reads it too unless the instruction only writes it.
    let destination = (!stores)
        .then(|| written(isa, instruction))
        .flatten()
        .filter(|_| lanes(isa, first).is_some() || general(isa, first));
    let keeps = match isa {
        Isa::X86 => {
            !["mov", "movzx", "movsx", "movsxd", "movabs", "lea", "pop"].contains(&mnemonic)
                && !mnemonic.starts_with("set")
                && !mnemonic.starts_with("vmov")
                && !mnemonic.starts_with("vpbroadcast")
                && (all.len() < 3 || (operands.contains("{k") && !operands.contains("{z}")))
        }
        // A64 overwrites its destination, except the inserts and the accumulates.
        Isa::A64 => {
            [
                "movk", "bfi", "bfxil", "bfm", "bfc", "ins", "mla", "mls", "bsl", "bif", "bit", "sli", "sri", "tbx",
            ]
            .contains(&mnemonic)
                || mnemonic.ends_with("ra")
                || (mnemonic == "mov" && first.contains('['))
                || operands.contains("/m")
        }
    };
    let skip = usize::from(destination.is_some() && !keeps);
    let zeroing = all.len() >= 2
        && all.iter().skip(1).all(|word| *word == first)
        && [
            "xor", "sub", "pxor", "vpxor", "vpxord", "vpxorq", "xorps", "vxorps", "eor", "psubb", "vpsubb",
        ]
        .contains(&mnemonic);
    let reads_flags = match isa {
        Isa::X86 => {
            mnemonic.starts_with("cmov")
                || mnemonic.starts_with("set")
                || READS_FLAGS_X86.split(' ').any(|m| m == mnemonic)
        }
        Isa::A64 => READS_FLAGS_A64.split_ascii_whitespace().any(|m| m == mnemonic),
    };
    let from_data = !zeroing
        && ((lea && loaded)
            || (!lea && loaded)
            || (reads_flags && state.flags)
            || all.iter().skip(skip).any(|word| state.holds(isa, word))
            || (isa == Isa::X86 && mnemonic == "pop" && state.pushed.last().copied().unwrap_or(false)));
    // The list of the owner, for the secret code.
    if secret {
        let sve = isa == Isa::A64
            && all
                .iter()
                .any(|word| word.starts_with(['z', 'p']) && lanes(isa, word).is_some());
        let name_of = match isa {
            Isa::X86 => doit_name(instruction),
            Isa::A64 => mnemonic.to_owned(),
        };
        if !listed(isa, &name_of, sve) {
            let source = all.get(1).copied().unwrap_or_default();
            let reduction = isa == Isa::A64
                && mnemonic == "fmov"
                && lanes(isa, source).is_some_and(|vector| state.reduced.contains(&vector));
            if reduction {
                findings.insert(format!("note: {name}: {mnemonic} {operands}, of a reduction"));
            } else if from_data {
                findings.insert(format!(
                    "{name}: {name_of} is not on the list and has a data operand ({mnemonic} {operands})"
                ));
            } else {
                findings.insert(format!(
                    "note: {name}: {name_of} is not on the list and has no data operand"
                ));
            }
        }
    }
    // The facts after the instruction.
    if stores
        && let (Some(slot), Some((Some(base), _))) = (&slot, &memory)
        && stack(base)
    {
        if all.iter().any(|word| state.holds(isa, word)) {
            state.slots.insert(slot.clone());
        } else {
            state.slots.remove(slot);
        }
    }
    if isa == Isa::X86 && mnemonic == "push" {
        state.pushed.push(state.holds(isa, first));
    }
    if isa == Isa::X86 && mnemonic == "pop" {
        state.pushed.pop();
    }
    let sets_flags = match isa {
        Isa::X86 => SETS_FLAGS_X86.split_ascii_whitespace().any(|m| m == mnemonic),
        Isa::A64 => SETS_FLAGS_A64.split_ascii_whitespace().any(|m| m == mnemonic),
    };
    // The padding `=` gives the length of the bytes, which is public (see Base-N of the design),
    // so a compare with it gives public flags.
    let padding = sets_flags
        && match isa {
            Isa::A64 => operands.ends_with("#0x3d") || operands.contains("#0x3d,"),
            Isa::X86 => operands.ends_with(",0x3d"),
        };
    if padding {
        state.flags = false;
        findings.insert(format!("note: {name}: {mnemonic} {operands}, a test of the padding"));
    } else if sets_flags {
        state.flags = !zeroing && (loaded || all.iter().any(|word| state.holds(isa, word)));
    }
    let made_constant = match isa {
        // `lea reg, [rip+...]`, and the load of the address of a static from the GOT.
        Isa::X86 => {
            // GNU writes `QWORD PTR`, and LLVM `qword ptr`.
            let operands = operands.to_ascii_lowercase();
            (mnemonic == "lea" || (mnemonic == "mov" && operands.contains("qword ptr"))) && operands.contains("[rip")
        }
        // `add x8, x8, #:lo12:table`, and the load of the address of a static from the GOT.
        Isa::A64 => {
            let source = match mnemonic {
                "add" => operands.split(',').nth(1).map(family),
                "ldr" => memory.as_ref().and_then(|(base, _)| base.clone()),
                _ => None,
            };
            mnemonic == "adrp" || source.is_some_and(|source| state.constant.contains(&source))
        }
    };
    if mnemonic == "call" || mnemonic == "bl" {
        state.constant.clear();
        state.data.clear();
        state.clean.clear();
        state.reduced.clear();
        state.flags = false;
        state.environment = None;
    }
    // x86 writes `rax` and `rdx` from `rax` and the operand in `mul`, `div`, one-operand `imul`,
    // and `cqo`.
    if isa == Isa::X86
        && (["mul", "div", "idiv", "cqo", "cdq"].contains(&mnemonic) || (mnemonic == "imul" && all.len() == 1))
    {
        let wide = from_data || state.data.contains("rax");
        for register in ["rax", "rdx"] {
            if wide {
                state.data.insert(register.to_owned());
            } else {
                state.data.remove(register);
            }
        }
        return;
    }
    // A64 `ldp` writes two registers.
    let mut targets: Vec<&str> = destination.iter().map(|_| first).collect();
    if isa == Isa::A64 && mnemonic.starts_with("ldp") {
        targets.extend(all.get(1).copied());
    }
    if isa == Isa::X86 && mnemonic == "xchg" {
        targets.extend(all.get(1).copied());
    }
    for target in targets {
        if let Some(vector) = lanes(isa, target) {
            if from_data {
                state.clean.remove(&vector);
            } else {
                state.clean.insert(vector.clone());
            }
            if REDUCTIONS.contains(&mnemonic) {
                state.reduced.insert(vector);
            } else {
                state.reduced.remove(&vector);
            }
            continue;
        }
        let register = family(target);
        if made_constant {
            state.constant.insert(register.clone());
        } else {
            state.constant.remove(&register);
        }
        if state.environment.as_ref() == Some(&register) {
            state.environment = None;
        }
        if from_data {
            state.data.insert(register);
        } else {
            state.data.remove(&register);
        }
    }
}

/// The register that an instruction writes as its first operand, if it writes one.
fn written(isa: Isa, instruction: &Instruction) -> Option<String> {
    let first = instruction.operands.split(',').next()?.trim();
    if first.contains('[') {
        return None;
    }
    let reads_only = match isa {
        Isa::X86 => {
            [
                "cmp", "test", "bt", "push", "call", "jmp", "ptest", "vptest", "ucomiss", "ucomisd", "comiss", "comisd",
            ]
            .contains(&instruction.mnemonic.as_str())
                || instruction.mnemonic.starts_with('j')
                || instruction.mnemonic.starts_with("kortest")
                || instruction.mnemonic.starts_with("ktest")
        }
        Isa::A64 => {
            [
                "cmp", "cmn", "tst", "ccmp", "ccmn", "msr", "prfm", "fcmp", "fcmpe", "br", "blr", "ret",
            ]
            .contains(&instruction.mnemonic.as_str())
                || branches(isa, &instruction.mnemonic)
                || instruction.mnemonic.starts_with("st")
        }
    };
    (!reads_only).then(|| family(first.trim_start_matches('{')))
}

/// The mnemonic of an x86 instruction as the DOIT list names it, with its prefixes.
fn doit_name(instruction: &Instruction) -> String {
    const PREFIXES: [&str; 12] = [
        "rep", "repz", "repe", "repnz", "repne", "lock", "notrack", "bnd", "data16", "cs", "ds", "addr32",
    ];
    let mut words = core::iter::once(instruction.mnemonic.as_str()).chain(instruction.operands.split_whitespace());
    let mut prefixes = Vec::new();
    let mut mnemonic = "";
    for word in words.by_ref() {
        if PREFIXES.contains(&word) {
            prefixes.push(word);
        } else {
            mnemonic = word;
            break;
        }
    }
    let operands = words.collect::<Vec<_>>().join(" ").to_ascii_uppercase();
    let upper = mnemonic.to_ascii_uppercase();
    if upper.starts_with("NOP") || (upper == "XCHG" && operands == "AX,AX") {
        return "NOP".to_owned();
    }
    if prefixes.iter().any(|prefix| prefix.starts_with("rep"))
        && (upper.starts_with("STOS") || upper.starts_with("MOVS"))
    {
        let size = ["BYTE", "WORD", "DWORD", "QWORD"]
            .iter()
            .zip(["B", "W", "D", "Q"])
            .rev()
            .find(|(width, _)| operands.contains(&format!("{width} PTR")))
            .map_or_else(|| upper.get(4..).unwrap_or_default(), |(_, letter)| letter);
        return format!("REP_{}{size}", upper.get(..4).unwrap_or_default());
    }
    if upper == "MOVABS" {
        return "MOV".to_owned();
    }
    if upper == "SAL" {
        return "SHL".to_owned();
    }
    for stem in ["CMOV", "SET"] {
        if let Some(condition) = upper.strip_prefix(stem) {
            let canonical = match condition {
                "E" => "Z",
                "NE" => "NZ",
                "A" => "NBE",
                "AE" | "NC" => "NB",
                "C" | "NAE" => "B",
                "NA" => "BE",
                "G" => "NLE",
                "GE" => "NL",
                "NGE" => "L",
                "NG" => "LE",
                "PE" => "P",
                "PO" => "NP",
                other => other,
            };
            return format!("{stem}{canonical}");
        }
    }
    // The compares of AVX-512 with a predicate in the name, as `vpcmpnltub`.
    if let Some(rest) = upper.strip_prefix("VPCMP") {
        for predicate in ["EQ", "LT", "LE", "FALSE", "NEQ", "NLT", "NLE", "TRUE"] {
            if let Some(kind) = rest.strip_prefix(predicate)
                && (kind.starts_with('U') || predicate != "EQ")
                && ["B", "W", "D", "Q", "UB", "UW", "UD", "UQ"].contains(&kind)
            {
                return format!("VPCMP{kind}");
            }
        }
    }
    upper
}

/// Whether the instruction writes `PSTATE.DIT`.
fn writes_dit(instruction: &Instruction) -> bool {
    let operands = instruction.operands.to_ascii_lowercase();
    instruction.mnemonic == "msr" && (operands.starts_with("dit") || operands.starts_with("s3_3_c4_c2_5"))
}

/// Checks that each write that sets `PSTATE.DIT` is followed by its barrier on each path, and gives
/// the number of such writes. A write sets DIT when its register holds `1 << 24`, the bit of DIT.
fn barriers(functions: &BTreeMap<String, Vec<Instruction>>, findings: &mut BTreeSet<String>) -> usize {
    /// The instructions that a path from the write may run before its barrier: the detection of
    /// the CPU that chooses the barrier takes about 15 on Linux.
    const REACH: usize = 32;
    let mut sets = 0;
    for (name, function) in functions {
        for (at, instruction) in function.iter().enumerate() {
            if !writes_dit(instruction) {
                continue;
            }
            let Some(source) = instruction.operands.split(',').nth(1).map(family) else {
                continue;
            };
            let sets_dit = function
                .get(..at)
                .unwrap_or_default()
                .iter()
                .rev()
                .take(REACH)
                .any(|earlier| {
                    earlier.mnemonic == "mov"
                        && earlier.operands.split(',').next().map(family) == Some(source.clone())
                        && (earlier.operands.contains("#0x1000000") || earlier.operands.contains("#16777216"))
                });
            if !sets_dit {
                continue;
            }
            sets += 1;
            // A walk of each path from the write, with the branch targets in the function.
            let mut paths = vec![(at + 1, 0_usize, false)];
            while let Some((index, steps, dsb)) = paths.pop() {
                let Some(next) = function.get(index) else {
                    findings.insert(format!(
                        "{name}: a write that sets DIT at {:x} ends with no barrier",
                        instruction.address
                    ));
                    break;
                };
                let mnemonic = next.mnemonic.as_str();
                if mnemonic == "sb" || (dsb && mnemonic == "isb") {
                    continue;
                }
                // The detection of the CPU, which chooses the barrier, can come first: its loads, its
                // stores of the cache of the features, and its call, which returns to the path. The
                // asm of DIT clobbers memory, so no access of the call that holds DIT moves before
                // the barrier: the path ends at the next other call or return.
                let detection = mnemonic == "bl" && target(&next.operands).is_some_and(|callee| public(&callee));
                let touches = !detection && ["bl", "blr", "br", "ret"].contains(&mnemonic);
                if touches || steps >= REACH {
                    findings.insert(format!(
                        "{name}: a write that sets DIT at {:x} reaches {mnemonic} at {:x} with no barrier",
                        instruction.address, next.address
                    ));
                    break;
                }
                let dsb = dsb || (mnemonic == "dsb" && next.operands == "nsh");
                let branch = mnemonic == "b"
                    || mnemonic.starts_with("b.")
                    || mnemonic.starts_with("cb")
                    || mnemonic.starts_with("tb");
                if branch {
                    let destination = next.operands.split(',').next_back().map(str::trim).unwrap_or_default();
                    let destination = destination
                        .split_whitespace()
                        .next()
                        .unwrap_or_default()
                        .trim_start_matches("0x");
                    if let Ok(destination) = u64::from_str_radix(destination, 16)
                        && let Some(target) = function.iter().position(|code| code.address == destination)
                    {
                        paths.push((target, steps + 1, dsb));
                    }
                }
                if mnemonic != "b" {
                    paths.push((index + 1, steps + 1, dsb));
                }
            }
        }
    }
    sets
}
