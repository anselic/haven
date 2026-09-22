//! Compilation-target facts shared by the compiler pipeline.
//!
//! Haven still compiles for its host, but target-dependent decisions belong in
//! this module rather than in scattered `cfg!` checks. `from_triple` is the
//! entry point a future `--target` flag can use; `host` only translates the
//! compiler's Rust build target into that same representation.

use std::fmt::{self, Display, Formatter};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    X86,
    X86_64,
    Aarch64,
}

impl Architecture {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::X86 => "x86",
            Self::X86_64 => "x86_64",
            Self::Aarch64 => "aarch64",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatingSystem {
    Windows,
    Linux,
    MacOs,
}

impl OperatingSystem {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Windows => "windows",
            Self::Linux => "linux",
            Self::MacOs => "macos",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Environment {
    Msvc,
    Gnu,
    Musl,
    Unknown,
}

impl Environment {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Msvc => "msvc",
            Self::Gnu => "gnu",
            Self::Musl => "musl",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CDataModel {
    /// `int`, `long`, and pointers are 32 bits.
    Ilp32,
    /// `int` is 32 bits; `long` and pointers are 64 bits.
    Lp64,
    /// `int` and `long` are 32 bits; pointers are 64 bits.
    Llp64,
}

/// The concrete Haven integer used to represent one C integer type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntRepr {
    pub bits: u16,
    pub signed: bool,
}

impl IntRepr {
    pub const fn signed(bits: u16) -> Self {
        Self { bits, signed: true }
    }

    pub const fn unsigned(bits: u16) -> Self {
        Self {
            bits,
            signed: false,
        }
    }
}

/// Integer types whose representation is defined by the target C ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CInteger {
    Char,
    SignedChar,
    UnsignedChar,
    Short,
    UnsignedShort,
    Int,
    UnsignedInt,
    Long,
    UnsignedLong,
    LongLong,
    UnsignedLongLong,
    WChar,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CAbi {
    pub data_model: CDataModel,
    pub char: IntRepr,
    pub short: IntRepr,
    pub int: IntRepr,
    pub long: IntRepr,
    pub long_long: IntRepr,
    pub wchar: IntRepr,
}

impl CAbi {
    /// Resolve an exact C integer type. Unsigned variants deliberately reuse
    /// the corresponding signed type's width so the relationship cannot drift.
    pub const fn integer(self, ty: CInteger) -> IntRepr {
        match ty {
            CInteger::Char => self.char,
            CInteger::SignedChar => IntRepr::signed(8),
            CInteger::UnsignedChar => IntRepr::unsigned(8),
            CInteger::Short => self.short,
            CInteger::UnsignedShort => IntRepr::unsigned(self.short.bits),
            CInteger::Int => self.int,
            CInteger::UnsignedInt => IntRepr::unsigned(self.int.bits),
            CInteger::Long => self.long,
            CInteger::UnsignedLong => IntRepr::unsigned(self.long.bits),
            CInteger::LongLong => self.long_long,
            CInteger::UnsignedLongLong => IntRepr::unsigned(self.long_long.bits),
            CInteger::WChar => self.wchar,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetSpec {
    pub architecture: Architecture,
    pub operating_system: OperatingSystem,
    pub environment: Environment,
    pub pointer_bits: u16,
    pub c: CAbi,
}

impl TargetSpec {
    /// Evaluate one leaf of a `@cfg` target predicate. Both keys and values are
    /// checked so a typo cannot silently compile the wrong branch away.
    pub fn matches_cfg(&self, key: &str, value: &str) -> Result<bool, String> {
        const ARCHES: &[&str] = &["x86", "x86_64", "aarch64"];
        const OSES: &[&str] = &["windows", "linux", "macos"];
        const ENVS: &[&str] = &["msvc", "gnu", "musl", "unknown"];
        const WIDTHS: &[&str] = &["8", "16", "32", "64"];
        const BOOLS: &[&str] = &["true", "false"];
        const KEYS: &[&str] = &[
            "target_arch",
            "target_os",
            "target_env",
            "target_pointer_width",
            "target_c_char_width",
            "target_c_char_signed",
            "target_c_short_width",
            "target_c_int_width",
            "target_c_long_width",
            "target_c_long_long_width",
            "target_c_wchar_width",
            "target_c_wchar_signed",
        ];

        let (actual, allowed): (String, &[&str]) = match key {
            "target_arch" => (self.architecture.as_str().to_string(), ARCHES),
            "target_os" => (self.operating_system.as_str().to_string(), OSES),
            "target_env" => (self.environment.as_str().to_string(), ENVS),
            "target_pointer_width" => (self.pointer_bits.to_string(), WIDTHS),
            "target_c_char_width" => (self.c.char.bits.to_string(), WIDTHS),
            "target_c_char_signed" => (self.c.char.signed.to_string(), BOOLS),
            "target_c_short_width" => (self.c.short.bits.to_string(), WIDTHS),
            "target_c_int_width" => (self.c.int.bits.to_string(), WIDTHS),
            "target_c_long_width" => (self.c.long.bits.to_string(), WIDTHS),
            "target_c_long_long_width" => (self.c.long_long.bits.to_string(), WIDTHS),
            "target_c_wchar_width" => (self.c.wchar.bits.to_string(), WIDTHS),
            "target_c_wchar_signed" => (self.c.wchar.signed.to_string(), BOOLS),
            _ => return Err(format!(
                "unknown cfg predicate '{}'; expected one of {}",
                key,
                KEYS.join(", "),
            )),
        };

        if !allowed.contains(&value) {
            return Err(format!(
                "invalid value '{}' for cfg predicate '{}'; expected {}",
                value,
                key,
                allowed.join(", "),
            ));
        }
        Ok(actual == value)
    }

    /// Parse the target triples Haven currently knows enough about to describe.
    /// This parser intentionally rejects unknown architectures and operating
    /// systems instead of guessing a C ABI from pointer width alone.
    pub fn from_triple(triple: &str) -> Result<Self, TargetError> {
        let normalized = triple.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            return Err(TargetError::MalformedTriple(triple.to_string()));
        }
        let parts: Vec<&str> = normalized.split('-').collect();

        let architecture = match parts.first().copied() {
            Some("i386" | "i486" | "i586" | "i686" | "x86") => Architecture::X86,
            Some("x86_64" | "amd64") => Architecture::X86_64,
            Some("aarch64" | "arm64") => Architecture::Aarch64,
            Some(other) => return Err(TargetError::UnsupportedArchitecture(other.to_string())),
            None => return Err(TargetError::MalformedTriple(triple.to_string())),
        };

        let operating_system = if parts.contains(&"windows") {
            OperatingSystem::Windows
        } else if parts.contains(&"linux") {
            OperatingSystem::Linux
        } else if parts.contains(&"darwin") || parts.contains(&"macos") {
            OperatingSystem::MacOs
        } else {
            return Err(TargetError::UnsupportedOperatingSystem(triple.to_string()));
        };

        let environment = if parts.contains(&"msvc") {
            Environment::Msvc
        } else if parts.contains(&"musl") {
            Environment::Musl
        } else if parts.contains(&"gnu") || parts.iter().any(|p| p.starts_with("gnu")) {
            Environment::Gnu
        } else {
            Environment::Unknown
        };

        let environment_is_supported = matches!(
            (operating_system, environment),
            (OperatingSystem::Windows, Environment::Msvc | Environment::Gnu)
                | (OperatingSystem::Linux, Environment::Gnu | Environment::Musl)
                | (OperatingSystem::MacOs, Environment::Unknown)
        );
        if !environment_is_supported {
            return Err(TargetError::UnsupportedEnvironment {
                environment,
                operating_system,
            });
        }

        Self::from_parts(architecture, operating_system, environment)
    }

    /// Describe the target on which this `havenc` executable was built.
    pub fn host() -> Result<Self, TargetError> {
        let architecture = match std::env::consts::ARCH {
            "x86" => Architecture::X86,
            "x86_64" => Architecture::X86_64,
            "aarch64" => Architecture::Aarch64,
            other => return Err(TargetError::UnsupportedArchitecture(other.to_string())),
        };
        let operating_system = match std::env::consts::OS {
            "windows" => OperatingSystem::Windows,
            "linux" => OperatingSystem::Linux,
            "macos" => OperatingSystem::MacOs,
            other => return Err(TargetError::UnsupportedOperatingSystem(other.to_string())),
        };
        let environment = if cfg!(target_env = "msvc") {
            Environment::Msvc
        } else if cfg!(target_env = "musl") {
            Environment::Musl
        } else if cfg!(target_env = "gnu") {
            Environment::Gnu
        } else {
            Environment::Unknown
        };

        Self::from_parts(architecture, operating_system, environment)
    }

    fn from_parts(
        architecture: Architecture,
        operating_system: OperatingSystem,
        environment: Environment,
    ) -> Result<Self, TargetError> {
        use Architecture::{Aarch64, X86, X86_64};
        use CDataModel::{Ilp32, Llp64, Lp64};
        use OperatingSystem::{Linux, MacOs, Windows};

        let (pointer_bits, data_model, char, long, wchar) = match (architecture, operating_system) {
            // Windows uses LLP64 on both supported 64-bit architectures and a
            // 16-bit unsigned wchar_t. Plain char is signed by the ABI default.
            (X86, Windows) => (
                32,
                Ilp32,
                IntRepr::signed(8),
                IntRepr::signed(32),
                IntRepr::unsigned(16),
            ),
            (X86_64 | Aarch64, Windows) => (
                64,
                Llp64,
                IntRepr::signed(8),
                IntRepr::signed(32),
                IntRepr::unsigned(16),
            ),

            // Linux AArch64's ABI defaults plain char and wchar_t to unsigned;
            // x86 Linux uses signed char and signed wchar_t.
            (X86, Linux) => (
                32,
                Ilp32,
                IntRepr::signed(8),
                IntRepr::signed(32),
                IntRepr::signed(32),
            ),
            (X86_64, Linux) => (
                64,
                Lp64,
                IntRepr::signed(8),
                IntRepr::signed(64),
                IntRepr::signed(32),
            ),
            (Aarch64, Linux) => (
                64,
                Lp64,
                IntRepr::unsigned(8),
                IntRepr::signed(64),
                IntRepr::unsigned(32),
            ),

            // Supported Darwin targets use LP64, signed char, and signed
            // 32-bit wchar_t. Haven does not claim support for 32-bit Darwin.
            (X86_64 | Aarch64, MacOs) => (
                64,
                Lp64,
                IntRepr::signed(8),
                IntRepr::signed(64),
                IntRepr::signed(32),
            ),
            (X86, MacOs) => {
                return Err(TargetError::UnsupportedCombination {
                    architecture,
                    operating_system,
                });
            }
        };

        Ok(Self {
            architecture,
            operating_system,
            environment,
            pointer_bits,
            c: CAbi {
                data_model,
                char,
                short: IntRepr::signed(16),
                int: IntRepr::signed(32),
                long,
                long_long: IntRepr::signed(64),
                wchar,
            },
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetError {
    MalformedTriple(String),
    UnsupportedArchitecture(String),
    UnsupportedOperatingSystem(String),
    UnsupportedEnvironment {
        environment: Environment,
        operating_system: OperatingSystem,
    },
    UnsupportedCombination {
        architecture: Architecture,
        operating_system: OperatingSystem,
    },
}

impl Display for TargetError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MalformedTriple(triple) => write!(f, "malformed target triple '{triple}'"),
            Self::UnsupportedArchitecture(arch) => {
                write!(f, "unsupported target architecture '{arch}'")
            }
            Self::UnsupportedOperatingSystem(os) => {
                write!(f, "unsupported target operating system in '{os}'")
            }
            Self::UnsupportedEnvironment {
                environment,
                operating_system,
            } => write!(
                f,
                "unsupported target environment {environment:?} for {operating_system:?}",
            ),
            Self::UnsupportedCombination {
                architecture,
                operating_system,
            } => write!(
                f,
                "unsupported target combination {architecture:?}-{operating_system:?}",
            ),
        }
    }
}

impl std::error::Error for TargetError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_x64_is_llp64_with_unsigned_16_bit_wchar() {
        let target = TargetSpec::from_triple("x86_64-pc-windows-msvc").unwrap();
        assert_eq!(target.pointer_bits, 64);
        assert_eq!(target.c.data_model, CDataModel::Llp64);
        assert_eq!(target.c.integer(CInteger::Long), IntRepr::signed(32));
        assert_eq!(
            target.c.integer(CInteger::UnsignedLong),
            IntRepr::unsigned(32)
        );
        assert_eq!(target.c.integer(CInteger::WChar), IntRepr::unsigned(16));
    }

    #[test]
    fn linux_x64_is_lp64_with_signed_32_bit_wchar() {
        let target = TargetSpec::from_triple("x86_64-unknown-linux-gnu").unwrap();
        assert_eq!(target.pointer_bits, 64);
        assert_eq!(target.c.data_model, CDataModel::Lp64);
        assert_eq!(target.c.integer(CInteger::Long), IntRepr::signed(64));
        assert_eq!(target.c.integer(CInteger::WChar), IntRepr::signed(32));
        assert_eq!(target.c.integer(CInteger::Char), IntRepr::signed(8));
    }

    #[test]
    fn linux_aarch64_uses_its_unsigned_char_defaults() {
        let target = TargetSpec::from_triple("aarch64-unknown-linux-gnu").unwrap();
        assert_eq!(target.c.integer(CInteger::Char), IntRepr::unsigned(8));
        assert_eq!(target.c.integer(CInteger::WChar), IntRepr::unsigned(32));
    }

    #[test]
    fn darwin_aarch64_is_lp64() {
        let target = TargetSpec::from_triple("aarch64-apple-darwin").unwrap();
        assert_eq!(target.operating_system, OperatingSystem::MacOs);
        assert_eq!(target.c.data_model, CDataModel::Lp64);
        assert_eq!(target.c.integer(CInteger::Long), IntRepr::signed(64));
        assert_eq!(target.c.integer(CInteger::WChar), IntRepr::signed(32));
    }

    #[test]
    fn host_is_a_supported_target() {
        TargetSpec::host().unwrap();
    }

    #[test]
    fn unknown_targets_are_rejected_instead_of_guessed() {
        assert!(matches!(
            TargetSpec::from_triple("riscv64-unknown-linux-gnu"),
            Err(TargetError::UnsupportedArchitecture(_)),
        ));
    }

    #[test]
    fn cfg_predicates_validate_keys_and_values() {
        let target = TargetSpec::from_triple("x86_64-pc-windows-msvc").unwrap();
        assert_eq!(target.matches_cfg("target_c_long_width", "32"), Ok(true));
        assert_eq!(target.matches_cfg("target_c_long_width", "64"), Ok(false));
        assert!(target.matches_cfg("target_c_long_width", "banana").is_err());
        assert!(target.matches_cfg("target_c_lnog_width", "32").is_err());
    }
}
