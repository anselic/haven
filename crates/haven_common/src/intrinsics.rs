#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Intrinsic {
    /// `null::<T>() -> *T` (a null pointer of type `*T`)
    Null,

    /// Numerical type cast
    /// `numerical_cast(0, iN, uN, fN) -> iN/uN/fN`
    NumericalCast,
    /// Size in bytes of a type, respecting the target ABI layout.
    /// `sizeof(T) -> u64`
    Sizeof,
    /// Reinterpret a pointer as a pointer of another type.
    /// `ptr_cast::<*T>(p: *U) -> *T`. A no-op at the machine level (LLVM
    /// pointers are opaque); it only changes the static pointee type. Used to
    /// turn the untyped `*void` from the allocator into a typed `*T`.
    PtrCast,

    /// `__simd_splat::<T, N>(value) -> T where T = simd<T, N>`
    /// e.g. `value = __simd_splat::<f32, 4>(1.0) -> simd<f32, 4> (1.0, 1.0, 1.0, 1.0)`
    SimdSplat,
    /// `__simd_load::<T, N>(slice, offset: iN/uN) -> T where T = simd<T, N>`
    /// e.g. `value = __simd_load::<f32, 4>(buf, i) -> simd<f32, 4>`
    SimdLoad,
    /// `__simd_store::<T, N>(slice, offset: iN/uN, value: T) -> () where T = simd<T, N>`
    /// e.g. `__simd_store::<f32, 4>(buf, i, value * 0.5) -> ()`
    SimdStore,
    /// `__simd_concat::<T, N>(value1, value2) -> T where T = simd<T, 2N>`
    SimdConcat,
    /// `__simd_low::<T, N>(value: simd<T, M>) -> simd<T, N> where N < M`
    /// e.g. `value = __simd_low::<f32, 2>(simd::<f32, 4>) -> simd<f32, 2> (value[0], value[1])`
    SimdLow,
    /// `__simd_high::<T, N>(value: simd<T, M>) -> simd<T, N> where N < M`
    SimdHigh,
}

impl Intrinsic {
    pub fn lookup(name: &str) -> Option<Self> {
        match name {
            "null" => Some(Self::Null),
            "numerical_cast" => Some(Self::NumericalCast),
            "sizeof" => Some(Self::Sizeof),
            "ptr_cast" => Some(Self::PtrCast),
            "__simd_splat" => Some(Self::SimdSplat),
            "__simd_load" => Some(Self::SimdLoad),
            "__simd_store" => Some(Self::SimdStore),
            "__simd_concat" => Some(Self::SimdConcat),
            "__simd_low" => Some(Self::SimdLow),
            "__simd_high" => Some(Self::SimdHigh),
            _ => None,
        }
    }
}

impl std::fmt::Display for Intrinsic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Null => "null",
            Self::NumericalCast => "numerical_cast",
            Self::Sizeof => "sizeof",
            Self::PtrCast => "ptr_cast",
            Self::SimdSplat => "__simd_splat",
            Self::SimdLoad => "__simd_load",
            Self::SimdStore => "__simd_store",
            Self::SimdConcat => "__simd_concat",
            Self::SimdLow => "__simd_low",
            Self::SimdHigh => "__simd_high",
        };
        write!(f, "{}", name)
    }
}

/// Constraint on an intrinsic's type parameter (the kind of type it accepts).
#[derive(Clone, Copy, Debug)]
pub enum TyConstraint {
    /// Any sized type: a scalar (including `bool`) or a struct. Used by `sizeof`.
    Any,
    /// A numeric scalar (`iN`/`uN`/`fN`) - also exactly the set of valid SIMD
    /// element types.
    Numeric,
    /// Any pointer type (`*T`). Used by `ptr_cast`.
    Pointer,
}

/// Bound on an intrinsic's const (compile-time integer) parameter: an inclusive
/// numeric range plus a divisibility requirement. Const constraints don't form a
/// clean taxonomy (unlike type kinds) - they're parameterized numeric predicates
/// - so this is a value, not an enum. `multiple_of == 1` means no divisibility
/// constraint.
#[derive(Clone, Copy, Debug)]
pub struct ConstBound {
    pub min: i64,
    pub max: i64,
    pub multiple_of: u32,
}

/// A SIMD lane count: `1..=64`.
pub const LANES: ConstBound = ConstBound { min: 1, max: 64, multiple_of: 1 };
/// An even SIMD lane count: an even value in `1..=64`.
pub const EVEN_LANES: ConstBound = ConstBound { min: 1, max: 64, multiple_of: 2 };

/// The generic header of an intrinsic: the leading type parameters and const
/// parameters it takes, followed by `value_arity` ordinary value arguments.
///
/// This drives arity, kind, and bound checking generically. Relationships
/// *between* value arguments (e.g. `simd_concat`'s inputs being half-width, or
/// `simd_low`'s input being wider than its result) remain bespoke in
/// `typecheck_intrinsic`, because they don't reduce to a simple substitution.
pub struct IntrinsicSig {
    pub type_params: &'static [TyConstraint],
    pub const_params: &'static [ConstBound],
    pub value_arity: usize,
}

impl Intrinsic {
    /// The intrinsic's generic signature header: type parameters and const
    /// parameters (supplied via turbofish) followed by `value_arity` ordinary
    /// value arguments.
    pub fn signature(self) -> IntrinsicSig {
        use TyConstraint::*;
        let (type_params, const_params, value_arity): (
            &'static [TyConstraint],
            &'static [ConstBound],
            usize,
        ) = match self {
            Self::Null          => (&[Pointer], &[],           0),
            Self::NumericalCast => (&[Numeric], &[],           1),
            Self::Sizeof        => (&[Any],     &[],           0),
            Self::PtrCast       => (&[Pointer], &[],           1),
            Self::SimdSplat     => (&[Numeric], &[LANES],      1),
            Self::SimdLoad      => (&[Numeric], &[LANES],      2),
            Self::SimdStore     => (&[Numeric], &[LANES],      3),
            Self::SimdConcat    => (&[Numeric], &[EVEN_LANES], 2),
            Self::SimdLow       => (&[Numeric], &[EVEN_LANES], 1),
            Self::SimdHigh      => (&[Numeric], &[EVEN_LANES], 1),
        };
        IntrinsicSig { type_params, const_params, value_arity }
    }
}