/// Define a model's calibrated parameters and the priors over them, together.
///
/// Generates two structs with identical fields — the drawn values, and the
/// priors that produce them — plus the [`Draw`] impl connecting them to the
/// engine's positional layout:
///
/// ```
/// use abcsmc::{IntPrior, Priors, RealPrior, define_priors};
///
/// define_priors! {
///     /// Calibrated parameters for the renewal fit.
///     pub struct RenewalDraw / RenewalPriors {
///         /// Basic reproduction number.
///         r0: Real,
///         /// Infections seeding the outbreak.
///         initial_infections: Int,
///     }
/// }
///
/// let priors: Priors = RenewalPriors {
///     r0: RealPrior::exponential(1.0).unwrap(),
///     initial_infections: IntPrior::discrete_uniform(1, 4).unwrap(),
/// }
/// .into();
/// assert_eq!(priors.len(), 2);
/// ```
///
/// `Real` fields become `f64` and take a [`RealPrior`]; `Int` fields become
/// `i64` and take an [`IntPrior`]. Since
/// the priors are a named struct literal, **omitting a prior is a compile error**
/// and giving a field the wrong kind of prior is a type error. Declaration order
/// fixes the engine's positional layout, and because both structs come from one
/// declaration they cannot drift apart.
///
/// Fixed parameters are not declared here — they are not part of the inference.
/// Keep them on the model and splice them in when building the simulator's own
/// parameters, e.g. `Parameters { r0: draw.r0, ..self.base.clone() }`.
///
/// [`Draw`]: crate::Draw
/// [`RealPrior`]: crate::RealPrior
/// [`IntPrior`]: crate::IntPrior
#[macro_export]
macro_rules! define_priors {
    (
        $(#[$meta:meta])*
        $vis:vis struct $draw:ident / $priors:ident {
            $(
                $(#[$field_meta:meta])*
                $field:ident : $kind:ident
            ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Debug, PartialEq)]
        $vis struct $draw {
            $(
                $(#[$field_meta])*
                pub $field: $crate::__draw_value_ty!($kind),
            )*
        }

        #[doc = ::core::concat!(
            "Priors over the fields of [`", ::core::stringify!($draw), "`]. ",
            "Every field must be given a prior, so omitting one is a compile error."
        )]
        $vis struct $priors {
            $(
                pub $field: $crate::__draw_prior_ty!($kind),
            )*
        }

        impl ::core::convert::From<$priors> for $crate::Priors {
            fn from(priors: $priors) -> $crate::Priors {
                $crate::Priors::new()
                    $( .push(priors.$field) )*
            }
        }

        impl $crate::Draw for $draw {
            const NAMES: &'static [&'static str] = &[
                $( ::core::stringify!($field) ),*
            ];

            fn from_values(values: &[$crate::Value]) -> Self {
                // Struct-literal fields evaluate in written order, which is the
                // declaration order the priors were pushed in — so consuming the
                // iterator field by field stays aligned without any indices.
                let mut values = values.iter();
                Self {
                    $(
                        $field: $crate::__draw_read!(
                            $kind,
                            ::core::stringify!($field),
                            values.next()
                        ),
                    )*
                }
            }
        }
    };
}

/// Field type for a declared parameter kind.
#[doc(hidden)]
#[macro_export]
macro_rules! __draw_value_ty {
    (Real) => {
        f64
    };
    (Int) => {
        i64
    };
    ($other:ident) => {
        ::core::compile_error!(::core::concat!(
            "unknown parameter kind `",
            ::core::stringify!($other),
            "`; expected `Real` or `Int`"
        ))
    };
}

/// Prior type accepted for a declared parameter kind. Distinct per kind so a
/// mismatch is caught at the call site.
#[doc(hidden)]
#[macro_export]
macro_rules! __draw_prior_ty {
    (Real) => {
        $crate::RealPrior
    };
    (Int) => {
        $crate::IntPrior
    };
    ($other:ident) => {
        ::core::compile_error!(::core::concat!(
            "unknown parameter kind `",
            ::core::stringify!($other),
            "`; expected `Real` or `Int`"
        ))
    };
}

/// Read one value out of the engine's positional vector.
#[doc(hidden)]
#[macro_export]
macro_rules! __draw_read {
    (Real, $name:expr, $value:expr) => {
        match $value {
            ::core::option::Option::Some($crate::Value::Real(x)) => *x,
            ::core::option::Option::Some($crate::Value::Int(_)) => ::core::panic!(
                "parameter `{}` is declared `Real` but was drawn as an integer",
                $name
            ),
            ::core::option::Option::None => ::core::panic!(
                "no value drawn for parameter `{}`; priors() and the declaration disagree",
                $name
            ),
        }
    };
    (Int, $name:expr, $value:expr) => {
        match $value {
            ::core::option::Option::Some($crate::Value::Int(k)) => *k,
            ::core::option::Option::Some($crate::Value::Real(_)) => ::core::panic!(
                "parameter `{}` is declared `Int` but was drawn as a real",
                $name
            ),
            ::core::option::Option::None => ::core::panic!(
                "no value drawn for parameter `{}`; priors() and the declaration disagree",
                $name
            ),
        }
    };
}
