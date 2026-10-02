/// Define a model's calibrated parameters and the priors over them, together.
///
/// Generates two structs with identical fields — the drawn values, and the
/// priors that produce them — plus the [`Draw`] impl connecting them to the
/// engine's positional layout:
///
/// ```
/// use abcsmc::{Priors, define_priors, discrete_uniform_prior, exponential_prior};
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
///     r0: exponential_prior!(rate = 1.0).unwrap(),
///     initial_infections: discrete_uniform_prior!(a = 1, b = 4).unwrap(),
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

/// Prior constructors with named arguments, one macro per family:
/// `exponential_prior!(rate = 1.0)`. Each expands to the [`RealPrior`] or
/// [`IntPrior`] constructor for the parameters named, so it evaluates to the
/// same `Result`, and a family with two parameterizations picks one by its
/// argument names. Arguments go in the order shown; any other set of names is
/// a compile error listing the accepted forms.
///
/// ```
/// use abcsmc::{exponential_prior, gamma_prior, negative_binomial_prior};
///
/// let by_rate = exponential_prior!(rate = 4.0).unwrap();
/// let by_scale = exponential_prior!(scale = 0.25).unwrap();
/// assert_eq!(by_scale.to_string(), "Exponential(scale = 0.25)");
/// # let _ = by_rate;
///
/// let gamma = gamma_prior!(shape = 2.0, rate = 0.5).unwrap();
/// assert_eq!(gamma.to_string(), "Gamma(shape = 2, rate = 0.5)");
///
/// let nb = negative_binomial_prior!(mean = 6.0, k = 3.0).unwrap();
/// assert_eq!(nb.to_string(), "NegativeBinomial(mean = 6, k = 3)");
/// ```
///
/// | macro | arguments |
/// |---|---|
/// | `uniform_prior!` | `a = .., b = ..` |
/// | `normal_prior!` | `mean = .., std_dev = ..` |
/// | `exponential_prior!` | `rate = ..` or `scale = ..` |
/// | `log_normal_prior!` | `mu = .., sigma = ..` |
/// | `gamma_prior!` | `shape = .., scale = ..`, `shape = .., rate = ..`, or `mean = .., shape = ..` |
/// | `weibull_prior!` | `shape = .., scale = ..` |
/// | `beta_prior!` | `alpha = .., beta = ..` or `alpha = .., beta = .., min = .., max = ..` |
/// | `discrete_uniform_prior!` | `a = .., b = ..` |
/// | `poisson_prior!` | `lambda = ..` |
/// | `binomial_prior!` | `n = .., p = ..` |
/// | `negative_binomial_prior!` | `r = .., p = ..` or `mean = .., k = ..` |
///
/// [`RealPrior`]: crate::RealPrior
/// [`IntPrior`]: crate::IntPrior
#[macro_export]
macro_rules! uniform_prior {
    (a = $a:expr, b = $b:expr $(,)?) => {
        $crate::RealPrior::uniform($a, $b)
    };
    ($($other:tt)*) => {
        ::core::compile_error!("uniform_prior! takes `a = .., b = ..`")
    };
}

/// See [`uniform_prior!`](crate::uniform_prior) for the arguments.
#[macro_export]
macro_rules! normal_prior {
    (mean = $mean:expr, std_dev = $std_dev:expr $(,)?) => {
        $crate::RealPrior::normal($mean, $std_dev)
    };
    ($($other:tt)*) => {
        ::core::compile_error!("normal_prior! takes `mean = .., std_dev = ..`")
    };
}

/// See [`uniform_prior!`](crate::uniform_prior) for the arguments.
#[macro_export]
macro_rules! exponential_prior {
    (rate = $rate:expr $(,)?) => {
        $crate::RealPrior::exponential_rate($rate)
    };
    (scale = $scale:expr $(,)?) => {
        $crate::RealPrior::exponential_scale($scale)
    };
    ($($other:tt)*) => {
        ::core::compile_error!("exponential_prior! takes `rate = ..` or `scale = ..`")
    };
}

/// See [`uniform_prior!`](crate::uniform_prior) for the arguments.
#[macro_export]
macro_rules! log_normal_prior {
    (mu = $mu:expr, sigma = $sigma:expr $(,)?) => {
        $crate::RealPrior::log_normal($mu, $sigma)
    };
    ($($other:tt)*) => {
        ::core::compile_error!("log_normal_prior! takes `mu = .., sigma = ..`")
    };
}

/// See [`uniform_prior!`](crate::uniform_prior) for the arguments.
#[macro_export]
macro_rules! gamma_prior {
    (shape = $shape:expr, scale = $scale:expr $(,)?) => {
        $crate::RealPrior::gamma_shape_scale($shape, $scale)
    };
    (shape = $shape:expr, rate = $rate:expr $(,)?) => {
        $crate::RealPrior::gamma_shape_rate($shape, $rate)
    };
    (mean = $mean:expr, shape = $shape:expr $(,)?) => {
        $crate::RealPrior::gamma_mean_shape($mean, $shape)
    };
    ($($other:tt)*) => {
        ::core::compile_error!(
            "gamma_prior! takes `shape = .., scale = ..`, `shape = .., rate = ..`, or `mean = .., shape = ..`"
        )
    };
}

/// See [`uniform_prior!`](crate::uniform_prior) for the arguments.
#[macro_export]
macro_rules! weibull_prior {
    (shape = $shape:expr, scale = $scale:expr $(,)?) => {
        $crate::RealPrior::weibull($shape, $scale)
    };
    ($($other:tt)*) => {
        ::core::compile_error!("weibull_prior! takes `shape = .., scale = ..`")
    };
}

/// See [`uniform_prior!`](crate::uniform_prior) for the arguments.
#[macro_export]
macro_rules! beta_prior {
    (alpha = $alpha:expr, beta = $beta:expr $(,)?) => {
        $crate::RealPrior::beta($alpha, $beta)
    };
    (alpha = $alpha:expr, beta = $beta:expr, min = $min:expr, max = $max:expr $(,)?) => {
        $crate::RealPrior::scaled_beta($alpha, $beta, $min, $max)
    };
    ($($other:tt)*) => {
        ::core::compile_error!("beta_prior! takes `alpha = .., beta = ..` or `alpha = .., beta = .., min = .., max = ..`")
    };
}

/// See [`uniform_prior!`](crate::uniform_prior) for the arguments.
#[macro_export]
macro_rules! discrete_uniform_prior {
    (a = $a:expr, b = $b:expr $(,)?) => {
        $crate::IntPrior::discrete_uniform($a, $b)
    };
    ($($other:tt)*) => {
        ::core::compile_error!("discrete_uniform_prior! takes `a = .., b = ..`")
    };
}

/// See [`uniform_prior!`](crate::uniform_prior) for the arguments.
#[macro_export]
macro_rules! poisson_prior {
    (lambda = $lambda:expr $(,)?) => {
        $crate::IntPrior::poisson($lambda)
    };
    ($($other:tt)*) => {
        ::core::compile_error!("poisson_prior! takes `lambda = ..`")
    };
}

/// See [`uniform_prior!`](crate::uniform_prior) for the arguments.
#[macro_export]
macro_rules! binomial_prior {
    (n = $n:expr, p = $p:expr $(,)?) => {
        $crate::IntPrior::binomial($n, $p)
    };
    ($($other:tt)*) => {
        ::core::compile_error!("binomial_prior! takes `n = .., p = ..`")
    };
}

/// See [`uniform_prior!`](crate::uniform_prior) for the arguments.
#[macro_export]
macro_rules! negative_binomial_prior {
    (r = $r:expr, p = $p:expr $(,)?) => {
        $crate::IntPrior::negative_binomial_r_p($r, $p)
    };
    (mean = $mean:expr, k = $k:expr $(,)?) => {
        $crate::IntPrior::negative_binomial_mean_dispersion($mean, $k)
    };
    ($($other:tt)*) => {
        ::core::compile_error!(
            "negative_binomial_prior! takes `r = .., p = ..` or `mean = .., k = ..`"
        )
    };
}
