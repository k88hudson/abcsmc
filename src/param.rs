/// A single parameter value, tagged by kind so the engine can perturb it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Real(f64),
    Int(i64),
}

impl Value {
    pub fn as_f64(self) -> f64 {
        match self {
            Value::Real(x) => x,
            Value::Int(k) => k as f64,
        }
    }
}

/// A particle's parameter vector, in the order the priors were declared.
///
/// Read individual parameters in `Model::simulate` with [`Params::real`] /
/// [`Params::int`] by their declaration index.
#[derive(Clone, Debug)]
pub struct Params(pub(crate) Vec<Value>);

impl Params {
    /// The real-valued parameter at index `i`. Panics if it is not real.
    pub fn real(&self, i: usize) -> f64 {
        match self.0[i] {
            Value::Real(x) => x,
            Value::Int(_) => panic!("parameter {i} is an integer, not real"),
        }
    }

    /// The integer-valued parameter at index `i`. Panics if it is not an integer.
    pub fn int(&self, i: usize) -> i64 {
        match self.0[i] {
            Value::Int(k) => k,
            Value::Real(_) => panic!("parameter {i} is real, not an integer"),
        }
    }

    pub fn new(values: Vec<Value>) -> Self {
        Params(values)
    }

    pub fn set(&mut self, i: usize, value: Value) {
        self.0[i] = value;
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The raw positional values, in prior-declaration order. This is what
    /// [`Draw::from_values`] consumes.
    ///
    /// [`Draw::from_values`]: crate::Draw::from_values
    pub fn values(&self) -> &[Value] {
        &self.0
    }
}
