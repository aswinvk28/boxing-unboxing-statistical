## calculating the weighted sum of denominator or numerator
context Boxing::weightedSum(A: NdArray, B: NdArray)
pre
   Tensors::initialize

return A.transpose().matmul(B).sum()

post
    Tensors::evaluate

## Uniform Distributions
context Boxing::distributions()
pre
    DMU::initialize

let d1 = Random::uniform(1,100,(288,200))
let d2 = Random::uniform(1,100,(288,200))

let inputs = Random::normal(1,20,100)

return (d1, d2, inputs)

post
    -- post conditions

## DMU specification
context DMU::initialize(distribution: String, dmu_input, dmu_output)
pre
    -- preconditions

let dmu_input = Random::normal(1,20,100)
if distribution == "F"
    let dmu_output = DMU::Spearmann
else if distribution == "Chi"
    let dmu_output = DMU::MMSE
else if distribution == "Mahalanobis"
    let dmu_output = DMU::Mahalanobis

return True

post
    -- postconditions

## Evaluate Unboxing
Unboxing::evaluate(distribution: String, dmu_input, dmu_output)
pre
    DMU::initialize

numerator = Boxing::weightedSum
denominator = Boxing::weightedSum
constraints = DMU::constraints

post
    DEA::Minimize(numerator, denominator, constraints)

