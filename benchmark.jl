using BenchmarkTools

# Different implementations to test
function exp_softmax(x)
    ex = exp.(x .- maximum(x))
    ex ./ sum(ex)
end

function pow2_softmax(x)
    pow2x = (2.0).^(x .- maximum(x))
    pow2x ./ sum(pow2x)
end

# Taylor expansion for exp(x), truncated
function taylor_exp(x, terms=5)
    result = one(x)
    term = one(x)
    for i in 1:terms
        term *= x/i
        result += term
    end
    result
end

function taylor_softmax(x)
    shifted = x .- maximum(x)
    ex = taylor_exp.(shifted)
    ex ./ sum(ex)
end

# Test vectors of different sizes
small_vec = randn(4)
medium_vec = randn(100)
large_vec = randn(10000)

# Benchmark individual operations first
println("Benchmarking individual operations on small vector:")
println("exp:")
@btime exp.($small_vec)
println("\n2^x:")
@btime (2.0).^($small_vec)
println("\ntaylor_exp:")
@btime taylor_exp.($small_vec)

println("\nBenchmarking full softmax on small vector:")
println("exp softmax:")
@btime exp_softmax($small_vec)
println("\n2^x softmax:")
@btime pow2_softmax($small_vec)
println("\ntaylor softmax:")
@btime taylor_softmax($small_vec)

println("\nBenchmarking full softmax on medium vector:")
println("exp softmax:")
@btime exp_softmax($medium_vec)
println("\n2^x softmax:")
@btime pow2_softmax($medium_vec)
println("\ntaylor softmax:")
@btime taylor_softmax($medium_vec)

println("\nBenchmarking full softmax on large vector:")
println("exp softmax:")
@btime exp_softmax($large_vec)
println("\n2^x softmax:")
@btime pow2_softmax($large_vec)
println("\ntaylor softmax:")
@btime taylor_softmax($large_vec)

# Check accuracy
println("\nAccuracy comparison on small vector:")
println("exp softmax:", exp_softmax(small_vec))
println("2^x softmax:", pow2_softmax(small_vec))
println("taylor softmax:", taylor_softmax(small_vec))

# Test numerical stability with large values
extreme_vec = [1000.0, 2000.0, 4000.0, 8000.0]
println("\nNumerical stability with extreme values:")
println("exp softmax:", exp_softmax(extreme_vec))
println("2^x softmax:", pow2_softmax(extreme_vec))
println("taylor softmax:", taylor_softmax(extreme_vec))

# Optional: Look at assembly
# using InteractiveUtils
# @code_native exp(1.0)
# @code_native 2.0^1.0
