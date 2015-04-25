using PyPlot


type Const
    C::Float64
    γ::Float64
    ϵ::Float64
    tol::Float64
    sampleSize::Int64
end


type Param
    α::Array{Float64}
    b::Float64
end


function rbfKernel(x, y, γ)
    s = 0
    for ele in x - y
        s += ele * ele
    end
    return exp(- γ * s)
end


function decision_f(params, y, G, i::Int64, CONST)
    f = 0
    for j in 1:CONST.sampleSize
        f += params.α[j] * y[j] * G[i, j]
    end
    return f - params.b
end


function decision_f(params, y, X, x::Array, CONST)
    f = 0
    for j in 1:CONST.sampleSize
        f += params.α[j] * y[j] * rbfKernel(X[j, :], x, CONST.γ)
    end
    return f - params.b
end


function takeStep(params, i1, i2, y2, α2, E2, y, G, errorCache, CONST)
    if i1 == i2 
        return false
    end
    α1 = params.α[i1]
    y1 = y[i1]
    E1 = errorCache[i1]
    s = y1 * y2
    if y1 != y2
        L = max(0, α2 - α1)
        H = min(CONST.C, CONST.C + α2 - α1)
    else
        L = max(0, α2 + α1 - CONST.C)
        H = min(CONST.C, α2 + α1)
    end
    if abs(L - H) < CONST.ϵ
        return false
    end
    k11 = G[i1, i1]
    k12 = G[i1, i2]
    k22 = G[i2, i2]
    η = k11 + k22 - 2 * k12
    if η > 0
        a2 = α2 + y2 * (E1 - E2) / η
        if a2 < L
            a2 = L
        elseif a2 > H
            a2 = H
        end
    else
        L1 = α1 + s * (α2 - L)
        H1 = α1 + s * (α2 - H)
        f1 = y1 * (E1 + params.b) - α1 * k11 - s * α2 * k12
        f2 = y2 * (E2 + params.b) - s * α1 * k12 - α2 * k22
        Lobj = L1*f1 + L*f2 + 0.5*L1*L1*k11 + 0.5*L*L*k22 + s*L*L1*k12
        Hobj = H1*f1 + H*f2 + 0.5*H1*H1*k11 + 0.5*H*H*k22 + s*H*H1*k12
        if Lobj > Hobj + CONST.ϵ
            a2 = L
        elseif Lobj < Hobj - CONST.ϵ
            a2 = H
        else
            a2 = α2
        end
    end
    if abs(a2 - α2) < CONST.ϵ * (a2 + α2 + CONST.ϵ)
        return false
    end
    a1 = α1 + s * (α2 - a2)
    b1 = E1 + y1 * (a1 - α1) * k11 + y2 * (a2 - α2) * k12 + params.b
    b2 = E2 + y1 * (a1 - α1) * k12 + y2 * (a2 - α2) * k22 + params.b
    if a1 > 0 && a1 < CONST.C
        params.b = b1
    else
        if a2 > 0 && a2 < CONST.C
            params.b = b2
        else
            params.b = (b1 + b2) / 2.0
        end
    end

    #Update α
    params.α[i1] = a1
    params.α[i2] = a2
    #Update errorCache
    errorCache[i1] = decision_f(params, y, G, i1, CONST) - y[i1]
    errorCache[i2] = decision_f(params, y, G, i2, CONST) - y[i2]
    return true 
end


function examineExample(params, y, G, errorCache, i2, CONST)
    y2 = y[i2]
    α2 = params.α[i2]
    E2 = errorCache[i2]
    r2 = E2 * y2
    if (r2 < -CONST.tol && α2 < CONST.C) || (r2 > CONST.tol && α2 > 0)
        idx_zero_or_C = Int64[]
        n = rand(1:CONST.sampleSize)
        for j in 1:CONST.sampleSize
            if j + n <= CONST.sampleSize
                j = j + n
            else
                j = j + n - CONST.sampleSize
            end
            if params.α[j] != 0 && params.α[j] != CONST.C
                i1 = j
                if takeStep(params, i1, i2, y2, α2, E2, y, G, errorCache, CONST)
                    return 1 
                end
            else
                push!(idx_zero_or_C, j)
            end
        end
        n = rand(1:size(idx_zero_or_C)[1])
        for j in 1:size(idx_zero_or_C)[1]
            if j + n <= size(idx_zero_or_C)[1]
                j = j + n
            else
                j = j + n - size(idx_zero_or_C)[1]
            end
            i1 = idx_zero_or_C[j]
            if takeStep(params, i1, i2, y2, α2, E2, y, G, errorCache, CONST)
                return 1
            end
        end
    end
    return 0 
end


function evaluate(predicted_P::Array{Float64, 2}, predicted_N::Array{Float64, 2})
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in 1:size(predicted_P)[1]
        if predicted_P[i, 1]^2 + predicted_P[i, 2]^2 <= 0.25
            TP += 1
        else
            FP += 1
        end
    end
    for i in 1:size(predicted_N)[1]
        if predicted_N[i, 1]^2 + predicted_N[i, 2]^2 > 0.25
            TN += 1
        else
            FN += 1
        end
    end
    return float(TP + TN) / (TP + FP + TN + FN)
end


function main()

    # Training sample preparation.
    n = 500 
    srand(1234567)
    sample = 2*rand(n, 2) - 1
    X_P = Array{Float64, 2}[]
    X_N = Array{Float64, 2}[]
    for i in 1:n
        if sample[i, 1]^2 + sample[i, 2]^2 <= 0.25 
            push!(X_P, sample[i, :])
        else
            push!(X_N, sample[i, :])
        end
    end
    X_P = vcat(X_P...)
    X_N = vcat(X_N...)
    y_p = ones(size(X_P)[1])
    y_n = ones(size(X_N)[1]) * -1
    #X_P = [1 1; -1 -1; 0.5 0.5; -0.5 -0.5; 0.3 0.3]
    #X_N = [-1 1; 1 -1; 0.5 -0.5; -0.5 0.5; -0.1 -0.1]
    #y_p = [1, 1, 1, 1, 1]
    #y_n = [-1, -1, -1, -1, -1]
    X = vcat(X_P, X_N)
    y = vcat(y_p, y_n)

    # Constant variables and Parameters.
    C = 2.0^-3
    γ = 2.0^2
    tol = 10.0 ^ -2
    b = 0.0
    ϵ = 10.0 ^ -6
    sampleSize = size(y)[1]
    CONST = Const(C, γ, ϵ, tol, sampleSize)
    α = zeros(CONST.sampleSize)
    params = Param(α, b)

    # Initialize Gram Matrix
    G = zeros(sampleSize, sampleSize)
    for j in 1:CONST.sampleSize
        for i in 1:CONST.sampleSize
            G[i, j] = rbfKernel(X[i, :], X[j, :], CONST.γ)
        end
    end

    # Initialize errorCache
    errorCache = zeros(CONST.sampleSize)
    for i in 1:CONST.sampleSize
        errorCache[i] = decision_f(params, y, G, i, CONST) - y[i]
    end

    # Optimization rootine
    numChanged = 0
    examineAll = true
    cnt = 0
    while numChanged > 0 || examineAll
        numChanged = 0
        if examineAll
            for i in 1:CONST.sampleSize
                numChanged += examineExample(params, y, G, errorCache, i, CONST)
            end
        else
            for i in 1:CONST.sampleSize
                if α[i] > 0 && α[i] < C
                    numChanged += examineExample(params, y, G, errorCache, i, CONST)
                end
            end
        end
        if examineAll
            examineAll = false
        elseif numChanged == 0
            examineAll = true
        end
        cnt += 1
    end
    #@show cnt

    # Test.
    #X_test = [0.6 0.6; -0.6 -0.6; 0.6 -0.6; -0.6 0.6]
    n = 5000
    srand(5678)
    X_test = 2*rand(n, 2) - 1.0

    # Prediction.
    predicted_P = Array{Float64, 2}[]
    predicted_N = Array{Float64, 2}[]
    for i in 1:n
        decision_v = decision_f(params, y, X, X_test[i, :], CONST)
        if decision_v >= 0
            push!(predicted_P, X_test[i, :])
        else
            push!(predicted_N, X_test[i, :])
        end
    end
    if size(predicted_P)[1] == 0
        println("predicted_P is empty")
    elseif size(predicted_N)[1] == 0
        println("predicted_N is empty")
    end
    predicted_P = vcat(predicted_P...)
    predicted_N = vcat(predicted_N...)

    # Evaluation.
    accuracy = evaluate(predicted_P, predicted_N)
    @show accuracy

    # Plotting.
    figure("sample", figsize=(8, 16))
    subplot(211)
    scatter(X_P[:, 1], X_P[:, 2], marker="o", color="r")
    scatter(X_N[:, 1], X_N[:, 2], marker="x", color="b")
    title("Training (500 samples)")
    subplot(212)
    scatter(predicted_P[:, 1], predicted_P[:, 2], marker="o", color="r")
    scatter(predicted_N[:, 1], predicted_N[:, 2], marker="x", color="b")
    title("Test (5000 samples)")
end


main()
