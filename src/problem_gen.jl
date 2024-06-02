function generate_problem_data_randomQP(n::Int, seed::Int=1)
    Random.seed!(seed)
    m = Int(0.5 * n)

    # Generate problem data
    P = sprandn(n, n, 0.01)

    println("check2")
    sleep(1)
    rowval = collect(1:n)
    colptr = collect(1:n+1)
    nzval = ones(n)
    P = P * P' + 1e-02 * SparseMatrixCSC(n, n, colptr, rowval, nzval)
    q = randn(n)
    A = sprandn(m, n, 0.01)

    v = randn(n)   # Fictitious solution
    delta = rand(m)  # To get inequality
    ru = A * v + delta
    rl = -Inf * ones(m)
    lb = -Inf * ones(n)
    ub = Inf * ones(n)
     
    return QuadraticProgrammingProblem(
        size(A, 2),
        size(A, 1),
        lb,
        ub,
        Vector{Bool}(isfinite.(lb)),
        Vector{Bool}(isfinite.(ub)),
        P,
        q,
        0.0,
        -A,
        -A',
        -ru,
        0,
    )
end

function generate_lasso_problem(n::Int, seed::Int=1)
    # Set random seed
    Random.seed!(seed)

    # Initialize parameters
    m = Int(n * 0.5)
    Ad = sprandn(m, n, 0.01)
    x_true = (rand(n) .> 0.5) .* randn(n) ./ sqrt(n)
    bd = Ad * x_true + randn(m)
    lambda_max = norm(Ad' * bd, Inf)
    lambda_param = (1/5) * lambda_max

    # Construct the QP problem
    rowval_m = collect(1:m)
    colptr_m = collect(1:m+1)
    nzval_m = ones(m)
    P = blockdiag(spzeros(n, n), SparseMatrixCSC(m, m, colptr_m, rowval_m, nzval_m .* 2), spzeros(n, n))
    q = vcat(zeros(m + n), lambda_param * ones(n))
    rowval_n = collect(1:n)
    colptr_n = collect(1:n+1)
    nzval_n = ones(n)
    In = SparseMatrixCSC(n, n, colptr_n, rowval_n, nzval_n)
    Onm = spzeros(n, m)
    A = vcat(hcat(Ad, -SparseMatrixCSC(m, m, colptr_m, rowval_m, nzval_m), spzeros(m, n)),
             hcat(In, Onm, -In),
             hcat(-In, Onm, -In))
    rl = vcat(bd, -Inf * ones(n), -Inf * ones(n))
    ru = vcat(bd, zeros(n), zeros(n))
    lb = -Inf * ones(2*n+m)
    ub = Inf * ones(2*n+m)

    return QuadraticProgrammingProblem(
        size(A, 2),
        size(A, 1),
        lb,
        ub,
        Vector{Bool}(isfinite.(lb)),
        Vector{Bool}(isfinite.(ub)),
        P,
        q,
        0.0,
        -A,
        -A',
        -ru,
        m,
    )
end

function generate_svm_example(n::Int, seed::Int=1)
    # 设置随机种子
    Random.seed!(seed)

    # 初始化属性
    n_features = n               # 特征数量
    m_data = Int(n_features*0.5)    # 数据点数量
    N_half = Int(m_data * 0.5)
    gamma_val = 1.0
    b_svm_val = vcat(ones(N_half), -ones(N_half))

    # 生成数据
    A_upp = sprandn(N_half, n_features, 0.01)
    A_low = sprandn(N_half, n_features, 0.01)
    A_svm_val = vcat(A_upp / sqrt(n_features) .+ (A_upp .!= 0) / n_features,
                     A_low / sqrt(n_features) .- (A_low .!= 0) / n_features)

    # 生成 QP 问题
    P = spdiagm(0 => vcat(ones(n_features), zeros(m_data)))
    q = vcat(zeros(n_features), (gamma_val) * ones(m_data))

    rowval1 = collect(1:length(b_svm_val))
    colptr1 = collect(1:length(b_svm_val)+1)
    rowval2 = collect(1:m_data)
    colptr2 = collect(1:m_data+1)
    nzval2 = ones(m_data)

    A = hcat(-SparseMatrixCSC(colptr1, rowval1, b_svm_val) * A_svm_val, SparseMatrixCSC(colptr2, rowval2, nzval2))
    ru = ones(m_data)

    lb = vcat(-Inf * ones(n_features), zeros(m_data))
    ub = vcat(Inf * ones(n_features), Inf * ones(m_data))

    println("norm_A")
    println(norm(A))
    return QuadraticProgrammingProblem(
        size(A, 2),
        size(A, 1),
        lb,
        ub,
        Vector{Bool}(isfinite.(lb)),
        Vector{Bool}(isfinite.(ub)),
        P,
        q,
        0.0,
        A,
        A',
        ru,
        0,
    )
end

function generate_portfolio_example(n::Int, seed::Int=1)
    Random.seed!(seed)
    
    n_assets = n 
    k = Int(n*1)
    F = sprandn(n_assets, k, 0.15)
    D = spdiagm(0 => rand(n_assets) .* sqrt(k))
    mu = randn(n_assets)
    gamma = 1.0

    # Generate QP problem
    rowval1 = collect(1:n_assets)
    colptr1 = collect(1:n_assets + 1)
    nzval1 = rand(n_assets) .* sqrt(k) .* 2

    rowval2 = collect(n_assets + 1:k + n_assets)
    colptr2 = collect(n_assets + 2:k + n_assets + 1)
    nzval2 = ones(k) .* 2

    rowval = vcat(rowval1, rowval2)
    colptr = vcat(colptr1, colptr2)
    nzval = vcat(nzval1, nzval2)

    rand(n_assets) .* sqrt(k)

    rowval_k = collect(1:k)
    colptr_k = collect(1:k + 1)
    nzval_k = ones(k)

    P = SparseMatrixCSC(n_assets + k, n_assets + k, colptr, rowval, nzval)
    q = vcat(-mu ./ gamma, zeros(k))
    A = vcat(
        hcat(sparse(ones(1, n_assets)), spzeros(1, k)),
        hcat(F', -SparseMatrixCSC(k, k, colptr_k, rowval_k, nzval_k)),
    )
    ru = vcat(1.0, zeros(k))

    lb = vcat(zeros(n_assets), -Inf * ones(k))
    ub = vcat(ones(n_assets), Inf * ones(k))

    return QuadraticProgrammingProblem(
        size(A, 2),
        size(A, 1),
        lb,
        ub,
        Vector{Bool}(isfinite.(lb)),
        Vector{Bool}(isfinite.(ub)),
        P,
        q,
        0.0,
        -A,
        -A',
        -ru,
        k+1,
    )
end