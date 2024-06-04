using Random
using SparseArrays
using LinearAlgebra
using JuMP
using Ipopt

function generate_problem_data_randomQP(n, seed=1)
    Random.seed!(seed)

    m = Int(0.5 * n)

    # Generate problem data
    P = sprandn(n, n, 0.15)
    P = P * P' + 1e-02 * I(n)
    q = randn(n)
    A = sprandn(m, n, 0.15)
    v = randn(n)   # Fictitious solution
    delta = rand(m)  # To get inequality
    ru = A * v + delta
    rl = -Inf * ones(m)
    lb = -Inf * ones(n)
    ub = Inf * ones(n)

    problem = Dict{String, Any}()
    problem["num_variables"] = size(A, 2)
    problem["num_constraints"] = size(A, 1)
    problem["variable_lower_bound"] = lb
    problem["variable_upper_bound"] = ub
    problem["isfinite_variable_lower_bound"] = Vector{Bool}(isfinite.(lb))
    problem["isfinite_variable_upper_bound"] = Vector{Bool}(isfinite.(ub))
    problem["objective_matrix"] = P
    problem["objective_constant"] = 0.0
    problem["objective_vector"] = q
    problem["constraint_matrix"] = A
    problem["constraint_matrix_t"] = A'
    problem["right_hand_side"] = ru
    problem["num_equalities"] = 0
    
    return problem
end

function generate_lasso_problem(n::Int, seed::Int=1)
    # Set random seed
    Random.seed!(seed)

    # Initialize parameters
    m = n * 10
    Ad = sprandn(m, n, 0.15)
    x_true = (rand(n) .> 0.5) .* randn(n) ./ sqrt(n)
    bd = Ad * x_true + randn(m)
    lambda_max = norm(Ad' * bd, Inf)
    lambda_param = (1/5) * lambda_max

    # Construct the QP problem
    P = blockdiag(spzeros(n, n), sparse(2 * I(m)), spzeros(n, n))
    q = vcat(zeros(m + n), lambda_param * ones(n))
    In = I(n)
    Onm = spzeros(n, m)
    A = vcat(hcat(Ad, -I(m), spzeros(m, n)),
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
        A,
        A',
        ru,
        m,
    )
    return problem
end

function generate_lasso_problem(n::Int, seed::Int=1)
    # Set random seed
    Random.seed!(seed)

    # Initialize parameters
    m = n * 2
    Ad = sprandn(m, n, 0.15)
    x_true = (rand(n) .> 0.5) .* randn(n) ./ sqrt(n)
    bd = Ad * x_true + randn(m)
    lambda_max = norm(Ad' * bd, Inf)
    lambda_param = (1/5) * lambda_max

    # Construct the QP problem
    P = blockdiag(spzeros(n, n), sparse(2 * I(m)), spzeros(n, n))
    q = vcat(zeros(m + n), lambda_param * ones(n))
    In = I(n)
    Onm = spzeros(n, m)
    A = vcat(hcat(Ad, -I(m), spzeros(m, n)),
             hcat(In, Onm, -In),
             hcat(-In, Onm, -In))
    l = vcat(bd, -Inf * ones(n), -Inf * ones(n))
    u = vcat(bd, zeros(n), zeros(n))

    problem = Dict{Symbol, Any}()
    problem[:P] = P
    problem[:q] = q
    problem[:A] = A
    problem[:l] = l
    problem[:u] = u
    problem[:m] = size(A, 1)
    problem[:n] = size(A, 2)

    return problem
end


function solve_lasso_problem(problem)
    # Extract problem data
    P = problem[:P]
    q = problem[:q]
    A = problem[:A]
    l = problem[:l]
    u = problem[:u]

    # Create JuMP model
    model = Model(Ipopt.Optimizer)

    # Define variables
    @variable(model, x[1:problem[:n]])

    # Define objective
    objective = 0.5 * dot(x, P * x) + dot(q, x)
    @objective(model, Min, objective)

    # Define constraints
    for i in 1:problem[:m]
        row = A[i, :]
        @constraint(model, l[i] <= dot(row, x) <= u[i])
    end

    # Solve the problem
    optimize!(model)

    # Get the solution
    solution = value.(x)
    return solution
end

# Example usage
# n = 100
# problem = generate_problem_data_randomQP(n)
# problem = generate_lasso_problem(n)
# solution = solve_lasso_problem(problem)

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
    A_upp = sprandn(N_half, n_features, 0.15)
    A_low = sprandn(N_half, n_features, 0.15)
    A_svm_val = vcat(A_upp / sqrt(n_features) .+ (A_upp .!= 0) / n_features,
                     A_low / sqrt(n_features) .- (A_low .!= 0) / n_features)

    # 生成 QP 问题
    P = spdiagm(0 => vcat(2*ones(n_features), zeros(m_data)))
    q = vcat(zeros(n_features), (gamma_val) * ones(m_data))
    A = hcat(-diagm(b_svm_val) * A_svm_val, I(m_data))
    ru = ones(m_data)

    lb = vcat(-Inf * ones(n_features), zeros(m_data))
    ub = vcat(Inf * ones(n_features), Inf * ones(m_data))
    println(norm(A))
    # 生成 JuMP 模型
    model = Model(Ipopt.Optimizer)
    @variable(model, x[1:n_features])
    @variable(model, t[1:m_data] >= 0)
    @objective(model, Min, 0.5*sum(x[i]^2 for i in 1:n_features) + gamma_val * sum(t))
    @constraint(model, [i=1:m_data], t[i] >= dot(A_svm_val[i, :], x) * b_svm_val[i] + 1)

    optimize!(model)

    return
end

# 生成示例
#example = generate_svm_example(1000)

function generate_svm_example(n::Int, seed::Int=1)
    # 设置随机种子
    Random.seed!(seed)

    # 初始化属性
    n_features = n               # 特征数量
    m_data = n_features * 2    # 数据点数量
    N_half = div(m_data, 2)
    gamma_val = 1.0
    b_svm_val = vcat(ones(N_half), -ones(N_half))

    # 生成数据
    A_upp = sprandn(N_half, n_features, 0.15)
    A_low = sprandn(N_half, n_features, 0.15)
    A_svm_val = vcat(A_upp / sqrt(n_features) .+ (A_upp .!= 0) / n_features,
                     A_low / sqrt(n_features) .- (A_low .!= 0) / n_features)

    # 生成 QP 问题
    P = spdiagm(0 => vcat(ones(n_features), zeros(m_data)))
    q = vcat(zeros(n_features), (gamma_val / 2) * ones(m_data))
    A = hcat(diagm(b_svm_val) * A_svm_val, -I(m_data))
    ru = -ones(m_data)

    lb = vcat(-Inf * ones(n_features), zeros(m_data))
    ub = vcat(Inf * ones(n_features), Inf * ones(m_data))

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
    k = Int(n*10)
    F = sprandn(n_assets, k, 0.5)
    D = spdiagm(0 => rand(n_assets) .* sqrt(k))
    mu = randn(n_assets)
    gamma = 1.0

    # Generate QP problem
    P = sparse(blockdiag(D, sparse(I, k, k)))
    q = vcat(-mu ./ gamma, zeros(k))
    A = vcat(
        hcat(sparse(ones(1, n_assets)), spzeros(1, k)),
        hcat(F', -sparse(I, k, k)),
    )
    rl = vcat(1.0, zeros(k))

    lb = vcat(zeros(n_assets), -Inf * ones(k))
    ub = vcat(ones(n_assets), Inf * ones(k))


    # Generate JuMP model
    model = Model(Ipopt.Optimizer)
    @variable(model, x[1:n_assets])
    @variable(model, y[1:k])
    
    # Define the objective function correctly
    @objective(model, Min, sum(D[i,i]*x[i]^2 for i in 1:n_assets) + 
                                     sum(y[j]^2 for j in 1:k) - 
                                     (1 / gamma) * dot(mu, x))
    @constraint(model, sum(x) == 1)
    @constraint(model, F' * x .== y)
    @constraint(model, 0 .<= x .<= 1)

    optimize!(model)

end

# Example usage
 example = generate_portfolio_example(1000,1)

 function generate_portfolio_example(n::Int, seed::Int=1)
    Random.seed!(seed)
    
    n_assets = n 
    k = Int(n*10)
    F = sprandn(n_assets, k, 0.5)
    D = spdiagm(0 => rand(n_assets) .* sqrt(k))
    mu = randn(n_assets)
    gamma = 1.0

    # Generate QP problem
    P = sparse(blockdiag(D, sparse(I, k, k)))
    q = vcat(-mu ./ gamma, zeros(k))
    A = vcat(
        hcat(sparse(ones(1, n_assets)), spzeros(1, k)),
        hcat(F', -sparse(I, k, k)),
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
        A,
        A',
        ru,
        k+1,
    )
end

