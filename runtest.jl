using Test
using JSON3
using PDHCG
#-------------Running Parameters--------------
#-----------------GPU setting-----------------
GPU_id = 0                     # The GPU id if there are multiple GPUs (if there is only one GPU, set it to 0)
#-----------------Task setting----------------
folder_path = "./example/"     # The folder path of the problems
time_limit = 3600              # The time limit for each problem
relat = 1e-3                   # The relative tolerance for the solver
#---------------------------------------------
#--------------------END----------------------

# Function to run the solver
function run_solver(file_name, use_gpu, time_limit=3600, relat=1e-6, GPU_id=0)
    project_scr = ["--project=scripts", "./test/solve_test.jl"]
    time_limit_arg = ["--time_sec_limit", "$time_limit"]
    relat_arg = ["--tolerance", "$relat"]
    gpu_option = use_gpu == 1 ? ["--use_gpu", "1"] : ["--use_gpu", "0"]
    
    if use_gpu == 1
        ENV["CUDA_VISIBLE_DEVICES"] = "$GPU_id"
    end

    ins_path = ["--instance_path", joinpath(folder_path, file_name)]
    local_problem = `julia $project_scr $ins_path $relat_arg $time_limit_arg $gpu_option`
    println("Running command: ", local_problem)
    try
        run(local_problem)
        return true
    catch e
        println("Failed to solve $file_name due to error: $e")
        return false
    end
end

# Read the first problem file
file_names = readdir(folder_path)
first_file = file_names[1]
@testset "Solver Test" begin
    println("Testing CPU...")
    cpu_success = run_solver(first_file, 0, time_limit, relat)
    @test cpu_success == true
    
    if !cpu_success
        println("CPU run failed")
    else
        println("CPU run success")
    end

    println("Testing GPU...")
    gpu_success = run_solver(first_file, 1, time_limit, relat)
    @test gpu_success == true
    
    if !gpu_success
        println("GPU run failed")
    else
        println("GPU run success")
    end

    println("\n--- Solver Test Summary ---")
    println("CPU test: $(cpu_success ? "SUCCESS" : "FAILURE")")
    println("GPU test: $(gpu_success ? "SUCCESS" : "FAILURE")")
    println("---------------------------")
end