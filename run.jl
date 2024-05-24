using PDHCG
#-------------Running Parameters--------------
#-----------------GPU setting-----------------
GPU_on = 0                     # 1: use GPU; 0: use CPU
GPU_id = 0                     # The GPU id if there are multiple GPUs (if there is only one GPU, set it to 0)
#-----------------Task setting----------------
folder_path = "./example/"     # The folder path of the problems
time_limit = 3600              # The time limit for each problem
relat = 1e-6                   # The relative tolerance for the solver
save_path = "saved_results/QP/$GPU_on"
#---------------------------------------------
#--------------------END----------------------

# Start solving the problems
file_names = readdir(folder_path)
problem_num = length(file_names)

for (i, file_name) in enumerate(file_names)
    println("Start solving the problem: $i, named: $file_name")
    ins_path = joinpath(folder_path, file_name)
    try
        PDHCG.run_solver(ins_path, save_path, GPU_on, GPU_id, time_limit, relat)
    catch e
        println("Failed to solve $file_name due to error: $e")
    end
end

function run_solver(file_path, save_path, use_gpu=0, GPU_id=0, time_limit=3600, relat=1e-6)
    "
     `file_path`: Path to the quadratic programming instance file.
     `save_path`: Directory where the output files will be saved.
     `use_gpu`: Enables GPU acceleration if set to 1; otherwise, it remains on CPU (default: 0).
     `GPU_id`: Identifies which GPU to use if GPU acceleration is enabled (default: 0).
     `time_limit`: Sets the maximum allowed time for the solver to run in seconds (default: 3600).
     `relat`: Specifies the solver's relative tolerance level (default: 1e-6).
     "
    project_scr = ["--project=scripts", "./test/solve_test.jl"]
    time_limit_arg = ["--time_sec_limit", "$time_limit"]
    relat_arg = ["--tolerance", "$relat"]
    gpu_option = use_gpu == 1 ? ["--use_gpu", "1"] : ["--use_gpu", "0"]
    out_dir = ["--output_dir", "$save_path"]
    file_path = ["--instance_path", file_path]
    if use_gpu == 1
        ENV["CUDA_VISIBLE_DEVICES"] = "$GPU_id"
    end

    local_problem = `julia $project_scr $file_path $out_dir $relat_arg $time_limit_arg $gpu_option`
    println("Running command: ", local_problem)
    try
        run(local_problem)
        return true
    catch e
        println("Failed to solve $file_path due to error: $e")
        return false
    end
end
