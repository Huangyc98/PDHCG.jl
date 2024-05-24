push!(LOAD_PATH, "src/.")
import ArgParse
import GZip
import JSON3
import CUDA
import PDHCG
using LinearAlgebra

function write_vector_to_file(filename, vector)
    open(filename, "w") do io
      for x in vector
        println(io, x)
      end
    end
end

function solve_instance_and_output(
    parameters::PDHCG.PdhgParameters,
    instance_path::String,
    gpu_flag::Bool,
)
  
    instance_name = replace(basename(instance_path), r"\.(mps|MPS|qps|QPS)(\.gz)?$" => "")
  
    function inner_solve()
        lower_file_name = lowercase(basename(instance_path))
        if endswith(lower_file_name, ".mps") ||
            endswith(lower_file_name, ".mps.gz") ||
            endswith(lower_file_name, ".qps") ||
            endswith(lower_file_name, ".qps.gz")
            qp = PDHCG.qps_reader_to_standard_form(instance_path)
        else
            error(
                "Instance has unrecognized file extension: ", 
                basename(instance_path),
            )
        end
    
        if parameters.verbosity >= 1
            println("Instance: ", instance_name)
        end

        if gpu_flag
            output = PDHCG.optimize_gpu(parameters, qp)
        else
            output = PDHCG.optimize(parameters, qp)
        end

        log = PDHCG.SolveLog()
        log.instance_name = instance_name
        log.command_line_invocation = join([PROGRAM_FILE; ARGS...], " ")
        log.termination_reason = output.termination_reason
        log.termination_string = output.termination_string
        log.iteration_count = output.iteration_count
        log.CG_total_iteration = output.CG_total_iteration
        log.solve_time_sec = output.iteration_stats[end].cumulative_time_sec
        log.solution_stats = output.iteration_stats[end]
        kkt_error =  Vector{Float64}()
        for i = 1:length(output.iteration_stats)
            c_i_current = output.iteration_stats[i].convergence_information[1]
            current_kkt_err = norm([c_i_current.relative_optimality_gap, c_i_current.relative_l2_primal_residual, c_i_current.relative_l2_dual_residual])
            
            push!(kkt_error,current_kkt_err)
        end
        log.kkt_error = kkt_error

        log.solution_type = PDHCG.POINT_TYPE_AVERAGE_ITERATE
 
        log.iteration_stats = output.iteration_stats

    end     

    inner_solve()
   
    return
end

function parse_command_line()
    arg_parse = ArgParse.ArgParseSettings()

    ArgParse.@add_arg_table! arg_parse begin
        "--instance_path"
        help = "The path to the instance to solve in .mps.gz or .mps format."
        arg_type = String

        "--tolerance"
        help = "KKT tolerance of the solution."
        arg_type = Float64
        default = 1e-3

        "--time_sec_limit"
        help = "Time limit."
        arg_type = Float64
        default = 3600.0

        "--use_gpu"
        help = "Using GPU: 0-false, 1-true"
        arg_type = Int64
        default = 0
    end

    return ArgParse.parse_args(arg_parse)
end


function main()
    parsed_args = parse_command_line()
    instance_path = parsed_args["instance_path"]
    tolerance = parsed_args["tolerance"]
    time_sec_limit = parsed_args["time_sec_limit"]
    gpu_flag = Bool(parsed_args["use_gpu"])

    if gpu_flag && !CUDA.functional()
        error("CUDA not found when --use_gpu=1")
    end

    qp = PDHCG.qps_reader_to_standard_form(instance_path)

    oldstd = stdout
    redirect_stdout(devnull)
    redirect_stdout(oldstd)

    restart_params = PDHCG.construct_restart_parameters(
        PDHCG.ADAPTIVE_KKT,    # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
        PDHCG.KKT_GREEDY,      # NO_RESTART_TO_CURRENT KKT_GREEDY
        1000,                   # restart_frequency_if_fixed
        0.2,                   # artificial_restart_threshold
        0.2,                    # sufficient_reduction_for_restart
        0.8,                    # necessary_reduction_for_restart
        0.2,                    # primal_weight_update_smoothing
    )

    termination_params = PDHCG.construct_termination_criteria(
        # optimality_norm = L2,
        eps_optimal_absolute = tolerance,
        eps_optimal_relative = tolerance,
        time_sec_limit = time_sec_limit,
        iteration_limit = typemax(Int32),
        kkt_matrix_pass_limit = Inf,
    )

    params = PDHCG.PdhgParameters(
        10,
        false,
        1.0,
        1.0,
        false,
        2,
        true,
        40,
        termination_params,
        restart_params,
        PDHCG.ConstantStepsizeParams(),  
    )

    solve_instance_and_output(
        params,
        instance_path,
        gpu_flag,
    )

end

main()
