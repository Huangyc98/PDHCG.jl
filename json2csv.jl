using JSON
using CSV
using DataFrames

# Setting the folder directory, write directory, and result file name.
folder_dir = "saved_results/QP/0"
write_dir = "saved_results/csv_result"
result_file = "output"

process_json_files(folder_dir, write_dir, result_file)

function process_json_files(folder_dir::String, write_dir::String, result_file::String)
    "
    This function reads all json files in the folder_dir and writes the selected data to a csv file.
    The selected data are instance_name, termination_string, iteration_count, solve_time_sec, and CG_total_iteration.
    The csv file is written in the write_dir with the name result_file.csv.
    "
    folder_path = folder_dir
    if !isdir(folder_path)
        mkdir(folder_path)
    end
    if !isdir(write_dir)
        mkdir(write_dir)
    end
    write_path = joinpath(write_dir, "$result_file.csv")

    file_names = readdir(folder_path)
    json_file_names = filter(x -> endswith(x, ".json"), file_names)
    file_num = length(json_file_names)

    all_data = DataFrame()

    for i = 1:file_num
        data = JSON.parsefile(joinpath(folder_path, json_file_names[i]))

        selected_data = DataFrame(
            instance_name = [data["instance_name"]],
            termination_string = [data["termination_string"]],
            iteration_count = [data["iteration_count"]],
            solve_time_sec = [data["solve_time_sec"]],
            CG_total_iteration = [data["CG_total_iteration"]]
        )

        all_data = vcat(all_data, selected_data)
    end

    CSV.write(write_path, all_data)
end