# Compile benchmark CSV files into C++ lookup tables.
function(dorado_compile_benchmarks)
    # Parse the args.
    set(options)
    set(oneValueArgs NAME OUTPUT)
    set(multiValueArgs FILES)
    cmake_parse_arguments(arg "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Save all the GPUs and models. We'll create variables on the fly to act as a map too.
    set(all_gpu_names)
    set(all_model_names)

    file(WRITE ${arg_OUTPUT} "#include \"compiled_timings.h\"\nnamespace dorado::batchsize_benchmarks::compiled_cache {\n")
    foreach(csv_file IN LISTS arg_FILES)
        # If the file changes we want to re-run this script.
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${csv_file}")

        # Parse the file line by line.
        file(STRINGS "${csv_file}" csv_lines)
        foreach(csv_line IN LISTS csv_lines)
            # Break the line up.
            if (csv_line MATCHES "^#")
                # Ignore comments.
                continue()
            elseif (NOT csv_line MATCHES "^([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)$")
                message(FATAL_ERROR "Failed to parse line: ${csv_line}")
                continue()
            endif()
            set(gpu_name "${CMAKE_MATCH_1}")
            set(model_name "${CMAKE_MATCH_2}")
            set(batchsize "${CMAKE_MATCH_3}")
            set(speed "${CMAKE_MATCH_4}")
            set(memory "${CMAKE_MATCH_5}")

            # Add the GPU and model to the full list.
            list(APPEND all_gpu_names "${gpu_name}")
            list(APPEND all_model_names "${model_name}")

            # Create the entry in our map.
            dorado_sanitize_name(gpu_name_sanitized "${gpu_name}")
            dorado_sanitize_name(model_name_sanitized "${model_name}")
            set(key "${gpu_name_sanitized}__${model_name_sanitized}")
            list(APPEND "sizes__${key}" "${batchsize}")
            set("speed_${batchsize}__${key}" "${speed}")
            set("memory_${batchsize}__${key}" "${memory}")
        endforeach()
    endforeach()

    # Trim the GPUs and models.
    list(REMOVE_DUPLICATES all_gpu_names)
    list(REMOVE_DUPLICATES all_model_names)

    # Write out the entries for each GPU+model.
    foreach(gpu_name IN LISTS all_gpu_names)
        dorado_sanitize_name(gpu_name_sanitized "${gpu_name}")
        set(gpu_models)

        foreach(model_name IN LISTS all_model_names)
            dorado_sanitize_name(model_name_sanitized "${model_name}")

            # See if there are entries for this GPU+model.
            set(key "${gpu_name_sanitized}__${model_name_sanitized}")
            if (NOT DEFINED "sizes__${key}")
                continue()
            endif()
            list(APPEND gpu_models "${model_name}")

            # Make sure they're sorted.
            set(sizes "${sizes__${key}}")
            list(SORT sizes COMPARE NATURAL)

            # Write the entries out.
            file(APPEND ${arg_OUTPUT} "constexpr SpeedEntry gpu_${gpu_name_sanitized}_and_model_${model_name_sanitized}[] = {\n")
            foreach(batchsize IN LISTS sizes)
                set(speed_var "speed_${batchsize}__${key}")
                set(memory_var "memory_${batchsize}__${key}")
                file(APPEND ${arg_OUTPUT} "  { ${batchsize}, ${${speed_var}}, ${${memory_var}} },\n")
            endforeach()
            file(APPEND ${arg_OUTPUT} "};\nstatic_assert(is_sorted(gpu_${gpu_name_sanitized}_and_model_${model_name_sanitized}));\n")
        endforeach()

        # Write out the models for this GPU.
        file(APPEND ${arg_OUTPUT} "constexpr ModelTimings ${gpu_name_sanitized}_models[] = {\n")
        foreach(model_name IN LISTS gpu_models)
            dorado_sanitize_name(model_name_sanitized "${model_name}")
            file(APPEND ${arg_OUTPUT} "  { \"${model_name}\" , gpu_${gpu_name_sanitized}_and_model_${model_name_sanitized} },\n")
        endforeach()
        file(APPEND ${arg_OUTPUT} "};\n")
    endforeach()

    # Write out all the gpus.
    file(APPEND ${arg_OUTPUT} "constexpr GPUModelTimings all_gpus[] = {\n")
    foreach(gpu_name IN LISTS all_gpu_names)
        dorado_sanitize_name(gpu_name_sanitized "${gpu_name}")
        file(APPEND ${arg_OUTPUT} "  { \"${gpu_name}\", ${gpu_name_sanitized}_models },\n")
    endforeach()
    file(APPEND ${arg_OUTPUT} "};\nstd::span<const GPUModelTimings> get() { return all_gpus; }\n}\n")
endfunction()

# Replace characters that break identifiers in C++
function(dorado_sanitize_name OUTPUT NAME)
    string(REGEX REPLACE "[^A-Za-z0-9_]" "_" NAME ${NAME})
    # Remove double underscores because they're a reserved identifier in C++, and
    # because we use them to emulate a map in cmake.
    while(NAME MATCHES "__")
        string(REGEX REPLACE "__" "_" NAME ${NAME})
    endwhile()
    set(${OUTPUT} "${NAME}" PARENT_SCOPE)
endfunction()
