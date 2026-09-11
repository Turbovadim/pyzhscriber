// Compare Conformer views and Metal depthwise layouts with the former graphs.
// Run separately with: cpu | metal-shared | metal-private
#include "conformer/conformer.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-metal.h"
#include "ggml.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using Scheduler = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

// Keep the old operations here so the reference stays independent of the helper.
ggml_tensor * old_shift(ggml_context * ctx, ggml_tensor * input, int64_t n_kv) {
    const int64_t positions = input->ne[0];
    const int64_t queries = input->ne[1];
    const int64_t heads = input->ne[2];
    const int64_t batch = input->ne[3];
    auto * zeros = ggml_fill(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, queries, heads, batch), 0.0f);
    auto * shifted = ggml_concat(ctx, zeros, input, 0);
    shifted = ggml_reshape_4d(ctx, shifted, queries, positions + 1, heads, batch);
    shifted = ggml_view_4d(ctx, shifted, queries, positions, heads, batch,
                           shifted->nb[1], shifted->nb[2], shifted->nb[3], shifted->nb[1]);
    shifted = ggml_cont(ctx, shifted);
    shifted = ggml_reshape_4d(ctx, shifted, positions, queries, heads, batch);
    return ggml_view_4d(ctx, shifted, n_kv, queries, heads, batch,
                        shifted->nb[1], shifted->nb[2], shifted->nb[3], 0);
}

std::vector<float> read_tensor(ggml_tensor * tensor) {
    require(ggml_is_contiguous(tensor), "readback tensor is not contiguous");
    std::vector<float> result(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size() * sizeof(float));
    return result;
}

void compare(const std::vector<float> & actual, const std::vector<float> & expected,
             const std::string & label) {
    require(actual.size() == expected.size(), label + ": length differs");
    if (std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0) return;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::memcmp(&actual[i], &expected[i], sizeof(float)) != 0) {
            throw std::runtime_error(label + ": first differing element " + std::to_string(i) +
                                     " actual=" + std::to_string(actual[i]) +
                                     " expected=" + std::to_string(expected[i]));
        }
    }
}

Scheduler allocate_graph(ggml_backend_t backend, ggml_backend_t cpu_fallback, ggml_cgraph * graph,
                         std::initializer_list<ggml_tensor *> inputs, const std::string & label) {
    // The scheduler requires a CPU backend last. Assign every node explicitly
    // and verify support so fallback cannot hide a broken Metal stride path.
    ggml_backend_t backends[] = {backend, cpu_fallback};
    Scheduler scheduler(ggml_backend_sched_new(backends, nullptr, cpu_fallback ? 2 : 1, 128, false, true),
                        ggml_backend_sched_free);
    require(scheduler != nullptr, label + ": scheduler allocation failed");
    for (auto * input : inputs) ggml_backend_sched_set_tensor_backend(scheduler.get(), input, backend);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * node = ggml_graph_node(graph, i);
        require(ggml_backend_supports_op(backend, node), label + ": backend does not support graph node");
        ggml_backend_sched_set_tensor_backend(scheduler.get(), node, backend);
    }
    require(ggml_backend_sched_alloc_graph(scheduler.get(), graph), label + ": graph allocation failed");
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        require(ggml_backend_sched_get_tensor_backend(scheduler.get(), ggml_graph_node(graph, i)) == backend,
                label + ": graph node assigned to a different backend");
    }
    return scheduler;
}

void check_shape(ggml_backend_t backend, ggml_backend_t cpu_fallback,
                 int64_t queries, int64_t keys, int64_t heads, int64_t batch) {
    const std::string label = "q=" + std::to_string(queries) + " k=" + std::to_string(keys) +
                              " h=" + std::to_string(heads) + " b=" + std::to_string(batch);
    const int64_t positions = queries + keys - 1;
    Context context(ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
    require(context != nullptr, label + ": context allocation failed");
    auto * input = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, positions, queries, heads, batch);
    auto * addend = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, keys, queries, heads, batch);
    ggml_set_name(input, "relative.scores");
    ggml_set_name(addend, "content.scores");
    ggml_set_input(input);
    ggml_set_input(addend);

    auto * old_view = old_shift(context.get(), input, keys);
    auto * new_view = transcribe::conformer::rel_shift_view(context.get(), input, keys);
    require(new_view != nullptr, label + ": cropped view missing");
    require(new_view->op == GGML_OP_VIEW && new_view->view_src == input,
            label + ": cropped result does not directly view the original scores");
    require(ggml_are_same_shape(old_view, new_view) && ggml_is_contiguous_rows(new_view),
            label + ": cropped shape or contiguous-row contract changed");

    // CONT is required by the flash path. ADD reads the strided view directly
    // in manual attention, with both operand orders checked here.
    std::array<ggml_tensor *, 6> outputs = {
        ggml_cont(context.get(), old_view),
        ggml_cont(context.get(), new_view),
        ggml_add(context.get(), addend, old_view),
        ggml_add(context.get(), addend, new_view),
        ggml_add(context.get(), old_view, addend),
        ggml_add(context.get(), new_view, addend),
    };
    auto * graph = ggml_new_graph_custom(context.get(), 128, false);
    for (auto * output : outputs) {
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
    }

    auto scheduler = allocate_graph(backend, cpu_fallback, graph, {input, addend}, label);

    const size_t n_input = static_cast<size_t>(positions * queries * heads * batch);
    const size_t n_output = static_cast<size_t>(keys * queries * heads * batch);
    std::vector<float> source(n_input), additions(n_output), expected(n_output), expected_add(n_output);
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 0; i < source.size(); ++i) {
            source[i] = pass == 0 ? static_cast<float>(i + 1) * 0.125f
                                  : static_cast<float>(static_cast<int>(i % 4093) - 2046) * 0.25f;
            if (pass == 1 && i % 17 == 0) source[i] = -std::numeric_limits<float>::infinity();
        }
        for (size_t i = 0; i < additions.size(); ++i) {
            additions[i] = static_cast<float>(static_cast<int>(i % 31) - 15) * 0.125f;
        }
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h = 0; h < heads; ++h) {
                for (int64_t q = 0; q < queries; ++q) {
                    for (int64_t k = 0; k < keys; ++k) {
                        const auto dst = static_cast<size_t>(((b * heads + h) * queries + q) * keys + k);
                        const int64_t relative = k - q + queries - 1;
                        require(relative >= 0 && relative < positions, label + ": oracle index outside row");
                        const auto src = static_cast<size_t>(((b * heads + h) * queries + q) * positions + relative);
                        expected[dst] = source[src];
                        expected_add[dst] = source[src] + additions[dst];
                    }
                }
            }
        }
        ggml_backend_tensor_set(input, source.data(), 0, source.size() * sizeof(float));
        ggml_backend_tensor_set(addend, additions.data(), 0, additions.size() * sizeof(float));
        require(ggml_backend_sched_graph_compute(scheduler.get(), graph) == GGML_STATUS_SUCCESS,
                label + ": graph compute failed");
        for (size_t i = 0; i < outputs.size(); ++i) {
            compare(read_tensor(outputs[i]), i < 2 ? expected : expected_add,
                    label + " pass=" + std::to_string(pass) + " output=" + std::to_string(i));
        }
    }
}

void check_position_matmul(ggml_backend_t backend, ggml_backend_t cpu_fallback,
                           int64_t head_dim, int64_t frames, int64_t heads, int64_t batch) {
    const std::string label = "position matmul d=" + std::to_string(head_dim) +
                              " t=" + std::to_string(frames) + " h=" + std::to_string(heads) +
                              " b=" + std::to_string(batch);
    const int64_t positions = 2 * frames - 1;
    Context context(ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
    require(context != nullptr, label + ": context allocation failed");
    auto * p_input = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, head_dim, heads, positions, 1);
    auto * q_input = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, head_dim, heads, frames, batch);
    ggml_set_name(p_input, "position.projection");
    ggml_set_name(q_input, "query.position.bias");
    ggml_set_input(p_input);
    ggml_set_input(q_input);

    auto * p_view = ggml_permute(context.get(), p_input, 0, 2, 1, 3);
    auto * q_view = ggml_permute(context.get(), q_input, 0, 2, 1, 3);
    auto * old_output = ggml_mul_mat(context.get(), ggml_cont(context.get(), p_view),
                                     ggml_cont(context.get(), q_view));
    auto * new_output = ggml_mul_mat(context.get(), p_view, q_view);
    ggml_set_output(old_output);
    ggml_set_output(new_output);
    auto * graph = ggml_new_graph_custom(context.get(), 128, false);
    ggml_build_forward_expand(graph, old_output);
    ggml_build_forward_expand(graph, new_output);
    auto scheduler = allocate_graph(backend, cpu_fallback, graph, {p_input, q_input}, label);

    std::vector<float> p(static_cast<size_t>(ggml_nelements(p_input)));
    std::vector<float> q(static_cast<size_t>(ggml_nelements(q_input)));
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 0; i < p.size(); ++i) {
            p[i] = static_cast<float>(static_cast<int>((i * 37 + pass * 23) % 257) - 128) / 113.0f;
        }
        for (size_t i = 0; i < q.size(); ++i) {
            q[i] = static_cast<float>(static_cast<int>((i * 53 + pass * 41) % 263) - 131) / 127.0f;
        }
        ggml_backend_tensor_set(p_input, p.data(), 0, p.size() * sizeof(float));
        ggml_backend_tensor_set(q_input, q.data(), 0, q.size() * sizeof(float));
        require(ggml_backend_sched_graph_compute(scheduler.get(), graph) == GGML_STATUS_SUCCESS,
                label + ": graph compute failed");
        compare(read_tensor(new_output), read_tensor(old_output), label + " pass=" + std::to_string(pass));
    }
}

void check_depthwise_layout(ggml_backend_t backend, ggml_backend_t cpu_fallback,
                            int64_t kernel_size, int64_t frames, int64_t channels, int64_t batch) {
    const std::string label = "depthwise k=" + std::to_string(kernel_size) +
                              " t=" + std::to_string(frames) + " c=" + std::to_string(channels) +
                              " b=" + std::to_string(batch);
    Context context(ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
    require(context != nullptr, label + ": context allocation failed");
    auto * input = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, channels, frames, batch);
    auto * weights = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, kernel_size, 1, 1, channels);
    ggml_set_name(input, "depthwise.input");
    ggml_set_name(weights, "depthwise.weights");
    ggml_set_input(input);
    ggml_set_input(weights);

    auto * old_input = ggml_cont(context.get(), ggml_permute(context.get(), input, 1, 0, 2, 3));
    old_input = ggml_reshape_4d(context.get(), old_input, frames, 1, channels, batch);
    auto * new_input = ggml_reshape_4d(context.get(), input, channels, 1, frames, batch);
    new_input = ggml_permute(context.get(), new_input, 2, 1, 0, 3);
    require(ggml_are_same_shape(old_input, new_input), label + ": input shape changed");
    require(old_input->nb[2] >= old_input->nb[0] && new_input->nb[2] < new_input->nb[0],
            label + ": input strides do not select different Metal depthwise kernels");

    const int padding = static_cast<int>((kernel_size - 1) / 2);
    auto * old_output = ggml_conv_2d_dw_direct(context.get(), weights, old_input, 1, 1, padding, 0, 1, 1);
    auto * new_output = ggml_conv_2d_dw_direct(context.get(), weights, new_input, 1, 1, padding, 0, 1, 1);
    require(ggml_are_same_shape(old_output, new_output) && old_output->ne[0] == frames,
            label + ": output shape changed");
    ggml_set_output(old_output);
    ggml_set_output(new_output);
    auto * graph = ggml_new_graph_custom(context.get(), 128, false);
    ggml_build_forward_expand(graph, old_output);
    ggml_build_forward_expand(graph, new_output);
    auto scheduler = allocate_graph(backend, cpu_fallback, graph, {input, weights}, label);

    std::vector<float> samples(static_cast<size_t>(ggml_nelements(input)));
    std::vector<float> kernel(static_cast<size_t>(ggml_nelements(weights)));
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 0; i < samples.size(); ++i) {
            samples[i] = static_cast<float>(static_cast<int>((i * 29 + pass * 71) % 251) - 125) / 97.0f;
        }
        for (size_t i = 0; i < kernel.size(); ++i) {
            kernel[i] = static_cast<float>(static_cast<int>((i * 43 + pass * 17) % 211) - 105) / 113.0f;
        }
        ggml_backend_tensor_set(input, samples.data(), 0, samples.size() * sizeof(float));
        ggml_backend_tensor_set(weights, kernel.data(), 0, kernel.size() * sizeof(float));
        require(ggml_backend_sched_graph_compute(scheduler.get(), graph) == GGML_STATUS_SUCCESS,
                label + ": graph compute failed");
        compare(read_tensor(new_output), read_tensor(old_output), label + " pass=" + std::to_string(pass));
    }
}

void run(const std::string & mode) {
    const bool metal = mode != "cpu";
    const bool shared = mode == "metal-shared";
    if (metal) {
        unsetenv(shared ? "GGML_METAL_SHARED_BUFFERS_DISABLE" : "GGML_METAL_SHARED_BUFFERS_ENABLE");
        setenv(shared ? "GGML_METAL_SHARED_BUFFERS_ENABLE" : "GGML_METAL_SHARED_BUFFERS_DISABLE", "1", 1);
    }
    Backend backend(metal ? ggml_backend_metal_init() : ggml_backend_cpu_init(), ggml_backend_free);
    require(backend != nullptr, "backend initialization failed");
    Backend cpu_fallback(metal ? ggml_backend_cpu_init() : nullptr, ggml_backend_free);
    if (metal) {
        require(cpu_fallback != nullptr, "CPU fallback initialization failed");
        require(ggml_backend_metal_buffer_type_is_shared(ggml_backend_get_default_buffer_type(backend.get())) == shared,
                "Metal buffer mode does not match requested mode");
    } else {
        ggml_backend_cpu_set_n_threads(backend.get(), 2);
    }

    constexpr std::array<std::array<int64_t, 2>, 25> geometries = {{
        {1, 1}, {2, 2}, {3, 3}, {7, 7}, {8, 8}, {9, 9},
        {15, 15}, {16, 16}, {17, 17}, {31, 31}, {32, 32}, {33, 33},
        {63, 63}, {64, 64}, {65, 65}, {127, 127}, {128, 128}, {129, 129},
        {1, 65}, {65, 1}, {2, 3}, {3, 2}, {7, 33}, {33, 7}, {63, 65},
    }};
    constexpr std::array<std::array<int64_t, 2>, 3> groups = {{{1, 1}, {8, 1}, {3, 2}}};
    for (const auto & shape : geometries) {
        for (const auto & group : groups) {
            check_shape(backend.get(), cpu_fallback.get(), shape[0], shape[1], group[0], group[1]);
        }
    }
    constexpr std::array<int64_t, 10> frame_counts = {1, 7, 8, 9, 31, 32, 33, 63, 64, 65};
    constexpr std::array<int64_t, 2> head_dims = {32, 128};
    constexpr std::array<std::array<int64_t, 2>, 2> matmul_groups = {{{1, 1}, {8, 2}}};
    for (const auto head_dim : head_dims) {
        for (const auto frames : frame_counts) {
            for (const auto & group : matmul_groups) {
                check_position_matmul(backend.get(), cpu_fallback.get(), head_dim, frames, group[0], group[1]);
            }
        }
    }
    size_t depthwise_cases = 0;
    if (metal) {
        for (const int64_t kernel : {9, 31}) {
            for (const int64_t frames : {1, 7, 32, 65, 129}) {
                for (const int64_t channels : {32, 1024}) {
                    for (const int64_t batch : {1, 2}) {
                        check_depthwise_layout(backend.get(), cpu_fallback.get(), kernel, frames, channels, batch);
                        ++depthwise_cases;
                    }
                }
            }
        }
    }
    std::printf("PASS %s: %zu relative-position shapes and %zu head-strided matmul shapes, "
                "%zu Metal depthwise layouts, two input passes, bitwise outputs\n", mode.c_str(),
                geometries.size() * groups.size(), head_dims.size() * frame_counts.size() * matmul_groups.size(),
                depthwise_cases);
}
}  // namespace

int main(int argc, char ** argv) {
    try {
        require(argc == 2, "usage: conformer cpu|metal-shared|metal-private");
        const std::string mode(argv[1]);
        require(mode == "cpu" || mode == "metal-shared" || mode == "metal-private", "unknown backend mode");
        run(mode);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL conformer: %s\n", error.what());
        return 1;
    }
}
