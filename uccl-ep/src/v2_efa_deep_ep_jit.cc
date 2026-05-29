#include "v2_efa/runtime.hpp"

#include <mutex>
#include <stdexcept>
#include <string>

#include "../../csrc/jit/compiler.hpp"
#include "../../csrc/jit/include_parser.hpp"
#include "../../csrc/jit/kernel_runtime.hpp"

namespace uccl::v2_efa {

namespace {

std::once_flag g_jit_init_once;
bool g_jit_initialized = false;

}  // namespace

void init_deep_ep_jit_bridge(const std::string& library_root_path,
                             const std::string& cuda_home_path,
                             const std::string& nccl_root_path) {
  std::call_once(g_jit_init_once, [&] {
    deep_ep::jit::Compiler::prepare_init(library_root_path, cuda_home_path,
                                         nccl_root_path);
    deep_ep::jit::KernelRuntime::prepare_init(cuda_home_path);
    deep_ep::jit::IncludeParser::prepare_init(library_root_path);
    g_jit_initialized = true;
  });
}

bool is_deep_ep_jit_bridge_initialized() { return g_jit_initialized; }

void compile_v2_efa_jit_plan(const V2EfaJitLaunchPlan& plan) {
  if (!is_deep_ep_jit_bridge_initialized()) {
    throw std::runtime_error(
        "DeepEP JIT bridge is not initialized; call init_deep_ep_jit first");
  }
  if (plan.name.empty() || plan.source.empty()) {
    throw std::invalid_argument("empty V2 EFA JIT plan");
  }
  const auto runtime = deep_ep::jit::compiler->build(plan.name, plan.source);
  if (runtime == nullptr) {
    throw std::runtime_error("DeepEP JIT compiler returned null runtime");
  }
}

}  // namespace uccl::v2_efa
