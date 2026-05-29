#include <stdexcept>
#include <string>

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace {

constexpr const char* kRewriteMessage =
    "uccl-ep is being rewritten as a native DeepEP V2 AWS EFA backend. "
    "The old V1 static internode/intranode kernels have been removed from "
    "the public runtime. Implement V2EfaRuntime dispatch/combine through "
    "JIT .cuh kernels before running benchmarks.";

struct Config {
  int num_sms;
  int num_max_nvl_chunked_send_tokens;
  int num_max_nvl_chunked_recv_tokens;
  int num_max_rdma_chunked_send_tokens;
  int num_max_rdma_chunked_recv_tokens;

  Config(int num_sms = 0, int num_max_nvl_chunked_send_tokens = 0,
         int num_max_nvl_chunked_recv_tokens = 0,
         int num_max_rdma_chunked_send_tokens = 0,
         int num_max_rdma_chunked_recv_tokens = 0)
      : num_sms(num_sms),
        num_max_nvl_chunked_send_tokens(num_max_nvl_chunked_send_tokens),
        num_max_nvl_chunked_recv_tokens(num_max_nvl_chunked_recv_tokens),
        num_max_rdma_chunked_send_tokens(num_max_rdma_chunked_send_tokens),
        num_max_rdma_chunked_recv_tokens(num_max_rdma_chunked_recv_tokens) {}
};

struct EventHandle {
  void current_stream_wait() const {}
};

class V2EfaRuntime {
 public:
  V2EfaRuntime() = default;

  bool is_ready() const { return false; }

  void init() const { fail(); }
  void launch_dispatch() const { fail(); }
  void launch_combine() const { fail(); }

 private:
  static void fail() { throw std::runtime_error(kRewriteMessage); }
};

}  // namespace

NB_MODULE(ep, m) {
  m.doc() = "DeepEP V2 AWS EFA native backend skeleton";
  m.attr("__native_v2_rewrite__") = true;
  m.attr("__native_v2_ready__") = false;

  nb::class_<Config>(m, "Config")
      .def(nb::init<int, int, int, int, int>(), nb::arg("num_sms") = 0,
           nb::arg("num_max_nvl_chunked_send_tokens") = 0,
           nb::arg("num_max_nvl_chunked_recv_tokens") = 0,
           nb::arg("num_max_rdma_chunked_send_tokens") = 0,
           nb::arg("num_max_rdma_chunked_recv_tokens") = 0)
      .def_rw("num_sms", &Config::num_sms)
      .def_rw("num_max_nvl_chunked_send_tokens",
              &Config::num_max_nvl_chunked_send_tokens)
      .def_rw("num_max_nvl_chunked_recv_tokens",
              &Config::num_max_nvl_chunked_recv_tokens)
      .def_rw("num_max_rdma_chunked_send_tokens",
              &Config::num_max_rdma_chunked_send_tokens)
      .def_rw("num_max_rdma_chunked_recv_tokens",
              &Config::num_max_rdma_chunked_recv_tokens);

  nb::class_<EventHandle>(m, "EventHandle")
      .def(nb::init<>())
      .def("current_stream_wait", &EventHandle::current_stream_wait);

  nb::class_<V2EfaRuntime>(m, "V2EfaRuntime")
      .def(nb::init<>())
      .def("is_ready", &V2EfaRuntime::is_ready)
      .def("init", &V2EfaRuntime::init)
      .def("launch_dispatch", &V2EfaRuntime::launch_dispatch)
      .def("launch_combine", &V2EfaRuntime::launch_combine);

  m.def("is_native_v2_ready", []() { return false; });
  m.def("native_v2_rewrite_message", []() { return std::string(kRewriteMessage); });
}
