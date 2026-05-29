#pragma once
#include <nanobind/nanobind.h>
#include <map>
#include <utility>
#include <vector>

namespace uccl {
namespace nb = nanobind;

// Keyed by (device_index, proxy_mode). The bool is still part of the RDMA
// proxy resource key so old and native experiments do not collide on shared
// per-device thread pools or /dev/shm barriers during development.
using ProxyRegistryKey = std::pair<int, bool>;
extern std::map<ProxyRegistryKey, std::vector<nb::object>> g_proxies_by_dev;

std::map<ProxyRegistryKey, std::vector<nb::object>>& proxies_by_dev();
}  // namespace uccl
