// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <core_descriptor.hpp>
#include <device.hpp>
#include <device_pool.hpp>
#include <dispatch_core_common.hpp>
#include <host_api.hpp>
#include <profiler.hpp>
#include <mesh_workload.hpp>
#include <mesh_command_queue.hpp>
#include <tt_metal.hpp>
#include <tt_metal_profiler.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "assert.hpp"
#include "buffer.hpp"
#include "core_coord.hpp"
#include "data_types.hpp"
#include "dev_msgs.h"
#include "hal_types.hpp"
#include "hostdevcommon/profiler_common.h"
#include "impl/context/metal_context.hpp"
#include "kernel_types.hpp"
#include "llrt.hpp"
#include "llrt/hal.hpp"
#include <tt-logger/tt-logger.hpp>
#include "metal_soc_descriptor.h"
#include "profiler_optional_metadata.hpp"
#include "profiler_paths.hpp"
#include "profiler_state.hpp"
#include "profiler_types.hpp"
#include "tools/profiler/noc_event_profiler_utils.hpp"
#include "tools/profiler/event_metadata.hpp"
#include "tt-metalium/program.hpp"
#include <tt-metalium/device_pool.hpp>
#include "rtoptions.hpp"
#include "tracy/Tracy.hpp"
#include "tracy/TracyTTDevice.hpp"
#include <tt-metalium/distributed.hpp>
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/tt_xy_pair.h>
#include <umd/device/types/arch.h>
#include <umd/device/types/cluster_descriptor_types.h>
#include <umd/device/types/xy_pair.h>

namespace tt {

namespace tt_metal {

namespace detail {

std::unordered_map<chip_id_t, DeviceProfiler> tt_metal_device_profiler_map;

std::unordered_map<chip_id_t, std::vector<std::pair<uint64_t, uint64_t>>> deviceHostTimePair;
std::unordered_map<chip_id_t, uint64_t> smallestHostime;

std::unordered_map<chip_id_t, std::unordered_map<chip_id_t, std::vector<std::pair<uint64_t, uint64_t>>>>
    deviceDeviceTimePair;

bool do_sync_on_close = true;
std::unordered_set<chip_id_t> sync_set_devices;
constexpr CoreCoord SYNC_CORE = {0, 0};

// 32bit FNV-1a hashing
uint32_t hash32CT(const char* str, size_t n, uint32_t basis) {
    return n == 0 ? basis : hash32CT(str + 1, n - 1, (basis ^ str[0]) * UINT32_C(16777619));
}

// XORe'd 16-bit FNV-1a hashing functions
uint16_t hash16CT(const std::string& str) {
    uint32_t res = hash32CT(str.c_str(), str.length(), UINT32_C(2166136261));
    return ((res & 0xFFFF) ^ ((res & 0xFFFF0000) >> 16)) & 0xFFFF;
}

void populateZoneSrcLocations(
    const std::string& new_log_name,
    const std::string& log_name,
    bool push_new,
    std::unordered_map<uint16_t, ZoneDetails>& hash_to_zone_src_locations,
    std::unordered_set<std::string>& zone_src_locations) {
    std::ifstream log_file_read(new_log_name);
    std::string line;
    while (std::getline(log_file_read, line)) {
        std::string delimiter = "'#pragma message: ";
        int delimiter_index = line.find(delimiter) + delimiter.length();
        std::string zone_src_location = line.substr(delimiter_index, line.length() - delimiter_index - 1);

        uint16_t hash_16bit = hash16CT(zone_src_location);

        auto did_insert = zone_src_locations.insert(zone_src_location);
        if (did_insert.second && (hash_to_zone_src_locations.find(hash_16bit) != hash_to_zone_src_locations.end())) {
            TT_THROW("Source location hashes are colliding, two different locations are having the same hash");
        }

        std::stringstream ss(zone_src_location);
        std::string zone_name;
        std::string source_file;
        std::string line_num_str;
        std::getline(ss, zone_name, ',');
        std::getline(ss, source_file, ',');
        std::getline(ss, line_num_str, ',');

        ZoneDetails details(zone_name, source_file, std::stoull(line_num_str));

        auto ret = hash_to_zone_src_locations.emplace(hash_16bit, details);
        if (ret.second && push_new) {
            std::ofstream log_file_write(log_name, std::ios::app);
            log_file_write << line << std::endl;
            log_file_write.close();
        }
    }
    log_file_read.close();
}

// Iterate through all zone source locations and generate hash
std::unordered_map<uint16_t, ZoneDetails> generateZoneSourceLocationsHashes() {
    std::unordered_map<uint16_t, ZoneDetails> hash_to_zone_src_locations;
    std::unordered_set<std::string> zone_src_locations;

    // Load existing zones from previous runs
    populateZoneSrcLocations(
        tt::tt_metal::PROFILER_ZONE_SRC_LOCATIONS_LOG, "", false, hash_to_zone_src_locations, zone_src_locations);

    // Load new zones from the current run
    populateZoneSrcLocations(
        tt::tt_metal::NEW_PROFILER_ZONE_SRC_LOCATIONS_LOG,
        tt::tt_metal::PROFILER_ZONE_SRC_LOCATIONS_LOG,
        true,
        hash_to_zone_src_locations,
        zone_src_locations);

    return hash_to_zone_src_locations;
}

void setControlBuffer(IDevice* device, std::vector<uint32_t>& control_buffer) {
#if defined(TRACY_ENABLE)
    const chip_id_t device_id = device->id();
    const metal_SocDescriptor& soc_d = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id);

    control_buffer[kernel_profiler::CORE_COUNT_PER_DRAM] = soc_d.profiler_ceiled_core_count_perf_dram_bank;
    for (auto core :
         tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_routing_to_profiler_flat_id(device_id)) {
        const CoreCoord curr_core = core.first;

        control_buffer[kernel_profiler::FLAT_ID] = core.second;

        writeToCoreControlBuffer(device, curr_core, ProfilerDumpState::NORMAL, control_buffer);
    }
#endif
}

void syncDeviceHost(IDevice* device, CoreCoord logical_core, bool doHeader) {
    ZoneScopedC(tracy::Color::Tomato3);
    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_sync_enabled()) {
        return;
    }
    auto device_id = device->id();
    auto core = device->worker_core_from_logical_core(logical_core);

    const metal_SocDescriptor& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id);
    auto phys_core = soc_desc.translate_coord_to(core, CoordSystem::TRANSLATED, CoordSystem::PHYSICAL);

    deviceHostTimePair.emplace(device_id, (std::vector<std::pair<uint64_t, uint64_t>>){});
    smallestHostime.emplace(device_id, 0);

    constexpr uint16_t sampleCount = 249;
    // TODO(MO): Always recreate a new program until subdevice
    // allows using the first program generated by default manager
    tt_metal::Program sync_program;

    std::map<std::string, std::string> kernel_defines = {
        {"SAMPLE_COUNT", std::to_string(sampleCount)},
    };

    tt_metal::CreateKernel(
        sync_program,
        "tt_metal/tools/profiler/sync/sync_kernel.cpp",
        logical_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = kernel_defines});

    // Using MeshDevice APIs if the current device is managed by MeshDevice
    tt_metal::detail::LaunchProgram(
        device, sync_program, false /* wait_until_cores_done */, /* force_slow_dispatch */ true);

    std::filesystem::path output_dir = std::filesystem::path(get_profiler_logs_dir());
    std::filesystem::path log_path = output_dir / "sync_device_info.csv";
    std::ofstream log_file;

    constexpr int millisecond_wait = 10;

    const double tracyToSecRatio = TracyGetTimerMul();
    const int64_t tracyBaseTime = TracyGetBaseTime();
    const int64_t hostStartTime = TracyGetCpuTime();
    std::vector<int64_t> writeTimes(sampleCount);

    auto* profiler_msg = reinterpret_cast<profiler_msg_t*>(device->get_dev_addr(core, HalL1MemAddrType::PROFILER));
    uint64_t control_addr = reinterpret_cast<uint64_t>(&profiler_msg->control_vector[kernel_profiler::FW_RESET_L]);
    for (int i = 0; i < sampleCount; i++) {
        ZoneScopedC(tracy::Color::Tomato2);
        std::this_thread::sleep_for(std::chrono::milliseconds(millisecond_wait));
        int64_t writeStart = TracyGetCpuTime();
        uint32_t sinceStart = writeStart - hostStartTime;

        tt::tt_metal::MetalContext::instance().get_cluster().write_reg(
            &sinceStart, tt_cxy_pair(device_id, core), control_addr);
        writeTimes[i] = (TracyGetCpuTime() - writeStart);
    }
    tt_metal::detail::WaitProgramDone(device, sync_program, false);
    std::vector<CoreCoord> cores = {core};
    tt_metal_device_profiler_map.at(device_id).readResults(
        device, cores, ProfilerDumpState::FORCE_UMD_READ, ProfilerDataBufferSource::L1);

    log_info(tt::LogMetal, "SYNC PROGRAM FINISH IS DONE ON {}", device_id);
    if ((smallestHostime[device_id] == 0) || (smallestHostime[device_id] > hostStartTime)) {
        smallestHostime[device_id] = hostStartTime;
    }

    constexpr uint32_t briscIndex = 0;
    uint64_t addr = reinterpret_cast<uint64_t>(&profiler_msg->buffer[briscIndex][kernel_profiler::CUSTOM_MARKERS]);

    std::vector<std::uint32_t> sync_times =
        tt::llrt::read_hex_vec_from_core(device_id, core, addr, (sampleCount + 1) * 2 * sizeof(uint32_t));

    uint32_t preDeviceTime = 0;
    uint32_t preHostTime = 0;
    bool firstSample = true;

    uint32_t deviceStartTime_H = sync_times[0] & 0xFFF;
    uint32_t deviceStartTime_L = sync_times[1];
    preDeviceTime = deviceStartTime_L;

    uint32_t hostStartTime_H = 0;

    for (int i = 2; i < 2 * (sampleCount + 1); i += 2) {
        uint32_t deviceTime = sync_times[i];
        if (deviceTime < preDeviceTime) {
            deviceStartTime_H++;
        }
        preDeviceTime = deviceTime;
        uint64_t deviceTimeLarge = (uint64_t(deviceStartTime_H) << 32) | deviceTime;

        uint32_t hostTime = sync_times[i + 1] + writeTimes[i / 2 - 1];
        if (hostTime < preHostTime) {
            hostStartTime_H++;
        }
        preHostTime = hostTime;
        uint64_t hostTimeLarge =
            hostStartTime - smallestHostime[device_id] + ((uint64_t(hostStartTime_H) << 32) | hostTime);

        deviceHostTimePair[device_id].push_back(std::pair<uint64_t, uint64_t>{deviceTimeLarge, hostTimeLarge});

        if (firstSample) {
            firstSample = false;
        }
    }

    double hostSum = 0;
    double deviceSum = 0;
    double hostSquaredSum = 0;
    double hostDeviceProductSum = 0;

    for (auto& deviceHostTime : deviceHostTimePair[device_id]) {
        double deviceTime = deviceHostTime.first;
        double hostTime = deviceHostTime.second;

        deviceSum += deviceTime;
        hostSum += hostTime;
        hostSquaredSum += (hostTime * hostTime);
        hostDeviceProductSum += (hostTime * deviceTime);
    }

    uint16_t accumulateSampleCount = deviceHostTimePair[device_id].size();

    double frequencyFit = (hostDeviceProductSum * accumulateSampleCount - hostSum * deviceSum) /
                          ((hostSquaredSum * accumulateSampleCount - hostSum * hostSum) * tracyToSecRatio);

    double delay = (deviceSum - frequencyFit * hostSum * tracyToSecRatio) / accumulateSampleCount;

    if (doHeader) {
        log_file.open(log_path);
        log_file << fmt::format(
                        "device id,core_x, "
                        "core_y,device,host_tracy,host_real,write_overhead,host_start,delay,frequency,tracy_ratio,"
                        "tracy_base_time,device_frequency_ratio,device_shift")
                 << std::endl;
    } else {
        log_file.open(log_path, std::ios_base::app);
    }

    int init = deviceHostTimePair[device_id].size() - sampleCount;
    for (int i = init; i < deviceHostTimePair[device_id].size(); i++) {
        log_file << fmt::format(
                        "{:5},{:5},{:5},{:20},{:20},{:20.2f},{:20},{:20},{:20.2f},{:20.15f},{:20.15f},{:20},1.0,0",
                        device_id,
                        phys_core.x,
                        phys_core.y,
                        deviceHostTimePair[device_id][i].first,
                        deviceHostTimePair[device_id][i].second,
                        (double)deviceHostTimePair[device_id][i].second * tracyToSecRatio,
                        writeTimes[i - init],
                        smallestHostime[device_id],
                        delay,
                        frequencyFit,
                        tracyToSecRatio,
                        tracyBaseTime)
                 << std::endl;
    }
    log_file.close();
    log_info(
        tt::LogMetal,
        "Host sync data for device: {}, cpu_start:{}, delay:{}, freq:{} Hz",
        device_id,
        smallestHostime[device_id],
        delay,
        frequencyFit);

    double host_timestamp = hostStartTime;
    double device_timestamp = delay + (host_timestamp - smallestHostime[device_id]) * frequencyFit * tracyToSecRatio;
    // disable linting here; slicing is __intended__
    // NOLINTBEGIN
    tt_metal_device_profiler_map.at(device_id).device_core_sync_info.emplace(
        CoreCoord(phys_core), SyncInfo(host_timestamp, device_timestamp, frequencyFit));
    // NOLINTEND
}

void setShift(int device_id, int64_t shift, double scale, const SyncInfo& root_sync_info) {
    if (std::isnan(scale)) {
        return;
    }
    log_info(tt::LogMetal, "Device sync data for device: {}, delay: {} ns, freq scale: {}", device_id, shift, scale);
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_tracy_mid_run_push()) {
        log_warning(
            tt::LogMetal,
            "Note that tracy mid-run push is enabled. This means device-device sync is not as accurate. "
            "Please do not use tracy mid-run push for sensitive device-device event analysis.");
    }

    auto device_profiler_it = tt_metal_device_profiler_map.find(device_id);
    if (device_profiler_it != tt_metal_device_profiler_map.end()) {
        device_profiler_it->second.freq_scale = scale;
        device_profiler_it->second.shift = shift;
        device_profiler_it->second.setSyncInfo(root_sync_info);

        std::filesystem::path output_dir = std::filesystem::path(get_profiler_logs_dir());
        std::filesystem::path log_path = output_dir / "sync_device_info.csv";
        std::ofstream log_file;
        log_file.open(log_path, std::ios_base::app);
        log_file << fmt::format("{:5},,,,,,,,,,,,{:20.15f},{:20}", device_id, scale, shift) << std::endl;
        log_file.close();
    }
}

void peekDeviceData(IDevice* device, std::vector<CoreCoord>& worker_cores) {
    ZoneScoped;
    auto device_id = device->id();
    std::string zoneName = fmt::format("peek {}", device_id);
    ZoneName(zoneName.c_str(), zoneName.size());
    const auto& device_profiler_it = tt_metal_device_profiler_map.find(device_id);
    if (device_profiler_it != tt_metal_device_profiler_map.end()) {
        DeviceProfiler& device_profiler = device_profiler_it->second;
        device_profiler.device_sync_new_events.clear();
        device_profiler.readResults(
            device, worker_cores, ProfilerDumpState::FORCE_UMD_READ, ProfilerDataBufferSource::L1);
        for (auto& event : device_profiler.device_events) {
            const ZoneDetails zone_details = device_profiler.getZoneDetails(event.timer_id);
            if (zone_details.zone_name_keyword_flags[static_cast<uint16_t>(ZoneDetails::ZoneNameKeyword::SYNC_ZONE)]) {
                ZoneScopedN("Adding_device_sync_event");
                auto ret = device_profiler.device_sync_events.insert(event);
                if (ret.second) {
                    device_profiler.device_sync_new_events.insert(event);
                }
            }
        }
    }
}

void syncDeviceDevice(chip_id_t device_id_sender, chip_id_t device_id_receiver) {
    ZoneScopedC(tracy::Color::Tomato4);
    std::string zoneName = fmt::format("sync_device_device_{}->{}", device_id_sender, device_id_receiver);
    ZoneName(zoneName.c_str(), zoneName.size());
    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_sync_enabled()) {
        return;
    }

    IDevice* device_sender = nullptr;
    IDevice* device_receiver = nullptr;

    if (tt::DevicePool::instance().is_device_active(device_id_receiver)) {
        device_receiver = tt::DevicePool::instance().get_active_device(device_id_receiver);
    }

    if (tt::DevicePool::instance().is_device_active(device_id_sender)) {
        device_sender = tt::DevicePool::instance().get_active_device(device_id_sender);
    }

    if (device_sender != nullptr and device_receiver != nullptr) {
        constexpr std::uint16_t sample_count = 240;
        constexpr std::uint16_t sample_size = 16;
        constexpr std::uint16_t channel_count = 1;

        const auto& active_eth_cores = device_sender->get_active_ethernet_cores(false);
        auto eth_sender_core_iter = active_eth_cores.begin();
        tt_xy_pair eth_receiver_core;
        tt_xy_pair eth_sender_core;

        chip_id_t device_id_receiver_curr = std::numeric_limits<chip_id_t>::max();
        while ((device_id_receiver != device_id_receiver_curr) and (eth_sender_core_iter != active_eth_cores.end())) {
            eth_sender_core = *eth_sender_core_iter;
            if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(
                    device_sender->id(), eth_sender_core)) {
                eth_sender_core_iter++;
                continue;
            }
            std::tie(device_id_receiver_curr, eth_receiver_core) =
                device_sender->get_connected_ethernet_core(eth_sender_core);
            eth_sender_core_iter++;
        }

        if (device_id_receiver != device_id_receiver_curr) {
            log_warning(
                tt::LogMetal,
                "No eth connection could be found between device {} and {}",
                device_id_sender,
                device_id_receiver);
            return;
        }

        const std::vector<uint32_t>& ct_args = {
            channel_count, static_cast<uint32_t>(sample_count), static_cast<uint32_t>(sample_size)};

        Program program_sender;
        Program program_receiver;

        tt_metal::CreateKernel(
            program_sender,
            "tt_metal/tools/profiler/sync/sync_device_kernel_sender.cpp",
            eth_sender_core,
            tt_metal::EthernetConfig{.noc = tt_metal::NOC::RISCV_0_default, .compile_args = ct_args});

        tt_metal::CreateKernel(
            program_receiver,
            "tt_metal/tools/profiler/sync/sync_device_kernel_receiver.cpp",
            eth_receiver_core,
            tt_metal::EthernetConfig{.noc = tt_metal::NOC::RISCV_0_default, .compile_args = ct_args});

        try {
            tt::tt_metal::detail::CompileProgram(device_sender, program_sender);
            tt::tt_metal::detail::CompileProgram(device_receiver, program_receiver);
        } catch (std::exception& e) {
            log_error(tt::LogMetal, "Failed compile: {}", e.what());
            throw e;
        }
        tt_metal::detail::LaunchProgram(
            device_sender, program_sender, false /* wait_until_cores_done */, true /* force_slow_dispatch */);
        tt_metal::detail::LaunchProgram(
            device_receiver, program_receiver, false /* wait_until_cores_done */, true /* force_slow_dispatch */);

        tt_metal::detail::WaitProgramDone(device_sender, program_sender, false);
        tt_metal::detail::WaitProgramDone(device_receiver, program_receiver, false);

        CoreCoord sender_core = {eth_sender_core.x, eth_sender_core.y};
        std::vector<CoreCoord> sender_cores = {
            device_sender->virtual_core_from_logical_core(sender_core, CoreType::ETH)};

        CoreCoord receiver_core = {eth_receiver_core.x, eth_receiver_core.y};
        std::vector<CoreCoord> receiver_cores = {
            device_receiver->virtual_core_from_logical_core(receiver_core, CoreType::ETH)};

        peekDeviceData(device_sender, sender_cores);
        peekDeviceData(device_receiver, receiver_cores);

        TT_ASSERT(
            tt_metal_device_profiler_map.at(device_id_sender).device_sync_new_events.size() ==
            tt_metal_device_profiler_map.at(device_id_receiver).device_sync_new_events.size());

        auto event_receiver = tt_metal_device_profiler_map.at(device_id_receiver).device_sync_new_events.begin();

        for (auto event_sender = tt_metal_device_profiler_map.at(device_id_sender).device_sync_new_events.begin();
             event_sender != tt_metal_device_profiler_map.at(device_id_sender).device_sync_new_events.end();
             event_sender++) {
            TT_ASSERT(event_receiver != tt_metal_device_profiler_map.at(device_id_receiver).device_sync_events.end());
            deviceDeviceTimePair.at(device_id_sender)
                .at(device_id_receiver)
                .push_back({event_sender->timestamp, event_receiver->timestamp});
            event_receiver++;
        }
    }
}

void setSyncInfo(
    chip_id_t device_id,
    std::pair<double, int64_t> syncInfo,
    SyncInfo& root_sync_info,
    std::unordered_map<chip_id_t, std::unordered_map<chip_id_t, std::pair<double, int64_t>>>& deviceDeviceSyncInfo,
    const std::string& parentInfo = "") {
    ZoneScoped;
    if (sync_set_devices.find(device_id) == sync_set_devices.end()) {
        sync_set_devices.insert(device_id);
        if (deviceDeviceSyncInfo.find(device_id) != deviceDeviceSyncInfo.end()) {
            std::string parentInfoNew =
                parentInfo + fmt::format("->{}: ({},{})", device_id, syncInfo.second, syncInfo.first);
            for (auto child_device : deviceDeviceSyncInfo.at(device_id)) {
                std::pair<double, int64_t> childSyncInfo = child_device.second;
                childSyncInfo.second *= syncInfo.first;
                childSyncInfo.second += syncInfo.second;
                childSyncInfo.first *= syncInfo.first;
                setSyncInfo(child_device.first, childSyncInfo, root_sync_info, deviceDeviceSyncInfo, parentInfo);
            }
        }
        detail::setShift(device_id, syncInfo.second, syncInfo.first, root_sync_info);
    }
}

void syncAllDevices(chip_id_t host_connected_device) {
    // Check if profiler on host connected device is initilized
    if (tt_metal_device_profiler_map.find(host_connected_device) == tt_metal_device_profiler_map.end()) {
        return;
    }

    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_sync_enabled()) {
        return;
    }
    // Update deviceDeviceTimePair
    for (const auto& sender : deviceDeviceTimePair) {
        for (const auto& receiver : sender.second) {
            syncDeviceDevice(sender.first, receiver.first);
        }
    }

    // Run linear regression to calculate scale and bias between devices
    // deviceDeviceSyncInfo[dev0][dev1] = {scale, bias} of dev0 over dev1
    std::unordered_map<chip_id_t, std::unordered_map<chip_id_t, std::pair<double, int64_t>>> deviceDeviceSyncInfo;
    for (auto& sender : deviceDeviceTimePair) {
        for (auto& receiver : sender.second) {
            std::vector<std::pair<uint64_t, uint64_t>> timePairs;
            for (int i = 0; i < receiver.second.size(); i += 2) {
                uint64_t senderTime = (receiver.second[i].first + receiver.second[i + 1].first) / 2;
                timePairs.push_back({senderTime, receiver.second[i].second});
            }
            double senderSum = 0;
            double receiverSum = 0;
            double receiverSquareSum = 0;
            double senderReceiverProductSum = 0;

            // Direct computation causes large error because sqaure of clock is very big
            // So apply linear regression on shifted values
            uint64_t senderBase = 0;
            uint64_t receiverBase = 0;

            if (timePairs.size() > 0) {
                senderBase = timePairs[0].first;
                receiverBase = timePairs[0].second;
            }
            for (auto& timePair : timePairs) {
                double senderTime = timePair.first - senderBase;
                double recieverTime = timePair.second - receiverBase;

                receiverSum += recieverTime;
                senderSum += senderTime;
                receiverSquareSum += (recieverTime * recieverTime);
                senderReceiverProductSum += (senderTime * recieverTime);
            }

            uint16_t accumulateSampleCount = timePairs.size();

            double freqScale = (senderReceiverProductSum * accumulateSampleCount - senderSum * receiverSum) /
                               (receiverSquareSum * accumulateSampleCount - receiverSum * receiverSum);

            uint64_t shift = (double)(senderSum - freqScale * (double)receiverSum) / accumulateSampleCount +
                             (senderBase - freqScale * receiverBase);
            deviceDeviceSyncInfo.emplace(sender.first, (std::unordered_map<chip_id_t, std::pair<double, int64_t>>){});
            deviceDeviceSyncInfo.at(sender.first)
                .emplace(receiver.first, (std::pair<double, int64_t>){freqScale, shift});

            deviceDeviceSyncInfo.emplace(receiver.first, (std::unordered_map<chip_id_t, std::pair<double, int64_t>>){});
            deviceDeviceSyncInfo.at(receiver.first)
                .emplace(sender.first, (std::pair<double, int64_t>){1.0 / freqScale, -1 * shift});
        }
    }

    // Find any sync info from root device
    // Currently, sync info only exists for SYNC_CORE
    SyncInfo root_sync_info;
    for (auto& [core, info] : tt_metal_device_profiler_map.at(host_connected_device).device_core_sync_info) {
        root_sync_info = info;
        break;
    }

    // Propagate sync info with DFS through sync tree
    sync_set_devices.clear();
    setSyncInfo(host_connected_device, (std::pair<double, int64_t>){1.0, 0}, root_sync_info, deviceDeviceSyncInfo);
}

void ProfilerSync(ProfilerSyncState state) {
#if defined(TRACY_ENABLE)
    ZoneScoped;
    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_sync_enabled()) {
        return;
    }
    if (!getDeviceProfilerState()) {
        return;
    }
    static chip_id_t first_connected_device_id = -1;
    if (state == ProfilerSyncState::INIT) {
        do_sync_on_close = true;
        auto ethernet_connections = tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_connections();
        std::unordered_set<chip_id_t> visited_devices = {};
        constexpr int TOTAL_DEVICE_COUNT = 36;
        for (int sender_device_id = 0; sender_device_id < TOTAL_DEVICE_COUNT; sender_device_id++) {
            if (tt::DevicePool::instance().is_device_active(sender_device_id)) {
                auto sender_device = tt::DevicePool::instance().get_active_device(sender_device_id);
                const auto& active_eth_cores = sender_device->get_active_ethernet_cores(false);

                chip_id_t receiver_device_id;
                tt_xy_pair receiver_eth_core;
                bool doSync = true;
                for (auto& sender_eth_core : active_eth_cores) {
                    if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(
                            sender_device_id, sender_eth_core)) {
                        continue;
                    }
                    doSync = false;
                    std::tie(receiver_device_id, receiver_eth_core) =
                        sender_device->get_connected_ethernet_core(sender_eth_core);

                    if (visited_devices.find(sender_device_id) == visited_devices.end() or
                        visited_devices.find(receiver_device_id) == visited_devices.end()) {
                        visited_devices.insert(sender_device_id);
                        visited_devices.insert(receiver_device_id);

                        deviceDeviceTimePair.emplace(
                            sender_device_id,
                            (std::unordered_map<chip_id_t, std::vector<std::pair<uint64_t, uint64_t>>>){});
                        deviceDeviceTimePair.at(sender_device_id)
                            .emplace(receiver_device_id, (std::vector<std::pair<uint64_t, uint64_t>>){});
                    }
                }
                if (doSync or first_connected_device_id == -1) {
                    if (first_connected_device_id == -1 and !doSync) {
                        first_connected_device_id = sender_device_id;
                    }
                    syncDeviceHost(sender_device, SYNC_CORE, true);
                }
            }
        }
        // If at least one sender reciever pair has been found
        if (first_connected_device_id != -1) {
            syncAllDevices(first_connected_device_id);
        }
    }

    if (state == ProfilerSyncState::CLOSE_DEVICE and do_sync_on_close) {
        do_sync_on_close = false;
        for (const auto& synced_with_host_device : deviceHostTimePair) {
            auto deviceToSync = tt::DevicePool::instance().get_active_device(synced_with_host_device.first);
            syncDeviceHost(deviceToSync, SYNC_CORE, false);
        }
        //  If at least one sender reciever pair has been found
        if (first_connected_device_id != -1) {
            syncAllDevices(first_connected_device_id);
        }
    }
#endif
}

void ClearProfilerControlBuffer(IDevice* device) {
#if defined(TRACY_ENABLE)
    std::vector<uint32_t> control_buffer(kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE, 0);
    setControlBuffer(device, control_buffer);
#endif
}

void InitDeviceProfiler(IDevice* device) {
#if defined(TRACY_ENABLE)
    ZoneScoped;

    TracySetCpuTime(TracyGetCpuTime());

    if (getDeviceProfilerState()) {
        static std::atomic<bool> firstInit = true;

        const chip_id_t device_id = device->id();

        if (tt_metal_device_profiler_map.find(device_id) == tt_metal_device_profiler_map.end()) {
            if (firstInit.exchange(false)) {
                tt_metal_device_profiler_map.emplace(device_id, DeviceProfiler(device, true));
            } else {
                tt_metal_device_profiler_map.emplace(device_id, DeviceProfiler(device, false));
            }
        }

        auto& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id);

        const uint32_t num_cores_per_dram_bank = soc_desc.profiler_ceiled_core_count_perf_dram_bank;
        const uint32_t bank_size_bytes =
            PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC * MAX_RISCV_PER_CORE * num_cores_per_dram_bank;
        TT_ASSERT(bank_size_bytes <= MetalContext::instance().hal().get_dev_size(HalDramMemAddrType::PROFILER));

        const uint32_t num_dram_banks = soc_desc.get_num_dram_views();

        auto& profiler = tt_metal_device_profiler_map.at(device_id);
        profiler.profile_buffer_bank_size_bytes = bank_size_bytes;
        profiler.profile_buffer.resize(profiler.profile_buffer_bank_size_bytes * num_dram_banks / sizeof(uint32_t));

        std::vector<uint32_t> control_buffer(kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE, 0);
        control_buffer[kernel_profiler::DRAM_PROFILER_ADDRESS] =
            MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::PROFILER);
        setControlBuffer(device, control_buffer);
    }
#endif
}

void convertNocTraceDataToJSON(
    nlohmann::ordered_json& noc_trace_json_log, const std::vector<DeviceProfilerDataPoint>& device_profiler_data) {
    if (!MetalContext::instance().rtoptions().get_profiler_noc_events_enabled()) {
        return;
    }

    for (const DeviceProfilerDataPoint& data_point : device_profiler_data) {
        if (data_point.packet_type == kernel_profiler::ZONE_START ||
            data_point.packet_type == kernel_profiler::ZONE_END) {
            if ((data_point.risc_name == "NCRISC" || data_point.risc_name == "BRISC") &&
                (data_point.zone_name.starts_with("TRUE-KERNEL-END") || data_point.zone_name.ends_with("-KERNEL"))) {
                tracy::TTDeviceEventPhase zone_phase = (data_point.packet_type == kernel_profiler::ZONE_END)
                                                           ? tracy::TTDeviceEventPhase::end
                                                           : tracy::TTDeviceEventPhase::begin;
                noc_trace_json_log.push_back(nlohmann::ordered_json{
                    {"run_host_id", data_point.run_host_id},
                    {"op_name", data_point.op_name},
                    {"proc", data_point.risc_name},
                    {"zone", data_point.zone_name},
                    {"zone_phase", magic_enum::enum_name(zone_phase)},
                    {"sx", data_point.core_x},
                    {"sy", data_point.core_y},
                    {"timestamp", data_point.timestamp},
                });
            }
        } else if (data_point.packet_type == kernel_profiler::TS_DATA) {
            using EMD = KernelProfilerNocEventMetadata;
            EMD ev_md(data_point.data);
            std::variant<EMD::LocalNocEvent, EMD::FabricNoCEvent, EMD::FabricRoutingFields> ev_md_contents =
                ev_md.getContents();
            if (std::holds_alternative<EMD::LocalNocEvent>(ev_md_contents)) {
                auto local_noc_event = std::get<EMD::LocalNocEvent>(ev_md_contents);

                // NOTE: assume here that src and dest device_id are local;
                // serialization will coalesce and update to correct destination
                // based on fabric events
                nlohmann::ordered_json data = {
                    {"run_host_id", data_point.run_host_id},
                    {"op_name", data_point.op_name},
                    {"proc", data_point.risc_name},
                    {"noc", magic_enum::enum_name(local_noc_event.noc_type)},
                    {"vc", int(local_noc_event.noc_vc)},
                    {"src_device_id", data_point.device_id},
                    {"sx", data_point.core_x},
                    {"sy", data_point.core_y},
                    {"num_bytes", local_noc_event.getNumBytes()},
                    {"type", magic_enum::enum_name(ev_md.noc_xfer_type)},
                    {"timestamp", data_point.timestamp},
                };

                // handle dst coordinates correctly for different NocEventType
                if (local_noc_event.dst_x == -1 || local_noc_event.dst_y == -1 ||
                    ev_md.noc_xfer_type == EMD::NocEventType::READ_WITH_STATE ||
                    ev_md.noc_xfer_type == EMD::NocEventType::WRITE_WITH_STATE) {
                    // DO NOT emit destination coord; it isn't meaningful

                } else if (ev_md.noc_xfer_type == EMD::NocEventType::WRITE_MULTICAST) {
                    auto phys_start_coord = getPhysicalAddressFromVirtual(
                        data_point.device_id, {local_noc_event.dst_x, local_noc_event.dst_y});
                    data["mcast_start_x"] = phys_start_coord.x;
                    data["mcast_start_y"] = phys_start_coord.y;
                    auto phys_end_coord = getPhysicalAddressFromVirtual(
                        data_point.device_id, {local_noc_event.mcast_end_dst_x, local_noc_event.mcast_end_dst_y});
                    data["mcast_end_x"] = phys_end_coord.x;
                    data["mcast_end_y"] = phys_end_coord.y;
                } else {
                    auto phys_coord = getPhysicalAddressFromVirtual(
                        data_point.device_id, {local_noc_event.dst_x, local_noc_event.dst_y});
                    data["dx"] = phys_coord.x;
                    data["dy"] = phys_coord.y;
                }

                noc_trace_json_log.push_back(std::move(data));
            } else if (std::holds_alternative<EMD::FabricNoCEvent>(ev_md_contents)) {
                EMD::FabricNoCEvent fabric_noc_event = std::get<EMD::FabricNoCEvent>(ev_md_contents);

                nlohmann::ordered_json data = {
                    {"run_host_id", data_point.run_host_id},
                    {"op_name", data_point.op_name},
                    {"proc", data_point.risc_name},
                    {"sx", data_point.core_x},
                    {"sy", data_point.core_y},
                    {"type", magic_enum::enum_name(ev_md.noc_xfer_type)},
                    {"routing_fields_type", magic_enum::enum_name(fabric_noc_event.routing_fields_type)},
                    {"timestamp", data_point.timestamp},
                };

                // For scatter write operations, include additional scatter information
                if (ev_md.noc_xfer_type == EMD::NocEventType::FABRIC_UNICAST_SCATTER_WRITE) {
                    data["scatter_address_index"] = fabric_noc_event.mcast_end_dst_x;
                    data["scatter_total_addresses"] = fabric_noc_event.mcast_end_dst_y;
                }

                // handle dst coordinates correctly for different NocEventType
                if (KernelProfilerNocEventMetadata::isFabricUnicastEventType(ev_md.noc_xfer_type)) {
                    auto phys_coord = getPhysicalAddressFromVirtual(
                        data_point.device_id, {fabric_noc_event.dst_x, fabric_noc_event.dst_y});
                    data["dx"] = phys_coord.x;
                    data["dy"] = phys_coord.y;
                } else {
                    auto phys_start_coord = getPhysicalAddressFromVirtual(
                        data_point.device_id, {fabric_noc_event.dst_x, fabric_noc_event.dst_y});
                    data["mcast_start_x"] = phys_start_coord.x;
                    data["mcast_start_y"] = phys_start_coord.y;
                    auto phys_end_coord = getPhysicalAddressFromVirtual(
                        data_point.device_id, {fabric_noc_event.mcast_end_dst_x, fabric_noc_event.mcast_end_dst_y});
                    data["mcast_end_x"] = phys_end_coord.x;
                    data["mcast_end_y"] = phys_end_coord.y;
                }

                noc_trace_json_log.push_back(std::move(data));
            } else if (std::holds_alternative<EMD::FabricRoutingFields>(ev_md_contents)) {
                uint32_t routing_fields_value = std::get<EMD::FabricRoutingFields>(ev_md_contents).routing_fields_value;
                noc_trace_json_log.push_back(nlohmann::ordered_json{
                    {"run_host_id", data_point.run_host_id},
                    {"op_name", data_point.op_name},
                    {"proc", data_point.risc_name},
                    {"sx", data_point.core_x},
                    {"sy", data_point.core_y},
                    {"routing_fields_value", routing_fields_value},
                    {"timestamp", data_point.timestamp},
                });
            }
        }
    }
}

void serializeJSONNocTraces(
    const nlohmann::ordered_json& noc_trace_json_log,
    const std::filesystem::path& output_dir,
    chip_id_t device_id,
    const FabricRoutingLookup& routing_lookup) {
    // create output directory if it does not exist
    std::filesystem::create_directories(output_dir);
    if (!std::filesystem::is_directory(output_dir)) {
        log_error(
            tt::LogMetal,
            "Could not write noc event json trace to '{}' because the directory path could not be created!",
            output_dir);
        return;
    }

    // bin events by runtime id
    using RuntimeID = uint32_t;
    std::unordered_map<RuntimeID, nlohmann::json::array_t> events_by_opname;
    for (auto& json_event : noc_trace_json_log) {
        RuntimeID runtime_id = json_event.value("run_host_id", -1);
        events_by_opname[runtime_id].push_back(json_event);
    }

    // sort events in each opname group by proc first, then timestamp
    for (auto& [runtime_id, events] : events_by_opname) {
        std::sort(events.begin(), events.end(), [](const auto& a, const auto& b) {
            auto sx_a = a.value("sx", 0);
            auto sy_a = a.value("sy", 0);
            auto sx_b = b.value("sx", 0);
            auto sy_b = b.value("sy", 0);
            auto proc_a = a.value("proc", "");
            auto proc_b = b.value("proc", "");
            auto timestamp_a = a.value("timestamp", 0);
            auto timestamp_b = b.value("timestamp", 0);
            return std::tie(sx_a, sy_a, proc_a, timestamp_a) < std::tie(sx_b, sy_b, proc_b, timestamp_b);
        });
    }

    // for each opname in events_by_opname, adjust timestamps to be relative to the smallest timestamp within the
    // group with identical sx,sy,proc
    for (auto& [runtime_id, events] : events_by_opname) {
        std::tuple<int, int, std::string> reference_event_loc;
        uint64_t reference_timestamp = 0;
        for (auto& event : events) {
            std::string zone = event.value("zone", "");
            std::string zone_phase = event.value("zone_phase", "");
            uint64_t curr_timestamp = event.value("timestamp", 0);
            // if -KERNEL::begin event is found, reset the reference timestamp
            if (zone.ends_with("-KERNEL") && zone_phase == "begin") {
                reference_timestamp = curr_timestamp;
            }

            // fix timestamp to be relative to reference_timestamp
            event["timestamp"] = curr_timestamp - reference_timestamp;
        }
    }

    auto process_fabric_event_group_if_valid =
        [&](const nlohmann::ordered_json& fabric_event,
            const nlohmann::ordered_json& fabric_routing_fields_event,
            const nlohmann::ordered_json& local_noc_write_event) -> std::optional<nlohmann::ordered_json> {
        bool local_event_is_valid_type =
            local_noc_write_event.contains("type") && local_noc_write_event["type"] == "WRITE_";
        if (!local_event_is_valid_type) {
            log_error(
                tt::LogMetal,
                "[profiler noc tracing] local noc event following fabric event is not a regular noc write, but instead "
                ": {}",
                local_noc_write_event["type"].get<std::string>());
            return std::nullopt;
        }

        // Check if timestamps are close enough; otherwise
        double ts_diff = local_noc_write_event.value("timestamp", 0.0) - fabric_event.value("timestamp", 0.0);
        if (ts_diff > 1000) {
            log_warning(
                tt::LogMetal,
                "[profiler noc tracing] Failed to coalesce fabric noc trace events because timestamps are implausibly "
                "far apart.");
            return std::nullopt;
        }

        try {
            // router eth core location is derived from the following noc WRITE_ event
            CoreCoord virt_eth_route_coord = {
                local_noc_write_event.at("dx").get<int>(), local_noc_write_event.at("dy").get<int>()};
            CoreCoord phys_eth_route_coord = getPhysicalAddressFromVirtual(device_id, virt_eth_route_coord);

            auto routing_fields_type_str = fabric_event.at("routing_fields_type").get<std::string>();
            auto maybe_routing_fields_type =
                magic_enum::enum_cast<KernelProfilerNocEventMetadata::FabricPacketType>(routing_fields_type_str);
            if (!maybe_routing_fields_type) {
                log_error(
                    tt::LogMetal,
                    "[profiler noc tracing] Failed to parse routing fields type: {}",
                    routing_fields_type_str);
                return std::nullopt;
            }
            auto routing_fields_type = maybe_routing_fields_type.value();

            // determine hop count and other routing metadata from routing fields value
            uint32_t routing_fields_value = fabric_routing_fields_event.at("routing_fields_value").get<uint32_t>();
            int start_distance = 0;
            int range = 0;
            switch (routing_fields_type) {
                case KernelProfilerNocEventMetadata::FabricPacketType::REGULAR: {
                    std::tie(start_distance, range) = get_routing_start_distance_and_range(routing_fields_value);
                    break;
                }
                case KernelProfilerNocEventMetadata::FabricPacketType::LOW_LATENCY: {
                    std::tie(start_distance, range) =
                        get_low_latency_routing_start_distance_and_range(routing_fields_value);
                    break;
                }
                case KernelProfilerNocEventMetadata::FabricPacketType::LOW_LATENCY_MESH: {
                    log_error(
                        tt::LogMetal, "[profiler noc tracing] noc tracing does not support LOW_LATENCY_MESH packets!");
                    return std::nullopt;
                }
            }

            auto eth_chan_opt = routing_lookup.getRouterEthCoreToChannelLookup(device_id, phys_eth_route_coord);
            if (!eth_chan_opt) {
                log_warning(
                    tt::LogMetal,
                    "[profiler noc tracing] Fabric edm_location->channel lookup failed for event in op '{}' at ts {}: "
                    "src_dev={}, "
                    "eth_core=({}, {}), start_distance={}. Keeping original events.",
                    fabric_event.value("op_name", "N/A"),
                    fabric_event.value("timestamp", 0.0),
                    device_id,
                    phys_eth_route_coord.x,
                    phys_eth_route_coord.y,
                    start_distance);
                return std::nullopt;
            }

            tt::tt_fabric::chan_id_t eth_chan = *eth_chan_opt;

            nlohmann::ordered_json modified_write_event = local_noc_write_event;
            modified_write_event["timestamp"] = fabric_event["timestamp"];

            // replace original eth core destination with true destination
            auto noc_xfer_type = magic_enum::enum_cast<KernelProfilerNocEventMetadata::NocEventType>(
                fabric_event["type"].get<std::string>());

            if (!noc_xfer_type.has_value() ||
                !KernelProfilerNocEventMetadata::isFabricEventType(noc_xfer_type.value())) {
                log_error(
                    tt::LogMetal,
                    "[profiler noc tracing] Failed to parse noc transfer type: {}",
                    fabric_event["type"].get<std::string>());
                return std::nullopt;
            }

            if (KernelProfilerNocEventMetadata::isFabricUnicastEventType(noc_xfer_type.value())) {
                modified_write_event["dx"] = fabric_event.at("dx").get<int>();
                modified_write_event["dy"] = fabric_event.at("dy").get<int>();
            } else {
                log_error(tt::LogMetal, "[profiler noc tracing] Noc multicasts in fabric events are not supported!");
                return std::nullopt;
            }

            // replace the type with fabric event type
            modified_write_event["type"] = fabric_event["type"];

            modified_write_event["fabric_send"] = {
                {"eth_chan", eth_chan}, {"start_distance", start_distance}, {"range", range}};

            return modified_write_event;
        } catch (const nlohmann::json::exception& e) {
            log_warning(
                tt::LogMetal,
                "[profiler noc tracing] JSON parsing error during event coalescing for event in op '{}': {}",
                fabric_event.value("op_name", "N/A"),
                e.what());
            return std::nullopt;
        }
    };

    // coalesce fabric events into single logical trace events with extra 'fabric_send' metadata
    std::unordered_map<RuntimeID, nlohmann::json::array_t> processed_events_by_opname;
    for (auto& [runtime_id, events] : events_by_opname) {
        nlohmann::json::array_t coalesced_events;
        for (size_t i = 0; i < events.size(); /* manual increment */) {
            const auto& current_event = events[i];

            bool fabric_event_group_detected =
                (current_event.contains("type") && current_event["type"].get<std::string>().starts_with("FABRIC_") &&
                 (i + 2 < events.size()));
            if (fabric_event_group_detected) {
                if (auto maybe_fabric_event =
                        process_fabric_event_group_if_valid(events[i], events[i + 1], events[i + 2]);
                    maybe_fabric_event) {
                    coalesced_events.push_back(maybe_fabric_event.value());
                }
                // Unconditionally advance past all coalesced events (fabric_event, fabric_routing_fields,
                // local_noc_write_event), even if a valid event cannot be generated
                i += 3;
            } else {
                // If not a fabric event group, simply copy existing event as-is
                coalesced_events.push_back(current_event);
                i += 1;
            }
        }
        // Store the final coalesced/processed list for this op_name
        processed_events_by_opname[runtime_id] = std::move(coalesced_events);
    }

    log_info(tt::LogMetal, "Writing profiler noc traces to '{}'", output_dir);
    for (auto& [runtime_id, events] : processed_events_by_opname) {
        // dump events to a json file inside directory output_dir named after the opname
        std::filesystem::path rpt_path = output_dir;
        std::string op_name = events.front().value("op_name", "UnknownOP");
        if (!op_name.empty()) {
            rpt_path /= fmt::format("noc_trace_dev{}_{}_ID{}.json", device_id, op_name, runtime_id);
        } else {
            rpt_path /= fmt::format("noc_trace_dev{}_ID{}.json", device_id, runtime_id);
        }
        std::ofstream file(rpt_path);
        if (file.is_open()) {
            // Write the final processed events for this op
            file << nlohmann::json(std::move(events)).dump(2);
        } else {
            log_error(tt::LogMetal, "Could not write noc event json trace to '{}'", rpt_path);
        }
    }
}

void logDeviceProfilerDataToCSVFile(
    std::ofstream& log_file_ofs, const std::vector<DeviceProfilerDataPoint>& device_profiler_data) {
    for (const DeviceProfilerDataPoint& data_point : device_profiler_data) {
        std::string meta_data_str = "";
        if (!data_point.meta_data.is_null()) {
            meta_data_str = data_point.meta_data.dump();
            std::replace(meta_data_str.begin(), meta_data_str.end(), ',', ';');
        }

        log_file_ofs << fmt::format(
            "{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            data_point.device_id,
            data_point.core_x,
            data_point.core_y,
            data_point.risc_name,
            data_point.timer_id,
            data_point.timestamp,
            data_point.data,
            data_point.run_host_id,
            data_point.zone_name,
            magic_enum::enum_name(data_point.packet_type),
            data_point.source_line,
            data_point.source_file,
            meta_data_str);
    }
}

void writeCSVHeader(std::ofstream& log_file_ofs, tt::ARCH device_architecture, int device_core_frequency) {
    log_file_ofs << "ARCH: " << get_string_lowercase(device_architecture)
                 << ", CHIP_FREQ[MHz]: " << device_core_frequency << std::endl;
    log_file_ofs << "PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], data, "
                    "run host ID,  zone name, type, source line, source file, meta data"
                 << std::endl;
}

std::ofstream openCSVFile(const std::filesystem::path& csv_path, const IDevice* device) {
    std::ofstream log_file_ofs;

    // append to existing CSV log file if it already exists
    if (std::filesystem::exists(csv_path)) {
        log_file_ofs.open(csv_path, std::ios_base::app);
    } else {
        log_file_ofs.open(csv_path);
        writeCSVHeader(
            log_file_ofs,
            device->arch(),
            tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(device->build_id()));
    }

    return log_file_ofs;
}

bool areAllCoresDispatchCores(IDevice* device, const std::vector<CoreCoord>& virtual_cores) {
    const chip_id_t device_id = device->id();
    const uint8_t device_num_hw_cqs = device->num_hw_cqs();
    const auto& dispatch_core_config = get_dispatch_core_config();
    std::vector<CoreCoord> dispatch_cores;
    for (const CoreCoord& core : tt::get_logical_dispatch_cores(device_id, device_num_hw_cqs, dispatch_core_config)) {
        const CoreCoord virtual_dispatch_core =
            device->virtual_core_from_logical_core(core, dispatch_core_config.get_core_type());
        dispatch_cores.push_back(virtual_dispatch_core);
    }

    for (const CoreCoord& core : virtual_cores) {
        if (std::find(dispatch_cores.begin(), dispatch_cores.end(), core) == dispatch_cores.end()) {
            return false;
        }
    }
    return true;
}

std::vector<CoreCoord> getVirtualCoresToRead(const IDevice* device, ProfilerDumpState state) {
    std::vector<CoreCoord> virtual_cores;

    const chip_id_t device_id = device->id();
    const uint8_t device_num_hw_cqs = device->num_hw_cqs();
    const auto& dispatch_core_config = get_dispatch_core_config();

    if (!onlyProfileDispatchCores(state)) {
        for (const CoreCoord& core :
             tt::get_logical_compute_cores(device_id, device_num_hw_cqs, dispatch_core_config)) {
            const CoreCoord curr_core = device->worker_core_from_logical_core(core);
            virtual_cores.push_back(curr_core);
        }
        for (const CoreCoord& core : device->get_active_ethernet_cores(true)) {
            const CoreCoord curr_core = device->virtual_core_from_logical_core(core, CoreType::ETH);
            virtual_cores.push_back(curr_core);
        }
    }

    if (tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_do_dispatch_cores()) {
        for (const CoreCoord& core :
             tt::get_logical_dispatch_cores(device_id, device_num_hw_cqs, dispatch_core_config)) {
            const CoreCoord curr_core =
                device->virtual_core_from_logical_core(core, dispatch_core_config.get_core_type());
            virtual_cores.push_back(curr_core);
        }
    }

    return virtual_cores;
}

void ReadDeviceProfilerResults(
    IDevice* device,
    ProfilerDumpState state,
    const std::optional<ProfilerOptionalMetadata>& metadata,
    std::vector<DeviceProfilerDataPoint>& device_profiler_data) {
#if defined(TRACY_ENABLE)
    ZoneScoped;

    if (getDeviceProfilerState()) {
        const std::vector<CoreCoord> virtual_cores = getVirtualCoresToRead(device, state);
        if (state != ProfilerDumpState::ONLY_DISPATCH_CORES) {
            if (tt::DevicePool::instance().is_dispatch_firmware_active() && !isGalaxyMMIODevice(device)) {
                for (uint8_t cq_id = 0; cq_id < device->num_hw_cqs(); ++cq_id) {
                    if (auto mesh_device = device->get_mesh_device()) {
                        mesh_device->mesh_command_queue(cq_id).finish();
                    } else {
                        Finish(device->command_queue(cq_id));
                    }
                }
            }
        } else if (onlyProfileDispatchCores(state)) {
            TT_ASSERT(areAllCoresDispatchCores(device, virtual_cores));

            constexpr uint8_t maxLoopCount = 10;
            constexpr uint32_t loopDuration_us = 10000;

            const auto& hal = MetalContext::instance().hal();
            for (const CoreCoord& core : virtual_cores) {
                bool is_core_done = false;

                const HalProgrammableCoreType core_type = tt::llrt::get_core_type(device->id(), core);

                profiler_msg_t* profiler_msg = hal.get_dev_addr<profiler_msg_t*>(core_type, HalL1MemAddrType::PROFILER);
                for (int i = 0; i < maxLoopCount; i++) {
                    const std::vector<std::uint32_t> control_buffer = tt::llrt::read_hex_vec_from_core(
                        device->id(),
                        core,
                        reinterpret_cast<uint64_t>(profiler_msg->control_vector),
                        kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE);
                    if (control_buffer[kernel_profiler::PROFILER_DONE] == 1) {
                        is_core_done = true;
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(loopDuration_us));
                }
                if (!is_core_done) {
                    std::string msg = fmt::format(
                        "Device profiling never finished on device {}, worker core {}, {}",
                        device->id(),
                        core.x,
                        core.y);
                    TracyMessageC(msg.c_str(), msg.size(), tracy::Color::Tomato3);
                    log_warning(tt::LogMetal, "{}", msg);
                }
            }
        } else {
            return;
        }

        TT_FATAL(
            not tt::tt_metal::MetalContext::instance().dprint_server(),
            "Debug print server is running, cannot read device profiler data");

        auto profiler_it = tt_metal_device_profiler_map.find(device->id());
        if (profiler_it != tt_metal_device_profiler_map.end()) {
            DeviceProfiler& profiler = profiler_it->second;
            // profiler.setDeviceArchitecture(device->arch());
            device_profiler_data =
                profiler.readResults(device, virtual_cores, state, ProfilerDataBufferSource::DRAM, metadata);
            if (tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_tracy_mid_run_push()) {
                // wrap this with mutex
                profiler.pushTracyDeviceResults();
            }
        }
    }
#endif
}

void DumpDeviceProfileResults(
    IDevice* device, ProfilerDumpState state, const std::optional<ProfilerOptionalMetadata>& metadata) {
#if defined(TRACY_ENABLE)
    ZoneScoped;

    const chip_id_t device_id = device->id();

    std::filesystem::path output_dir = tt_metal_device_profiler_map.at(device_id).getOutputDir();
    std::filesystem::path log_path = output_dir / DEVICE_SIDE_LOG;
    std::ofstream log_file_ofs = openCSVFile(log_path, device);

    if (!log_file_ofs) {
        log_error(tt::LogMetal, "Could not open kernel profiler dump file '{}'", log_path);
        return;
    }

    if (tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_noc_events_enabled()) {
        log_warning(
            tt::LogAlways, "Profiler NoC events are enabled; this can add 1-15% cycle overhead to typical operations!");
    }

    DeviceProfiler& device_profiler = tt_metal_device_profiler_map.at(device_id);
    device_profiler.hash_to_zone_src_locations = generateZoneSourceLocationsHashes();

    std::vector<DeviceProfilerDataPoint> device_profiler_data;
    ReadDeviceProfilerResults(device, state, metadata, device_profiler_data);

    // serialize noc traces only in normal state, to avoid overwriting individual trace files
    if (state == ProfilerDumpState::NORMAL &&
        tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_noc_events_enabled()) {
        // if defined, use profiler_noc_events_report_path to write json log. otherwise use output_dir
        std::string rpt_path = tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_noc_events_report_path();
        if (rpt_path.empty()) {
            rpt_path = device_profiler.getOutputDir().string();
        }

        FabricRoutingLookup routing_lookup(device);
        nlohmann::ordered_json noc_trace_json_log = nlohmann::json::array();
        convertNocTraceDataToJSON(noc_trace_json_log, device_profiler_data);
        serializeJSONNocTraces(noc_trace_json_log, rpt_path, device_id, routing_lookup);

        dumpClusterCoordinatesAsJson(std::filesystem::path(rpt_path) / "cluster_coordinates.json");
        dumpRoutingInfo(std::filesystem::path(rpt_path) / "topology.json");
    }

    logDeviceProfilerDataToCSVFile(log_file_ofs, device_profiler_data);
    log_file_ofs.close();
#endif
}

void SetDeviceProfilerDir(const std::string& output_dir) {
#if defined(TRACY_ENABLE)
    for (auto& device_id : tt_metal_device_profiler_map) {
        tt_metal_device_profiler_map.at(device_id.first).setOutputDir(output_dir);
    }
#endif
}

void FreshProfilerDeviceLog() {
#if defined(TRACY_ENABLE)
    for (auto& device_id : tt_metal_device_profiler_map) {
        tt_metal_device_profiler_map.at(device_id.first).freshDeviceLog();
    }
#endif
}

uint32_t EncodePerDeviceProgramID(uint32_t base_program_id, uint32_t device_id, bool is_host_fallback_op) {
    // Given the base (host assigned id) for a program running on multiple devices, generate a unique per-device
    // id by coalescing the physical_device id with the program id.
    // For ops running on device, the MSB is 0. For host-fallback ops, the MSB is 1. This avoids aliasing.
    constexpr uint32_t DEVICE_ID_NUM_BITS = 10;
    constexpr uint32_t DEVICE_OP_ID_NUM_BITS = 31;
    return (is_host_fallback_op << DEVICE_OP_ID_NUM_BITS) | (base_program_id << DEVICE_ID_NUM_BITS) | device_id;
}

}  // namespace detail

void DumpMeshDeviceProfileResults(
    distributed::MeshDevice& mesh_device,
    ProfilerDumpState state,
    const std::optional<ProfilerOptionalMetadata>& metadata) {
#if defined(TRACY_ENABLE)
    ZoneScoped;

    std::filesystem::path output_dir = detail::tt_metal_device_profiler_map.at(mesh_device.build_id()).getOutputDir();
    std::filesystem::path log_path = output_dir / DEVICE_SIDE_LOG;
    std::ofstream log_file_ofs = detail::openCSVFile(log_path, &mesh_device);

    if (!log_file_ofs) {
        log_error(tt::LogMetal, "Could not open kernel profiler dump file '{}'", log_path);
        return;
    }

    if (tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_noc_events_enabled()) {
        log_warning(
            tt::LogAlways, "Profiler NoC events are enabled; this can add 1-15% cycle overhead to typical operations!");
    }

    const std::unordered_map<uint16_t, ZoneDetails> hash_to_zone_src_locations =
        detail::generateZoneSourceLocationsHashes();
    for (auto& [_, device_profiler] : detail::tt_metal_device_profiler_map) {
        device_profiler.hash_to_zone_src_locations = hash_to_zone_src_locations;
    }

    std::unordered_map<chip_id_t, std::vector<DeviceProfilerDataPoint>> device_profiler_data;

    for (IDevice* device : mesh_device.get_devices()) {
        const chip_id_t device_id = device->build_id();
        device_profiler_data[device_id] = std::vector<DeviceProfilerDataPoint>();

        std::vector<DeviceProfilerDataPoint>& profiler_data = device_profiler_data.at(device_id);
        mesh_device.enqueue_to_thread_pool([device, state, metadata, &profiler_data]() {
            detail::ReadDeviceProfilerResults(device, state, metadata, profiler_data);
        });
    }

    mesh_device.wait_for_thread_pool();

    // serialize noc traces only in normal state, to avoid overwriting individual trace files
    if (state == ProfilerDumpState::NORMAL &&
        tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_noc_events_enabled()) {
        for (auto& [device_id, device_profiler] : detail::tt_metal_device_profiler_map) {
            // if defined, use profiler_noc_events_report_path to write json log. otherwise use output_dir
            std::string rpt_path =
                tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_noc_events_report_path();
            if (rpt_path.empty()) {
                rpt_path = device_profiler.getOutputDir().string();
            }

            FabricRoutingLookup routing_lookup(tt::DevicePool::instance().get_active_device(device_id));
            nlohmann::ordered_json noc_trace_json_log = nlohmann::json::array();
            detail::convertNocTraceDataToJSON(noc_trace_json_log, device_profiler_data.at(device_id));
            detail::serializeJSONNocTraces(noc_trace_json_log, rpt_path, device_id, routing_lookup);

            dumpClusterCoordinatesAsJson(std::filesystem::path(rpt_path) / "cluster_coordinates.json");
            dumpRoutingInfo(std::filesystem::path(rpt_path) / "topology.json");
        }
    }

    for (IDevice* device : mesh_device.get_devices()) {
        const chip_id_t device_id = device->build_id();
        detail::logDeviceProfilerDataToCSVFile(log_file_ofs, device_profiler_data.at(device_id));
    }

    log_file_ofs.close();
#endif
}

}  // namespace tt_metal

}  // namespace tt
