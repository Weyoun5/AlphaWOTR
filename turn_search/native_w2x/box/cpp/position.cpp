#include "wotr_native.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
static_assert(sizeof(WotrUnit) == 3 * sizeof(int32_t), "WotrUnit ABI layout");
static_assert(sizeof(WotrLeader) == 4 * sizeof(int32_t), "WotrLeader ABI layout");
static_assert(sizeof(WotrRegion) == 5 * sizeof(int32_t), "WotrRegion ABI layout");
static_assert(sizeof(WotrNation) == 3 * sizeof(int32_t), "WotrNation ABI layout");
static_assert(offsetof(WotrUnit, kind) == 2 * sizeof(int32_t), "WotrUnit ABI offsets");
static_assert(offsetof(WotrLeader, nazgul) == 3 * sizeof(int32_t), "WotrLeader ABI offsets");
static_assert(offsetof(WotrRegion, fellowship) == 4 * sizeof(int32_t), "WotrRegion ABI offsets");
static_assert(offsetof(WotrNation, active) == 2 * sizeof(int32_t), "WotrNation ABI offsets");
thread_local std::array<char, 512> last_error{};

void set_error(const char* message) noexcept {
    if (!message) message = "unknown native error";
    const auto length = std::min(std::strlen(message), last_error.size() - 1);
    std::memcpy(last_error.data(), message, length);
    last_error[length] = '\0';
}

template<class Result, class Function>
Result checked(Function function, Result failure) noexcept {
    last_error[0] = '\0';
    try { return function(); }
    catch (const std::exception& error) { set_error(error.what()); }
    catch (...) { set_error("unknown native exception"); }
    return failure;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
bool flag(int32_t value) noexcept { return value == 0 || value == 1; }

template<class Record>
void check_array(const Record* records, size_t count) {
    require(count == 0 || records != nullptr, "nonempty array has a null pointer");
    require(count <= std::vector<Record>().max_size(), "record count exceeds vector capacity");
    require(count <= static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) / sizeof(Record),
            "record count exceeds addressable byte range");
}

template<class Record>
std::vector<Record> copy_array(const Record* records, size_t count) {
    // Pointer arithmetic on a null pointer is avoided even for an empty array.
    return count ? std::vector<Record>(records, records + count) : std::vector<Record>();
}

size_t observation_size(size_t regions, size_t nations) {
    const size_t maximum = std::numeric_limits<size_t>::max();
    require(nations <= (maximum - 14) / 2, "nation feature width overflows");
    const size_t width = 14 + 2 * nations;
    require(nations <= maximum / 4, "political feature width overflows");
    const size_t politics = 4 * nations;
    require(regions <= (maximum - politics) / width, "observation count overflows");
    const size_t count = regions * width + politics;
    require(count <= static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) / sizeof(float),
            "observation exceeds addressable byte range");
    return count;
}

// CSR indexes preserve input order within each region. The original Unit/Leader
// vectors also preserve their global order and own every input record.
template<class Record>
void index_regions(const std::vector<Record>& records, size_t region_count,
                   std::vector<size_t>& offsets, std::vector<size_t>& order) {
    offsets.assign(region_count + 1, 0);
    for (const auto& record : records) ++offsets[static_cast<size_t>(record.region) + 1];
    for (size_t i = 1; i < offsets.size(); ++i) offsets[i] += offsets[i - 1];
    order.resize(records.size());
    auto next = offsets;
    for (size_t i = 0; i < records.size(); ++i)
        order[next[static_cast<size_t>(records[i].region)]++] = i;
}

struct Position {
    std::vector<WotrRegion> regions;
    std::vector<WotrNation> nations;
    std::vector<WotrUnit> units;
    std::vector<WotrLeader> leaders;
    std::vector<size_t> unit_offsets, unit_order, leader_offsets, leader_order;
    size_t feature_count;
};

const Position& position(const void* handle) {
    require(handle != nullptr, "position handle is null");
    return *static_cast<const Position*>(handle);
}

std::unique_ptr<Position> create(const WotrRegion* regions, size_t region_count,
                                const WotrNation* nations, size_t nation_count,
                                const WotrUnit* units, size_t unit_count,
                                const WotrLeader* leaders, size_t leader_count) {
    check_array(regions, region_count); check_array(nations, nation_count);
    check_array(units, unit_count); check_array(leaders, leader_count);
    const auto index_limit = static_cast<size_t>(std::numeric_limits<int32_t>::max());
    require(region_count <= index_limit, "too many indexed regions");
    require(nation_count <= index_limit, "too many indexed nations");
    const size_t feature_count = observation_size(region_count, nation_count);
    for (size_t i = 0; i < region_count; ++i) {
        const auto& r = regions[i];
        require(flag(r.occupied) && flag(r.besieged) && flag(r.controlled_fp)
                && flag(r.controlled_sh) && flag(r.fellowship), "invalid region boolean");
    }
    for (size_t i = 0; i < nation_count; ++i) {
        const auto& n = nations[i];
        require(flag(n.side), "invalid nation side");
        require(flag(n.active), "invalid nation active boolean");
        require(n.box >= 0 && n.box <= 3, "political box must be between zero and three");
    }
    for (size_t i = 0; i < unit_count; ++i) {
        const auto& u = units[i];
        require(u.region >= 0 && static_cast<size_t>(u.region) < region_count, "unit region index out of range");
        require(u.nation >= 0 && static_cast<size_t>(u.nation) < nation_count, "unit nation index out of range");
        require(flag(u.kind), "invalid unit kind");
    }
    for (size_t i = 0; i < leader_count; ++i) {
        const auto& l = leaders[i];
        require(l.region >= 0 && static_cast<size_t>(l.region) < region_count, "leader region index out of range");
        require(flag(l.side) && flag(l.named) && flag(l.nazgul), "invalid leader classification");
        require(!(l.named && l.nazgul), "named and Nazgul classifications are exclusive");
        require(!l.nazgul || l.side == 1, "Nazgul classification requires Shadow side");
    }
    auto p = std::make_unique<Position>();
    p->regions = copy_array(regions, region_count); p->nations = copy_array(nations, nation_count);
    p->units = copy_array(units, unit_count); p->leaders = copy_array(leaders, leader_count);
    p->feature_count = feature_count;
    index_regions(p->units, region_count, p->unit_offsets, p->unit_order);
    index_regions(p->leaders, region_count, p->leader_offsets, p->leader_order);
    return p;
}

void encode(const Position& p, int32_t side, float* output, size_t count) {
    require(flag(side), "invalid observation side");
    require(count == p.feature_count, "output count must equal observation size");
    require(count == 0 || output != nullptr, "nonempty output has a null pointer");
    std::vector<double> nation_counts(2 * p.nations.size());
    size_t index = 0;
    for (size_t region = 0; region < p.regions.size(); ++region) {
        const auto& r = p.regions[region];
        double regular[2]{}, elite[2]{}, leadership[2]{}, named[2]{}, nazgul[2]{};
        std::fill(nation_counts.begin(), nation_counts.end(), 0.0);
        if (r.occupied) {
            for (size_t j = p.unit_offsets[region]; j < p.unit_offsets[region + 1]; ++j) {
                const auto& u = p.units[p.unit_order[j]];
                const auto nation = static_cast<size_t>(u.nation);
                const auto unit_side = p.nations[nation].side;
                if (u.kind == 0) {
                    regular[unit_side] += 1.0;
                    nation_counts[2 * nation] += 1.0 / 15.0;
                } else {
                    elite[unit_side] += 1.0;
                    nation_counts[2 * nation + 1] += 1.0 / 5.0;
                }
            }
            for (size_t j = p.leader_offsets[region]; j < p.leader_offsets[region + 1]; ++j) {
                const auto& l = p.leaders[p.leader_order[j]];
                leadership[l.side] += 1.0;
                if (l.named) named[l.side] += 1.0;
                if (l.nazgul) nazgul[l.side] += 1.0;
            }
        }
        const auto enemy = 1 - side;
        const double values[] = {
            regular[side] / 10.0, elite[side] / 10.0, leadership[side] / 7.0,
            regular[enemy] / 10.0, elite[enemy] / 10.0, leadership[enemy] / 7.0,
            static_cast<double>(r.besieged),
            static_cast<double>(side == 0 ? r.controlled_fp : r.controlled_sh),
            static_cast<double>(side == 0 ? r.controlled_sh : r.controlled_fp),
            static_cast<double>(r.fellowship), named[side] / 3.0, named[enemy] / 3.0,
            nazgul[side] / 8.0, nazgul[enemy] / 8.0
        };
        for (double value : values) output[index++] = static_cast<float>(value);
        for (double value : nation_counts) output[index++] = static_cast<float>(value);
    }
    for (const auto& n : p.nations) {
        output[index++] = static_cast<float>(static_cast<double>(n.box) / 3.0);
        output[index++] = static_cast<float>(n.active);
        output[index++] = n.box >= 3 ? 1.0f : 0.0f;
        output[index++] = n.side == side ? 1.0f : 0.0f;
    }
}
} // namespace

extern "C" {
uint32_t wotr_abi_version() { return 1; }
const char* wotr_last_error() { return last_error.data(); }
void* wotr_position_create(const WotrRegion* r, size_t nr, const WotrNation* n, size_t nn,
                           const WotrUnit* u, size_t nu, const WotrLeader* l, size_t nl) {
    return checked<void*>([&]() -> void* { return create(r, nr, n, nn, u, nu, l, nl).release(); }, nullptr);
}
void* wotr_position_clone(const void* handle) {
    return checked<void*>([&]() -> void* { return new Position(position(handle)); }, nullptr);
}
void wotr_position_destroy(void* handle) {
    checked<int>([&]() { delete static_cast<Position*>(handle); return 0; }, -1);
}
size_t wotr_observation_size(const void* handle) {
    return checked<size_t>([&]() { return position(handle).feature_count; }, 0);
}
int wotr_encode_regions_politics(const void* handle, int32_t side, float* output, size_t count) {
    return checked<int>([&]() { encode(position(handle), side, output, count); return 0; }, -1);
}
int wotr_set_political_box(void* handle, size_t nation, int32_t box) {
    return checked<int>([&]() {
        const auto& original = position(handle);
        require(nation < original.nations.size(), "political nation index out of range");
        require(box >= 0 && box <= 3, "political box must be between zero and three");
        static_cast<Position*>(handle)->nations[nation].box = box;
        return 0;
    }, -1);
}
}
