// The live box's engine translation unit (cpp/snapshot.cpp: every rule, menu, encoder and the search controller)
// followed by the turn search box's combat kernel, the module's COMBAT_DIST reply and ai._strictly_better, which use its
// internal types.
#include "snapshot.cpp"
#include "chance_driver.inc"   // cd_pow, cd_comb (CPython float_pow, math.comb) that the kernel uses
#include "combat_kernel.inc"
#include "combat_reply.inc"
#include "strictly_better.inc"
// board.sanitize for the adapter's port (adapter/adapter.h)
std::string awad_sanitize(const std::string& name) { return sanitize_name(name); }
