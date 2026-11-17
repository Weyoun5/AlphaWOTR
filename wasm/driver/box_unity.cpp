// The live box's engine translation unit (cpp/snapshot.cpp: every rule, menu, encoder and the search controller)
// followed by the turn search box's combat kernel and the module's COMBAT_DIST reply, which use its internal types.
#include "snapshot.cpp"
#include "chance_driver.inc"   // cd_pow, cd_comb (CPython float_pow, math.comb) that the kernel uses
#include "combat_kernel.inc"
#include "combat_reply.inc"
