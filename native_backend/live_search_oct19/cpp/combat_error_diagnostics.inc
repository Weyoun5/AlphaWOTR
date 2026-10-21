// Compact army projection for the frozen search's PlanMismatch diagnostic.
// Preserve actual Army/leader identities, fields and aliases within the result.
std::unique_ptr<Snapshot> native_combat_search_army(const Snapshot& s,const Snapshot& args){
    const auto region=std::string(text(args,args.root));
    auto army=cbs_lookup(s,region);csf_small(s,army);
    sm_Builder b(s);b.out.root=b.raw(army);return std::make_unique<Snapshot>(b.out);
}
