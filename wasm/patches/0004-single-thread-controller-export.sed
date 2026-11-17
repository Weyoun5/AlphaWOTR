# file: cpp/search_controller_exports.inc
# targets: wasm32 wasm64 native
# The export of patch 0003: register the provider (null: back to the threaded mode) before the first advance().
s#^WOTR_API void wotr_search_controller_destroy(void\* ctl){delete static_cast<sc::Controller\*>(ctl);}#&\
WOTR_API int wotr_search_controller_set_sync_provider(void* ctl,int (*provider)(void*,void*),void* user){return checked<int>([\&](){auto\& c=sc::controller(ctl);std::lock_guard<std::mutex> lk(c.mu);require(c.phase==sc::Controller::Phase::Idle,"The provider is set before the first advance");c.sync_provider=provider;c.sync_user=user;return 0;},-1);}#
