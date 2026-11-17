# file: cpp/search_controller.inc
# targets: wasm32 wasm64 native
# Single-threaded WebAssembly has no threads: the controller runs its search on a std::thread and waits on a
# condition variable for every inference reply (await_reply), the caller answering through advance/request/submit.
# With a provider registered (wotr_search_controller_set_sync_provider, patch 0004), advance() runs the whole search
# in the calling thread and await_reply calls the provider instead of waiting: the provider reads the request
# (wotr_search_controller_request / _request_payload) and answers it (wotr_search_controller_submit /
# _submit_payload) exactly as the Python driver does between two advance() calls, so the search sees the same
# sequence of requests and replies. Without a provider nothing changes. The root-priors and root-plan exchanges
# (add_noise, WOTR_ROOT_BREADTH; neither is used by the live search) are refused in this mode.
s#std::thread worker;std::mutex mu;std::condition_variable cv;#&int (*sync_provider)(void*,void*)=nullptr;void* sync_user=nullptr;   // wasm patch 0003#
s#{Scoped _sp(this,"await_reply");cv.wait(#{Scoped _sp(this,"await_reply");if(sync_provider){lk.unlock();const int sync_rc=sync_provider(sync_user,this);lk.lock();require(sync_rc==0\&\&phase==Phase::ReplyReady,"The single-thread inference provider did not answer the request");phase=Phase::Resumed;}else cv.wait(#
s#worker=std::thread(\[this\]{run();});#if(sync_provider){lk.unlock();run();lk.lock();}else worker=std::thread([this]{run();});#
s#phase=Phase::RootPriorsPending;cv.notify_all();#require(!sync_provider,"Root-priors exchange (add_noise) is not available in the single-thread controller");&#
s#phase=Phase::RootPlanPending;cv.notify_all();#require(!sync_provider,"The root plan exchange is not available in the single-thread controller");&#
