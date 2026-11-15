#include <mimalloc.h>
#include <cstddef>
#include <new>

namespace {
// Use mi_malloc, not mi_new: preserve the standard unlimited new-handler retry
// contract even when a handler returns repeatedly before eventually throwing.
void* allocate(std::size_t n,std::size_t alignment){
    for(;;){
        void* p=alignment?mi_malloc_aligned(n?n:1,alignment):mi_malloc(n?n:1);
        if(p)return p;
        const auto handler=std::get_new_handler();
        if(!handler)throw std::bad_alloc();
        handler();
    }
}
}
void* operator new(std::size_t n){return allocate(n,0);}
void* operator new[](std::size_t n){return allocate(n,0);}
void operator delete(void* p) noexcept{mi_free(p);}
void operator delete[](void* p) noexcept{mi_free(p);}
void operator delete(void* p,std::size_t) noexcept{mi_free(p);}
void operator delete[](void* p,std::size_t) noexcept{mi_free(p);}
void* operator new(std::size_t n,const std::nothrow_t&) noexcept{try{return allocate(n,0);}catch(...){return nullptr;}}
void* operator new[](std::size_t n,const std::nothrow_t&) noexcept{try{return allocate(n,0);}catch(...){return nullptr;}}
void operator delete(void* p,const std::nothrow_t&) noexcept{mi_free(p);}
void operator delete[](void* p,const std::nothrow_t&) noexcept{mi_free(p);}
void* operator new(std::size_t n,std::align_val_t a){return allocate(n,static_cast<std::size_t>(a));}
void* operator new[](std::size_t n,std::align_val_t a){return allocate(n,static_cast<std::size_t>(a));}
void* operator new(std::size_t n,std::align_val_t a,const std::nothrow_t&) noexcept{try{return allocate(n,static_cast<std::size_t>(a));}catch(...){return nullptr;}}
void* operator new[](std::size_t n,std::align_val_t a,const std::nothrow_t&) noexcept{try{return allocate(n,static_cast<std::size_t>(a));}catch(...){return nullptr;}}
void operator delete(void* p,std::align_val_t) noexcept{mi_free(p);}
void operator delete[](void* p,std::align_val_t) noexcept{mi_free(p);}
void operator delete(void* p,std::size_t,std::align_val_t) noexcept{mi_free(p);}
void operator delete[](void* p,std::size_t,std::align_val_t) noexcept{mi_free(p);}
void operator delete(void* p,std::align_val_t,const std::nothrow_t&) noexcept{mi_free(p);}
void operator delete[](void* p,std::align_val_t,const std::nothrow_t&) noexcept{mi_free(p);}

