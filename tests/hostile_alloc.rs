//! A list count inside the cap must fail before element storage is allocated.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use kiln::{decode, Error, Schema};

static TRACK: AtomicBool = AtomicBool::new(false);
static BYTES: AtomicUsize = AtomicUsize::new(0);

struct Audit;

unsafe impl GlobalAlloc for Audit {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the layout comes from the caller. System receives it unchanged.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && TRACK.load(Ordering::Relaxed) {
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by this allocator with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded with the same layout.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() && TRACK.load(Ordering::Relaxed) {
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwarded. Only growth is counted.
        let next = unsafe { System.realloc(ptr, layout, new_size) };
        if !next.is_null() && TRACK.load(Ordering::Relaxed) && new_size > layout.size() {
            BYTES.fetch_add(new_size - layout.size(), Ordering::Relaxed);
        }
        next
    }
}

#[global_allocator]
static AUDIT: Audit = Audit;

#[test]
fn hostile_list_does_not_allocate() {
    let schema = Schema::list(Schema::struct_fields([("a", Schema::I64)]));
    let buf = [0x00u8, 0x00, 0x00, 0x01];
    BYTES.store(0, Ordering::SeqCst);
    TRACK.store(true, Ordering::SeqCst);
    let result = decode(&schema, &buf);
    TRACK.store(false, Ordering::SeqCst);
    let used = BYTES.load(Ordering::SeqCst);
    assert_eq!(result.unwrap_err(), Error::Eof);
    assert_eq!(used, 0, "hostile list allocated {used} bytes");
}
