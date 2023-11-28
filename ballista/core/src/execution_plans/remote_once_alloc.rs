// you can only alloc once.

use std::{
    alloc::{Allocator, Global, Layout},
    ffi::CString,
    marker::PhantomData,
    ptr::{null_mut, NonNull},
    sync::atomic::{AtomicUsize, Ordering},
};

use std::sync::atomic::AtomicBool;

pub struct RemoteOnceAlloc {
    shm_path: String,
    has_allocated: AtomicBool,
    allocated_ptr: AtomicUsize,
}

impl RemoteOnceAlloc {
    pub fn new(shm_path: &str) -> Self {
        Self {
            shm_path: shm_path.to_string(),
            has_allocated: AtomicBool::new(false),
            allocated_ptr: AtomicUsize::new(0),
        }
    }
}

impl RemoteOnceAlloc {
    pub fn has_allocated(&self) -> bool {
        self.has_allocated.load(Ordering::SeqCst)
    }

    pub fn allocated_ptr(&self) -> *mut u8 {
        let val = self.allocated_ptr.load(Ordering::SeqCst);
        val as *mut u8
    }
}

unsafe impl Allocator for RemoteOnceAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        let ok = self.has_allocated.compare_exchange_weak(
            false,
            true,
            Ordering::SeqCst,
            Ordering::SeqCst,
        );
        if ok.is_err() {
            return Err(std::alloc::AllocError);
        }

        let shm_name = CString::new(self.shm_path.clone()).unwrap();
        let raw_fd = unsafe {
            libc::shm_open(
                shm_name.as_ptr(),
                libc::O_CREAT | libc::O_RDWR,
                libc::S_IRUSR | libc::S_IWUSR,
            )
        };
        if raw_fd < 0 {
            panic!("Failed to create shared memory");
        }

        // Set the size of the shared memory object.
        if unsafe { libc::ftruncate(raw_fd, layout.size() as _) } < 0 {
            panic!("Failed to set shared memory size");
        }

        let heap_start_addr = unsafe {
            libc::mmap(
                null_mut(),
                layout.size(),
                libc::PROT_WRITE,
                libc::MAP_SHARED,
                raw_fd,
                0,
            )
        };

        if heap_start_addr == libc::MAP_FAILED {
            panic!("Failed to mmap shared memory");
        }
        self.allocated_ptr
            .store(heap_start_addr as usize, Ordering::SeqCst);
        let ptr_slice =
            std::ptr::slice_from_raw_parts_mut(heap_start_addr as *mut u8, layout.size());
        Ok(std::ptr::NonNull::new(ptr_slice).unwrap())
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        // no-op
    }
}

/// This sturct is intentionally to have the same layout as the hashbrown::raw::RawTable<T>,
/// so that we can transmute between them.
#[allow(dead_code)]
struct RawTableCopy<T, A: Allocator = Global> {
    table: RawTableCopyInner,
    alloc: A,
    // Tell dropck that we own instances of T.
    marker: PhantomData<T>,
}

#[allow(dead_code)]
struct RawTableCopyInner {
    bucket_mask: usize,
    ctrl: NonNull<u8>,
    growth_left: usize,
    items: usize,
}

pub struct RawTableReconstructor {}

impl RawTableReconstructor {
    /// returns (bucket_mask, start, growth_left, items)
    /// # Safety
    /// Never call it unless you know what you are doing.
    pub unsafe fn into_raw_parts<T, A: Allocator>(
        ht: hashbrown::raw::RawTable<T, A>,
    ) -> (usize, *mut T, usize, usize) {
        let buckets = ht.buckets();
        let bucket_mask = buckets - 1;
        let items = ht.len();
        let growth_left = ht.capacity() - items;
        let start = ht.data_start();

        (bucket_mask, start.as_ptr(), growth_left, items)
    }

    /// by the time the table is reconstructed, no new allocation shall happen.
    /// # Safety
    /// Never call it unless you know what you are doing.
    pub unsafe fn from_raw_parts<T>(
        bucket_mask: usize,
        start: *mut T,
        growth_left: usize,
        items: usize,
    ) -> hashbrown::raw::RawTable<T> {
        let buckets = bucket_mask + 1;
        let ctrl = unsafe { std::mem::transmute::<_, NonNull<u8>>(start.add(buckets)) };
        let raw_table = RawTableCopyInner {
            bucket_mask,
            ctrl,
            growth_left,
            items,
        };
        let table = RawTableCopy {
            table: raw_table,
            alloc: Global,
            marker: PhantomData::<T>,
        };

        unsafe { std::mem::transmute::<_, hashbrown::raw::RawTable<T>>(table) }
    }
}

#[cfg(test)]

mod tests {

    use ahash::RandomState;

    use super::*;

    #[test]
    fn test_hashmap() {
        let allocator = RemoteOnceAlloc::new("/shm-test");
        let mut ht = hashbrown::HashMap::with_capacity_in(1_000_000, &allocator);
        for i in 0..1_000_000 {
            ht.insert(i, i);
        }

        for i in 0..1_000_000 {
            assert_eq!(ht.get(&i), Some(&i));
        }
    }

    #[test]
    fn test_insert_and_recover() {
        let allocator = RemoteOnceAlloc::new("/shm-test");
        let random_state = RandomState::with_seed(42);

        let mut ht = hashbrown::raw::RawTable::with_capacity_in(1_000_000, &allocator);
        for i in 0..1_000_000 {
            let h = random_state.hash_one(i);
            let item = ht.get_mut(h, |(old, _v)| *old == h);
            assert!(item.is_none());
            ht.insert(h, (h, i), |(hash, _)| *hash);
        }

        for i in 0..1_000_000 {
            let h = random_state.hash_one(i);
            let item = ht.get_mut(h, |(old, _)| *old == h);
            assert!(item.is_some());
            assert!(item.unwrap().1 == i);
        }

        let raw_parts = unsafe { RawTableReconstructor::into_raw_parts(ht) };
        let mut ht = unsafe {
            RawTableReconstructor::from_raw_parts(
                raw_parts.0,
                raw_parts.1,
                raw_parts.2,
                raw_parts.3,
            )
        };
        for i in 0..1_000_000 {
            let h = random_state.hash_one(i);
            let item = ht.get_mut(h, |(old, _)| *old == h);
            assert!(item.is_some());
            assert!(item.unwrap().1 == i);
        }

        std::mem::forget(ht)
    }
}
