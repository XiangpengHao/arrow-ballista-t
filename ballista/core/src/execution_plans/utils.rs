use std::{
    alloc::{Allocator, Layout},
    sync::Arc,
};

use datafusion::arrow::{
    array::{Array, ArrayData, PrimitiveArray},
    buffer::Buffer,
    datatypes::{DataType, Int32Type},
    record_batch::RecordBatch,
};

#[allow(dead_code)]
fn clone_with_custome_allocator<Alloc: Allocator>(
    original: &RecordBatch,
    allocator: &Alloc,
) -> RecordBatch {
    let schema = original.schema();
    let num_columns = original.num_columns();
    let mut arrays = Vec::with_capacity(num_columns);

    for i in 0..num_columns {
        let array = original.column(i);
        let cloned_array = clone_array_with_allocator(array, allocator);
        arrays.push(cloned_array);
    }

    RecordBatch::try_new(schema.clone(), arrays).unwrap()
}

#[allow(dead_code)]
fn clone_array_with_allocator<Alloc: Allocator>(
    original: &dyn Array,
    allocator: &Alloc,
) -> Arc<dyn Array> {
    let original_data = original.to_data();
    let original_buffers = original_data.buffers();

    let mut new_buffers = Vec::new();
    for buffer in original_buffers {
        let layout = Layout::from_size_align(buffer.len(), 8).unwrap();
        let new_memory = allocator.allocate(layout).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer.as_ptr(),
                new_memory.as_ptr() as *mut u8,
                buffer.len(),
            );
        }
        let ptr =
            unsafe { std::ptr::NonNull::new_unchecked(new_memory.as_ptr() as *mut u8) };
        let buffer =
            unsafe { Buffer::from_custom_allocation(ptr, buffer.len(), Arc::new(())) };

        new_buffers.push(buffer);
    }
    let new_data = ArrayData::builder(original_data.data_type().clone())
        .buffers(new_buffers)
        .len(original_data.len())
        .build()
        .unwrap();

    match original.data_type() {
        DataType::Int32 => Arc::new(PrimitiveArray::<Int32Type>::from(new_data)),
        _ => todo!(),
    }
}
