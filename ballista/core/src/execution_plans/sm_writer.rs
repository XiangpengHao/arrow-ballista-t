use std::{ffi::CString, fs::File, io::Write, os::fd::FromRawFd};

use datafusion::arrow::{
    datatypes::Schema, error::Result, ipc::writer::FileWriter, record_batch::RecordBatch,
};

pub struct MemoryWriter {
    ptr: *mut u8,
    capacity: usize,
}

impl Write for MemoryWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        assert!(self.capacity >= buf.len());
        unsafe {
            std::ptr::copy_nonoverlapping(buf.as_ptr(), self.ptr, buf.len());
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub struct SharedMemoryWriter {
    pub identifier: String,
    pub num_batches: u64,
    pub num_rows: u64,
    pub num_bytes: u64,
    pub writer: FileWriter<File>,
}

impl SharedMemoryWriter {
    pub fn new(identifier: String, schema: &Schema) -> Result<Self> {
        let shm_name = CString::new("/ballista-shm-".to_owned() + &identifier).unwrap();
        let raw_fd = unsafe {
            libc::shm_open(
                shm_name.as_ptr(),
                libc::O_CREAT | libc::O_RDWR,
                libc::S_IRUSR | libc::S_IWUSR,
            )
        };
        if raw_fd == -1 {
            panic!("Failed to create shared memory");
        }

        let file = unsafe { File::from_raw_fd(raw_fd) };

        Ok(Self {
            identifier,
            num_batches: 0,
            num_rows: 0,
            num_bytes: 0,
            writer: FileWriter::try_new(file, schema).unwrap(),
        })
    }

    pub fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        self.writer.write(batch)?;
        self.num_batches += 1;
        self.num_rows += batch.num_rows() as u64;
        self.num_bytes += batch.get_array_memory_size() as u64;
        Ok(())
    }

    pub fn finish(&mut self) -> Result<()> {
        self.writer.finish().map_err(Into::into)
    }

    pub fn identifier(&self) -> &str {
        &self.identifier
    }
}
