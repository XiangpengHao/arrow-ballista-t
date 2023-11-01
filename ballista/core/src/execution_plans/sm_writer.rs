use std::{
    collections::HashMap,
    ffi::CString,
    fs::File,
    io::Write,
    os::fd::AsRawFd,
    path::{Path, PathBuf},
    ptr::null_mut,
};

use datafusion::{
    arrow::{
        datatypes::Schema,
        error::{ArrowError, Result},
        ipc::{
            convert,
            writer::{
                write_message, DictionaryTracker, FileWriter, IpcDataGenerator,
                IpcWriteOptions,
            },
            Block, FooterBuilder, MetadataVersion,
        },
        record_batch::RecordBatch,
    },
    error::DataFusionError,
};
use flatbuffers::FlatBufferBuilder;

pub struct MemoryWriter {
    ptr: *mut u8,
    capacity: usize,
    offset: usize,
}

unsafe impl Send for MemoryWriter {}

impl MemoryWriter {
    pub fn new(ptr: *mut u8, capacity: usize) -> Self {
        Self {
            ptr,
            capacity,
            offset: 0,
        }
    }
}

impl Write for MemoryWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        assert!(self.capacity - self.offset >= buf.len());
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.as_ptr(),
                self.ptr.add(self.offset),
                buf.len(),
            );
        }
        self.offset += buf.len();
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub struct NoFlushFile {
    raw_fd: i32,
}

impl NoFlushFile {
    pub fn new(raw_fd: i32) -> Self {
        Self { raw_fd }
    }
}

impl Write for NoFlushFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let len = unsafe {
            libc::write(self.raw_fd, buf.as_ptr() as *const libc::c_void, buf.len())
        };
        if len < 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(len as usize)
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub struct SharedMemoryByteWriter {
    pub identifier: String,
    pub num_batches: u64,
    pub num_rows: u64,
    pub num_bytes: u64,
    writer: NoBuffWriter<MemoryWriter>,
}

impl SharedMemoryByteWriter {
    pub fn new(identifier: String, schema: &Schema, size: usize) -> Result<Self> {
        let shm_name = CString::new(identifier.clone()).unwrap();
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
        if unsafe { libc::ftruncate(raw_fd, size as i64) } < 0 {
            panic!("Failed to set shared memory size");
        }

        let shm_ptr = unsafe {
            libc::mmap(
                null_mut(),
                size,
                libc::PROT_WRITE,
                libc::MAP_SHARED,
                raw_fd,
                0,
            )
        } as *mut u8;

        let write = MemoryWriter::new(shm_ptr, size);

        let writer = NoBuffWriter::try_new(write, schema).unwrap();

        Ok(Self {
            identifier,
            num_batches: 0,
            num_rows: 0,
            num_bytes: 0,
            writer,
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

pub struct SharedMemoryFileWriter {
    pub path: String,
    pub num_batches: u64,
    pub num_rows: u64,
    pub num_bytes: u64,
    pub writer: FileWriter<NoFlushFile>,
}

impl SharedMemoryFileWriter {
    pub fn new(path: String, schema: &Schema) -> Result<Self> {
        let shm_name = CString::new(path.clone()).unwrap();
        let raw_fd = unsafe {
            libc::shm_open(
                shm_name.as_ptr(),
                libc::O_CREAT | libc::O_RDWR,
                libc::S_IRUSR | libc::S_IWUSR,
            )
        };
        if raw_fd < 0 {
            panic!("Failed to create shared memory: {}", path);
        }

        let file = NoFlushFile::new(raw_fd);

        Ok(Self {
            path,
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

    pub fn path(&self) -> &str {
        &self.path
    }
}

/// Mirror of arrow_ipc::FileWriter, but without the BufWriter
struct NoBuffWriter<W: Write> {
    writer: W,
    /// IPC write options
    write_options: IpcWriteOptions,
    /// A reference to the schema, used in validating record batches
    schema: Schema,
    /// The number of bytes between each block of bytes, as an offset for random access
    block_offsets: usize,
    /// Dictionary blocks that will be written as part of the IPC footer
    dictionary_blocks: Vec<Block>,
    /// Record blocks that will be written as part of the IPC footer
    record_blocks: Vec<Block>,
    /// Whether the writer footer has been written, and the writer is finished
    finished: bool,
    /// Keeps track of dictionaries that have been written
    dictionary_tracker: DictionaryTracker,
    /// User level customized metadata
    custom_metadata: HashMap<String, String>,

    data_gen: IpcDataGenerator,
}

const ARROW_MAGIC: [u8; 6] = [b'A', b'R', b'R', b'O', b'W', b'1'];
const CONTINUATION_MARKER: [u8; 4] = [0xff; 4];

impl<W: Write> NoBuffWriter<W> {
    /// Try create a new writer, with the schema written as part of the header
    pub fn try_new(writer: W, schema: &Schema) -> Result<Self> {
        let write_options = IpcWriteOptions::default();
        Self::try_new_with_options(writer, schema, write_options)
    }

    /// Try create a new writer with IpcWriteOptions
    pub fn try_new_with_options(
        mut writer: W,
        schema: &Schema,
        write_options: IpcWriteOptions,
    ) -> Result<Self> {
        let data_gen = IpcDataGenerator::default();
        // write magic to header aligned on 8 byte boundary
        let header_size = ARROW_MAGIC.len() + 2;
        assert_eq!(header_size, 8);
        writer.write_all(&ARROW_MAGIC[..])?;
        writer.write_all(&[0, 0])?;
        // write the schema, set the written bytes to the schema + header
        let encoded_message = data_gen.schema_to_bytes(schema, &write_options);
        let (meta, data) = write_message(&mut writer, encoded_message, &write_options)?;
        Ok(Self {
            writer,
            write_options,
            schema: schema.clone(),
            block_offsets: meta + data + header_size,
            dictionary_blocks: vec![],
            record_blocks: vec![],
            finished: false,
            dictionary_tracker: DictionaryTracker::new(true),
            custom_metadata: HashMap::new(),
            data_gen,
        })
    }

    /// Write a record batch to the file
    pub fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        if self.finished {
            return Err(ArrowError::IoError(
                "Cannot write record batch to file writer as it is closed".to_string(),
            ));
        }

        let (encoded_dictionaries, encoded_message) = self.data_gen.encoded_batch(
            batch,
            &mut self.dictionary_tracker,
            &self.write_options,
        )?;

        for encoded_dictionary in encoded_dictionaries {
            let (meta, data) =
                write_message(&mut self.writer, encoded_dictionary, &self.write_options)?;

            let block = Block::new(self.block_offsets as i64, meta as i32, data as i64);
            self.dictionary_blocks.push(block);
            self.block_offsets += meta + data;
        }

        let (meta, data) =
            write_message(&mut self.writer, encoded_message, &self.write_options)?;
        // add a record block for the footer
        let block = Block::new(
            self.block_offsets as i64,
            meta as i32, // TODO: is this still applicable?
            data as i64,
        );
        self.record_blocks.push(block);
        self.block_offsets += meta + data;
        Ok(())
    }

    /// Write footer and closing tag, then mark the writer as done
    pub fn finish(&mut self) -> Result<()> {
        if self.finished {
            return Err(ArrowError::IoError(
                "Cannot write footer to file writer as it is closed".to_string(),
            ));
        }

        // write EOS
        write_continuation(&mut self.writer, 0)?;

        let mut fbb = FlatBufferBuilder::new();
        let dictionaries = fbb.create_vector(&self.dictionary_blocks);
        let record_batches = fbb.create_vector(&self.record_blocks);
        let schema = convert::schema_to_fb_offset(&mut fbb, &self.schema);
        let fb_custom_metadata = (!self.custom_metadata.is_empty())
            .then(|| convert::metadata_to_fb(&mut fbb, &self.custom_metadata));

        let root = {
            let mut footer_builder = FooterBuilder::new(&mut fbb);
            footer_builder.add_version(MetadataVersion::V5);
            footer_builder.add_schema(schema);
            footer_builder.add_dictionaries(dictionaries);
            footer_builder.add_recordBatches(record_batches);
            if let Some(fb_custom_metadata) = fb_custom_metadata {
                footer_builder.add_custom_metadata(fb_custom_metadata);
            }
            footer_builder.finish()
        };
        fbb.finish(root, None);
        let footer_data = fbb.finished_data();
        self.writer.write_all(footer_data)?;
        self.writer
            .write_all(&(footer_data.len() as i32).to_le_bytes())?;
        self.writer.write_all(&ARROW_MAGIC)?;
        self.writer.flush()?;
        self.finished = true;

        Ok(())
    }
}

/// Write a record batch to the writer, writing the message size before the message
/// if the record batch is being written to a stream
fn write_continuation<W: Write>(mut writer: W, total_len: i32) -> Result<usize> {
    let written = 8;

    // write continuation marker and message length
    writer.write_all(&CONTINUATION_MARKER)?;
    writer.write_all(&total_len.to_le_bytes()[..])?;

    writer.flush()?;

    Ok(written)
}

/// Write in Arrow IPC format.
pub struct IPCWriter {
    /// path
    pub path: PathBuf,
    /// inner writer
    pub writer: FileWriter<File>,
    /// batches written
    pub num_batches: u64,
    /// rows written
    pub num_rows: u64,
    /// bytes written
    pub num_bytes: u64,
}

fn advise_no_cache(file: &File) -> i32 {
    unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) }
}

impl IPCWriter {
    /// Create new writer
    pub fn new(path: &Path, schema: &Schema) -> Result<Self> {
        let file = File::create(path).map_err(|e| {
            DataFusionError::Execution(format!(
                "Failed to create partition file at {path:?}: {e:?}"
            ))
        })?;
        advise_no_cache(&file);
        Ok(Self {
            num_batches: 0,
            num_rows: 0,
            num_bytes: 0,
            path: path.into(),
            writer: FileWriter::try_new(file, schema)?,
        })
    }

    /// Write one single batch
    pub fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        self.writer.write(batch)?;
        self.num_batches += 1;
        self.num_rows += batch.num_rows() as u64;
        let num_bytes: usize = batch.get_array_memory_size();
        self.num_bytes += num_bytes as u64;
        Ok(())
    }

    /// Finish the writer
    pub fn finish(&mut self) -> Result<()> {
        self.writer.finish()?;
        advise_no_cache(self.writer.get_ref());
        Ok(())
    }

    /// Path write to
    pub fn path(&self) -> &Path {
        &self.path
    }
}
