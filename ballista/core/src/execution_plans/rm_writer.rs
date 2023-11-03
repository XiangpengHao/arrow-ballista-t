use std::{
    collections::HashMap,
    ffi::CString,
    fmt,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    os::fd::{AsRawFd, FromRawFd},
    path::{Path, PathBuf},
    ptr::null_mut,
    sync::Arc,
};

use datafusion::{
    arrow::{
        array::ArrayRef,
        buffer::MutableBuffer,
        datatypes::{Schema, SchemaRef},
        error::{ArrowError, Result},
        ipc::{
            convert,
            reader::{read_dictionary, read_record_batch},
            root_as_footer, root_as_message,
            writer::{
                write_message, DictionaryTracker, FileWriter, IpcDataGenerator,
                IpcWriteOptions,
            },
            Block, FooterBuilder, MessageHeader, MetadataVersion,
        },
        record_batch::{RecordBatch, RecordBatchReader},
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

pub struct MemoryReader {
    ptr: *const u8,
    capacity: usize,
    offset: usize,
}

unsafe impl Send for MemoryReader {}

impl MemoryReader {
    pub fn new(ptr: *const u8, capacity: usize) -> Self {
        Self {
            ptr,
            capacity,
            offset: 0,
        }
    }
}

impl Read for MemoryReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        assert!(self.capacity - self.offset >= buf.len());
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.ptr.add(self.offset),
                buf.as_mut_ptr(),
                buf.len(),
            );
        }
        self.offset += buf.len();
        Ok(buf.len())
    }
}

impl Seek for MemoryReader {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match pos {
            SeekFrom::Start(offset) => {
                self.offset = offset as usize;
            }
            SeekFrom::End(offset) => {
                self.offset = self.capacity - offset as usize;
            }
            SeekFrom::Current(offset) => {
                self.offset += offset as usize;
            }
        }
        Ok(self.offset as u64)
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
    pub writer: FileWriter<File>,
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

        let file = unsafe { File::from_raw_fd(raw_fd) };

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
pub struct BuffedDirectWriter {
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

impl BuffedDirectWriter {
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

pub struct NoBufReader<R: Read + Seek> {
    /// Buffered file reader that supports reading and seeking
    reader: R,

    /// The schema that is read from the file header
    schema: SchemaRef,

    /// The blocks in the file
    ///
    /// A block indicates the regions in the file to read to get data
    blocks: Vec<Block>,

    /// A counter to keep track of the current block that should be read
    current_block: usize,

    /// The total number of blocks, which may contain record batches and other types
    total_blocks: usize,

    /// Optional dictionaries for each schema field.
    ///
    /// Dictionaries may be appended to in the streaming format.
    dictionaries_by_id: HashMap<i64, ArrayRef>,

    /// Metadata version
    metadata_version: MetadataVersion,

    /// Optional projection and projected_schema
    projection: Option<(Vec<usize>, Schema)>,
}

impl<R: Read + Seek> fmt::Debug for NoBufReader<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::result::Result<(), fmt::Error> {
        f.debug_struct("FileReader<R>")
            .field("reader", &"BufReader<..>")
            .field("schema", &self.schema)
            .field("blocks", &self.blocks)
            .field("current_block", &self.current_block)
            .field("total_blocks", &self.total_blocks)
            .field("dictionaries_by_id", &self.dictionaries_by_id)
            .field("metadata_version", &self.metadata_version)
            .field("projection", &self.projection)
            .finish()
    }
}

impl<R: Read + Seek> NoBufReader<R> {
    /// Try to create a new file reader
    ///
    /// Returns errors if the file does not meet the Arrow Format header and footer
    /// requirements
    pub fn try_new(reader: R, projection: Option<Vec<usize>>) -> Result<Self> {
        let mut reader = reader;
        // check if header and footer contain correct magic bytes
        let mut magic_buffer: [u8; 6] = [0; 6];
        reader.read_exact(&mut magic_buffer)?;
        if magic_buffer != ARROW_MAGIC {
            return Err(ArrowError::IoError(
                "Arrow file does not contain correct header".to_string(),
            ));
        }
        reader.seek(SeekFrom::End(-6))?;
        reader.read_exact(&mut magic_buffer)?;
        if magic_buffer != ARROW_MAGIC {
            return Err(ArrowError::IoError(
                "Arrow file does not contain correct footer".to_string(),
            ));
        }
        // read footer length
        let mut footer_size: [u8; 4] = [0; 4];
        reader.seek(SeekFrom::End(-10))?;
        reader.read_exact(&mut footer_size)?;
        let footer_len = i32::from_le_bytes(footer_size);

        // read footer
        let mut footer_data = vec![0; footer_len as usize];
        reader.seek(SeekFrom::End(-10 - footer_len as i64))?;
        reader.read_exact(&mut footer_data)?;

        let footer = root_as_footer(&footer_data[..]).map_err(|err| {
            ArrowError::IoError(format!("Unable to get root as footer: {err:?}"))
        })?;

        let blocks = footer.recordBatches().ok_or_else(|| {
            ArrowError::IoError(
                "Unable to get record batches from IPC Footer".to_string(),
            )
        })?;

        let total_blocks = blocks.len();

        let ipc_schema = footer.schema().unwrap();
        let schema = convert::fb_to_schema(ipc_schema);

        // Create an array of optional dictionary value arrays, one per field.
        let mut dictionaries_by_id = HashMap::new();
        if let Some(dictionaries) = footer.dictionaries() {
            for block in dictionaries {
                // read length from end of offset
                let mut message_size: [u8; 4] = [0; 4];
                reader.seek(SeekFrom::Start(block.offset() as u64))?;
                reader.read_exact(&mut message_size)?;
                if message_size == CONTINUATION_MARKER {
                    reader.read_exact(&mut message_size)?;
                }
                let footer_len = i32::from_le_bytes(message_size);
                let mut block_data = vec![0; footer_len as usize];

                reader.read_exact(&mut block_data)?;

                let message = root_as_message(&block_data[..]).map_err(|err| {
                    ArrowError::IoError(format!("Unable to get root as message: {err:?}"))
                })?;

                match message.header_type() {
                    MessageHeader::DictionaryBatch => {
                        let batch = message.header_as_dictionary_batch().unwrap();

                        // read the block that makes up the dictionary batch into a buffer
                        let mut buf =
                            MutableBuffer::from_len_zeroed(message.bodyLength() as usize);
                        reader.seek(SeekFrom::Start(
                            block.offset() as u64 + block.metaDataLength() as u64,
                        ))?;
                        reader.read_exact(&mut buf)?;

                        read_dictionary(
                            &buf.into(),
                            batch,
                            &schema,
                            &mut dictionaries_by_id,
                            &message.version(),
                        )?;
                    }
                    t => {
                        return Err(ArrowError::IoError(format!(
                            "Expecting DictionaryBatch in dictionary blocks, found {t:?}."
                        )));
                    }
                }
            }
        }
        let projection = match projection {
            Some(projection_indices) => {
                let schema = schema.project(&projection_indices)?;
                Some((projection_indices, schema))
            }
            _ => None,
        };

        Ok(Self {
            reader,
            schema: Arc::new(schema),
            blocks: blocks.iter().copied().collect(),
            current_block: 0,
            total_blocks,
            dictionaries_by_id,
            metadata_version: footer.version(),
            projection,
        })
    }

    /// Return the schema of the file
    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn maybe_next(&mut self) -> Result<Option<RecordBatch>> {
        let block = self.blocks[self.current_block];
        self.current_block += 1;

        // read length
        self.reader.seek(SeekFrom::Start(block.offset() as u64))?;
        let mut meta_buf = [0; 4];
        self.reader.read_exact(&mut meta_buf)?;
        if meta_buf == CONTINUATION_MARKER {
            // continuation marker encountered, read message next
            self.reader.read_exact(&mut meta_buf)?;
        }
        let meta_len = i32::from_le_bytes(meta_buf);

        let mut block_data = vec![0; meta_len as usize];
        self.reader.read_exact(&mut block_data)?;
        let message = root_as_message(&block_data[..]).map_err(|err| {
            ArrowError::IoError(format!("Unable to get root as footer: {err:?}"))
        })?;

        // some old test data's footer metadata is not set, so we account for that
        if self.metadata_version != MetadataVersion::V1
            && message.version() != self.metadata_version
        {
            return Err(ArrowError::IoError(
                "Could not read IPC message as metadata versions mismatch".to_string(),
            ));
        }

        match message.header_type() {
            MessageHeader::Schema => Err(ArrowError::IoError(
                "Not expecting a schema when messages are read".to_string(),
            )),
            MessageHeader::RecordBatch => {
                let batch = message.header_as_record_batch().ok_or_else(|| {
                    ArrowError::IoError(
                        "Unable to read IPC message as record batch".to_string(),
                    )
                })?;
                // read the block that makes up the record batch into a buffer
                let mut buf = MutableBuffer::from_len_zeroed(message.bodyLength() as usize);
                self.reader.seek(SeekFrom::Start(
                    block.offset() as u64 + block.metaDataLength() as u64,
                ))?;
                self.reader.read_exact(&mut buf)?;

                read_record_batch(
                    &buf.into(),
                    batch,
                    self.schema(),
                    &self.dictionaries_by_id,
                    self.projection.as_ref().map(|x| x.0.as_ref()),
                    &message.version()

                ).map(Some)
            }
            MessageHeader::NONE => {
                Ok(None)
            }
            t => Err(ArrowError::IoError(format!(
                "Reading types other than record batches not yet supported, unable to read {t:?}"
            ))),
        }
    }
}

impl<R: Read + Seek> Iterator for NoBufReader<R> {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        // get current block
        if self.current_block < self.total_blocks {
            self.maybe_next().transpose()
        } else {
            None
        }
    }
}

impl<R: Read + Seek> RecordBatchReader for NoBufReader<R> {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
