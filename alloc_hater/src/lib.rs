//! A small library for ~~hating on~~ evaluating the correctness of allocators.
#![deny(unsafe_op_in_unsafe_fn)]

use core::{alloc::Layout, mem::MaybeUninit, ptr::NonNull, slice};
use std::cmp;

#[derive(arbitrary::Arbitrary)]
enum AllocatorOpTag {
    Alloc,
    Dealloc,
}

#[derive(Clone, Debug)]
pub enum AllocatorOp {
    Alloc(Layout),
    Dealloc(usize),
}

impl arbitrary::Arbitrary<'_> for AllocatorOp {
    fn arbitrary(u: &mut arbitrary::Unstructured<'_>) -> arbitrary::Result<Self> {
        let tag = AllocatorOpTag::arbitrary(u)?;

        let op = match tag {
            AllocatorOpTag::Alloc => {
                // Select a random bit index and shift to obtain a power of two.
                let align_shift = u8::arbitrary(u)? % usize::BITS as u8;
                let align: usize = 1 << align_shift;
                assert!(align.is_power_of_two());

                // Clamp size to prevent Layout creation errors.
                let size = cmp::min(usize::arbitrary(u)?, isize::MAX as usize - (align - 1));

                let layout = match Layout::from_size_align(size, align) {
                    Ok(l) => l,
                    Err(_) => {
                        panic!("invalid layout params: size=0x{size:X} align=0x{align:X}");
                    }
                };

                AllocatorOp::Alloc(layout)
            }

            AllocatorOpTag::Dealloc => AllocatorOp::Dealloc(usize::arbitrary(u)?),
        };

        Ok(op)
    }
}

pub trait Subject {
    type AllocError;

    /// Allocates a block of memory according to `layout`.
    fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, Self::AllocError>;

    /// Deallocates the block of memory with layout `layout` pointed to by `ptr`.
    ///
    /// # Safety
    ///
    /// `ptr` must denote a block of memory currently allocated by this
    /// allocator, and it must have been allocated with `layout`.
    unsafe fn deallocate(&mut self, ptr: NonNull<u8>, layout: Layout);
}

struct Block {
    // A pointer to the allocated region.
    ptr: NonNull<[u8]>,
    // The original allocation layout.
    layout: Layout,
    // The unique ID of the last operation that wrote to this allocation.
    id: u64,
}

unsafe fn paint(ptr: NonNull<[u8]>, id: u64) {
    let slice: &mut [MaybeUninit<u8>] =
        unsafe { slice::from_raw_parts_mut(ptr.cast().as_ptr(), ptr.len()) };
    let id_bytes = id.to_le_bytes().into_iter().cycle();

    for (byte, value) in slice.iter_mut().zip(id_bytes) {
        byte.write(value);
    }
}

impl Block {
    unsafe fn init(ptr: NonNull<[u8]>, layout: Layout, id: u64) -> Block {
        unsafe { paint(ptr, id) };

        Block { ptr, layout, id }
    }

    unsafe fn paint(&mut self, id: u64) {
        unsafe { paint(self.ptr, id) };
    }

    // Safety: must be initialized
    unsafe fn verify(&self) -> bool {
        let slice: &[u8] = unsafe { self.ptr.as_ref() };
        let id_bytes = self.id.to_le_bytes().into_iter().cycle();

        for (byte, value) in slice.iter().zip(id_bytes) {
            if *byte != value {
                return false;
            }
        }

        true
    }
}

pub struct Evaluator<S: Subject> {
    subject: S,
}

#[derive(Clone, Debug)]
pub struct Failed {
    pub completed: Vec<AllocatorOp>,
    pub failed_op: AllocatorOp,
}

impl<S: Subject> Evaluator<S> {
    pub fn new(subject: S) -> Evaluator<S> {
        Evaluator { subject }
    }

    pub fn evaluate(&mut self, ops: impl IntoIterator<Item = AllocatorOp>) -> Result<(), Failed> {
        let mut completed = Vec::new();
        let mut blocks = Vec::new();

        for (op_id, op) in ops.into_iter().enumerate() {
            match op {
                AllocatorOp::Alloc(layout) => {
                    let ptr = match self.subject.allocate(layout) {
                        Ok(p) => p,
                        Err(_) => continue,
                    };

                    let id: u64 = op_id.try_into().unwrap();
                    let block = unsafe { Block::init(ptr, layout, id) };
                    blocks.push(block);
                }

                AllocatorOp::Dealloc(raw_idx) => {
                    if blocks.is_empty() {
                        continue;
                    }

                    let idx = raw_idx % blocks.len();
                    let mut block = blocks.swap_remove(idx);
                    if unsafe { !block.verify() } {
                        return Err(Failed {
                            completed,
                            failed_op: op,
                        });
                    }

                    let id: u64 = op_id.try_into().unwrap();
                    unsafe {
                        block.paint(id);
                        self.subject.deallocate(block.ptr.cast(), block.layout);
                    }
                }
            }

            completed.push(op);
        }

        Ok(())
    }
}
