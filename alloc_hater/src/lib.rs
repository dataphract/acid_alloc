//! A small library for ~~hating on~~ evaluating the correctness of allocators.
#![deny(unsafe_op_in_unsafe_fn)]

use core::{alloc::Layout, mem::MaybeUninit, ptr::NonNull, slice};

/// A wrapper around `Layout` which implements `Arbitrary`.
#[derive(Clone, Debug)]
pub struct ArbLayout(pub Layout);

impl arbitrary::Arbitrary<'_> for ArbLayout {
    fn arbitrary(u: &mut arbitrary::Unstructured<'_>) -> arbitrary::Result<Self> {
        // Select a random bit index and shift to obtain a power of two.
        let align_shift = u8::arbitrary(u)? % (usize::BITS - 1) as u8;

        let align: usize = 1 << align_shift;
        assert!(align.is_power_of_two());

        let size = usize::arbitrary(u)? % (isize::MAX as usize - (align - 1));

        let layout = match Layout::from_size_align(size, align) {
            Ok(l) => l,
            Err(_) => {
                panic!("invalid layout params: size=0x{size:X} align=0x{align:X}");
            }
        };

        Ok(ArbLayout(layout))
    }
}

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
            AllocatorOpTag::Alloc => AllocatorOp::Alloc(ArbLayout::arbitrary(u)?.0),
            AllocatorOpTag::Dealloc => AllocatorOp::Dealloc(usize::arbitrary(u)?),
        };

        Ok(op)
    }
}

pub trait Subject {
    type Op: for<'a> arbitrary::Arbitrary<'a>;
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

    fn handle_custom_op(&mut self, op: Self::Op) {
        // To silence the unused variable warning.
        drop(op);
    }
}

/// A list of allocated blocks.
#[derive(Default)]
pub struct Blocks {
    blocks: Vec<Block>,
}

impl Blocks {
    pub fn new() -> Blocks {
        Blocks { blocks: Vec::new() }
    }

    pub fn push(&mut self, block: Block) {
        self.blocks.push(block);
    }

    pub fn remove_modulo(&mut self, idx: usize) -> Option<Block> {
        let len = self.blocks.len();
        (len != 0).then(|| self.blocks.swap_remove(idx % len))
    }
}

impl IntoIterator for Blocks {
    type Item = Block;

    type IntoIter = std::vec::IntoIter<Block>;

    fn into_iter(self) -> Self::IntoIter {
        self.blocks.into_iter()
    }
}

/// An allocated block of memory.
pub struct Block {
    // A pointer to the allocated region.
    ptr: NonNull<[u8]>,
    // The original allocation layout.
    layout: Layout,
    // The unique ID of the last operation that wrote to this allocation.
    id: u64,
}

unsafe fn slice_ptr_to_uninit_slice_mut<'a>(ptr: NonNull<[u8]>) -> &'a mut [MaybeUninit<u8>] {
    unsafe { slice::from_raw_parts_mut(ptr.cast().as_ptr(), ptr.len()) }
}

unsafe fn paint(slice: &mut [MaybeUninit<u8>], id: u64) {
    let id_bytes = id.to_le_bytes().into_iter().cycle();

    for (byte, value) in slice.iter_mut().zip(id_bytes) {
        byte.write(value);
    }
}

impl Block {
    /// Creates a block from `ptr` and paints it according to `id`.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `ptr` must be valid for reads and writes for `ptr.len()` bytes.
    /// - `ptr` must have been allocated according to `layout`.
    /// - No references to the memory at `ptr` may exist when this function is called.
    /// - No accesses to the memory at `ptr` may be made except by way of the returned `Block` said
    ///  `Block` is dropped.
    pub unsafe fn init(ptr: NonNull<[u8]>, layout: Layout, id: u64) -> Block {
        let mut b = Block { ptr, layout, id };
        b.paint(id);
        b
    }

    /// Returns the `Block`'s memory as a slice of uninitialized bytes.
    pub fn as_uninit_slice(&self) -> &[MaybeUninit<u8>] {
        // SAFETY: self is immutably borrowed, so only immutable references to
        // the slice can exist
        unsafe { &*slice_ptr_to_uninit_slice_mut(self.ptr) }
    }

    /// Returns the `Block`'s memory as a mutable slice of uninitialized bytes.
    pub fn as_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        // SAFETY: self is mutably borrowed, so no other references to the
        // slice can exist
        unsafe { slice_ptr_to_uninit_slice_mut(self.ptr) }
    }

    pub fn into_raw_parts(self) -> (NonNull<[u8]>, Layout) {
        (self.ptr, self.layout)
    }

    /// "Paints" the memory contained by `self` with the value of `id`.
    pub fn paint(&mut self, id: u64) {
        unsafe { paint(self.as_uninit_slice_mut(), id) };
    }

    /// Verifies that the memory contained by `self` has not been overwritten.
    pub fn verify(&self) -> bool {
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

    pub fn evaluate<I>(&mut self, ops: I) -> Result<(), Failed>
    where
        I: for<'a> IntoIterator<Item = AllocatorOp>,
    {
        let mut completed = Vec::new();
        let mut blocks = Blocks::new();

        for (op_id, op) in ops.into_iter().enumerate() {
            let op_id: u64 = op_id.try_into().unwrap();
            match op {
                AllocatorOp::Alloc(layout) => {
                    let ptr = match self.subject.allocate(layout) {
                        Ok(p) => p,
                        Err(_) => continue,
                    };

                    let block = unsafe { Block::init(ptr, layout, op_id) };
                    blocks.push(block);
                }

                AllocatorOp::Dealloc(raw_idx) => {
                    let mut block = match blocks.remove_modulo(raw_idx) {
                        Some(b) => b,
                        None => continue,
                    };

                    if !block.verify() {
                        return Err(Failed {
                            completed,
                            failed_op: op,
                        });
                    }

                    unsafe {
                        block.paint(op_id);
                        self.subject.deallocate(block.ptr.cast(), block.layout);
                    }
                }
            }

            completed.push(op);
        }

        for block in blocks {
            // TODO: verify these blocks
            unsafe { self.subject.deallocate(block.ptr.cast(), block.layout) };
        }

        Ok(())
    }
}
