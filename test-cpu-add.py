import triton
import triton.language as tl
import torch

def test_01():
    @triton.jit
    def kernel(in1_ptr, in2_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        id = tl.program_id(0)
        offs = id * BLOCK_SIZE
        in1_block_ptr = tl.make_block_ptr(base=in1_ptr, shape=(BLOCK_SIZE, ), strides=(1, ),
                                        offsets=(offs, ), block_shape=(BLOCK_SIZE, ),
                                        order=(0, ))
        in2_block_ptr = tl.make_block_ptr(base=in2_ptr, shape=(BLOCK_SIZE, ), strides=(1, ),
                                        offsets=(offs, ), block_shape=(BLOCK_SIZE, ),
                                        order=(0, ))
        out_block_ptr = tl.make_block_ptr(base=out_ptr, shape=(BLOCK_SIZE, ), strides=(1, ),
                                        offsets=(offs, ), block_shape=(BLOCK_SIZE, ),
                                        order=(0, ))
        val1 = tl.load(in1_block_ptr, boundary_check=())
        val2 = tl.load(in2_block_ptr, boundary_check=())
        val = val1 + val2
        tl.store(out_block_ptr, val, boundary_check=())

    total_size = 1024
    block_size = 128
    x = torch.randn((total_size,), device='cpu')
    y = torch.randn((total_size,), device='cpu')
    z = torch.zeros((total_size,), device='cpu')
    assert total_size % block_size == 0
    kernel[(total_size // block_size,)](x, y, z, BLOCK_SIZE=block_size)
    assert torch.all(x + y == z)

def test_02():
    @triton.jit
    def kernel(in1_ptr, in2_ptr, out_ptr, BLOCK_SIZE: tl.constexpr, TILE_SIZE: tl.constexpr):
        id = tl.program_id(0)
        offs = id * BLOCK_SIZE
        tl.static_assert(BLOCK_SIZE % TILE_SIZE == 0, "block size is not tile aligned")
        in1_block_ptr = tl.make_block_ptr(base=in1_ptr, shape=(BLOCK_SIZE, ), strides=(1, ),
                                        offsets=(offs, ), block_shape=(TILE_SIZE, ),
                                        order=(0, ))
        in2_block_ptr = tl.make_block_ptr(base=in2_ptr, shape=(BLOCK_SIZE, ), strides=(1, ),
                                        offsets=(offs, ), block_shape=(TILE_SIZE, ),
                                        order=(0, ))
        out_block_ptr = tl.make_block_ptr(base=out_ptr, shape=(BLOCK_SIZE, ), strides=(1, ),
                                        offsets=(offs, ), block_shape=(TILE_SIZE, ),
                                        order=(0, ))
        for i in range(0, tl.cdiv(BLOCK_SIZE, TILE_SIZE)):
            val1 = tl.load(in1_block_ptr, boundary_check=())
            val2 = tl.load(in2_block_ptr, boundary_check=())
            val = val1 + val2
            tl.store(out_block_ptr, val, boundary_check=())

            in1_block_ptr = tl.advance(in1_block_ptr, (TILE_SIZE,))
            in2_block_ptr = tl.advance(in2_block_ptr, (TILE_SIZE,))
            out_block_ptr = tl.advance(out_block_ptr, (TILE_SIZE,))

    total_size = 1024
    block_size = 128
    tile_size = 16
    x = torch.randn((total_size,), device='cpu')
    y = torch.randn((total_size,), device='cpu')
    z = torch.zeros((total_size,), device='cpu')
    assert total_size % block_size == 0
    assert block_size % tile_size == 0
    kernel[(total_size // block_size,)](x, y, z, BLOCK_SIZE=block_size, TILE_SIZE=tile_size)
    assert torch.all(x + y == z)

#test_01()
test_02()
