add_triton_library(TritonCPUToLLVM
    FuncOpToLLVM.cpp
    GetProgramIdOpToLLVM.cpp
    MemoryOpToLLVM.cpp
    Pipeline.cpp
    TypeConverter.cpp

    DEPENDS
    TritonCPUConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRVectorToLLVMPass
)
