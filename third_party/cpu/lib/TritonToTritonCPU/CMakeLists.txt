add_triton_library(TritonToTritonCPU
    ConvertElementwiseOps.cpp
    ConvertMemoryOps.cpp
    Pipeline.cpp

    DEPENDS
    TritonToTritonCPUPassIncGen

    LINK_LIBS PUBLIC
    TritonCPUIR
    MLIRVectorDialect
)
