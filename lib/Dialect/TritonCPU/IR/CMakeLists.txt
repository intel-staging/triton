add_triton_library(TritonCPUIR
  Dialect.cpp

  DEPENDS
  TritonCPUTableGen
  TritonCPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonIR
)
