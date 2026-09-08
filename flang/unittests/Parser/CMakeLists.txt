set(LLVM_LINK_COMPONENTS
  FrontendOpenACC
  FrontendOpenMP
  Support
  TargetParser
)

add_flang_unittest(FlangParserTests
  CommentRetentionTest.cpp
)

target_link_libraries(FlangParserTests
  PRIVATE
  FortranParser
  FortranSupport
)
