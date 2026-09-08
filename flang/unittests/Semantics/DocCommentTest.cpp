//===-- flang/unittests/Semantics/DocCommentTest.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SemanticsContext::GetDocComment() binds a retained '!>' or '!!' comment to
// the Symbol it documents, caching the result on the Symbol.  The binding rule
// is what makes it useful: a doc comment documents the declaration it sits
// immediately above, so an unrelated doc comment elsewhere in the file must not
// attach itself to every symbol that happens to follow it.
//
//===----------------------------------------------------------------------===//

#include "flang/Semantics/semantics.h"
#include "flang/Parser/options.h"
#include "flang/Parser/parsing.h"
#include "flang/Parser/provenance.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/symbol.h"
#include "flang/Support/Fortran-features.h"
#include "flang/Support/LangOptions.h"
#include "flang/Support/default-kinds.h"
#include "gtest/gtest.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <string>

namespace Fortran::semantics {
namespace {

// Parses and semantically analyzes a snippet, then hands back symbols by name.
//
// Everything here has to stay alive together: a bound SourceComment points into
// the CookedSource's comment vector, which points into the SourceFile buffer,
// and the Symbol caches the resulting pointer.
class AnalyzedSource {
public:
  explicit AnalyzedSource(llvm::StringRef source) {
    llvm::SmallString<128> path;
    int fd{-1};
    std::error_code ec{
        llvm::sys::fs::createTemporaryFile("flang-doc-comment", "f90", fd, path)};
    EXPECT_FALSE(ec) << "could not create temporary file: " << ec.message();
    {
      llvm::raw_fd_ostream os{fd, /*shouldClose=*/true};
      os << source;
    }
    path_ = std::string{path};

    parser::Options options;
    options.isFixedForm = false;
    prescanned_ = parsing_.Prescan(path_, options) != nullptr;
    if (!prescanned_) {
      return;
    }
    parsing_.Parse(llvm::nulls());
    if (!parsing_.parseTree()) {
      return;
    }
    Semantics semantics{context_, *parsing_.parseTree()};
    semantics.Perform();
    analyzed_ = !context_.AnyFatalError();
  }

  ~AnalyzedSource() { llvm::sys::fs::remove(path_); }

  bool analyzed() const { return analyzed_; }
  SemanticsContext &context() { return context_; }

  // Resolves a declared name to its Symbol by locating the name in the cooked
  // stream, which is the same route a tool consuming the parse tree would take.
  Symbol *findSymbol(llvm::StringRef name) {
    parser::CharBlock all{parsing_.cooked().AsCharBlock()};
    llvm::StringRef text{all.begin(), all.size()};
    std::size_t pos{text.find(name)};
    if (pos == llvm::StringRef::npos) {
      return nullptr;
    }
    parser::CharBlock cb{all.begin() + pos, name.size()};
    return context_.FindScope(cb).FindSymbol(cb);
  }

private:
  std::string path_;
  parser::AllSources allSources_;
  parser::AllCookedSources allCooked_{allSources_};
  parser::Parsing parsing_{allCooked_};
  common::IntrinsicTypeDefaultKinds defaultKinds_;
  common::LanguageFeatureControl features_;
  common::LangOptions langOpts_;
  SemanticsContext context_{defaultKinds_, features_, langOpts_, allCooked_};
  bool prescanned_{false};
  bool analyzed_{false};
};

// The ordinary case: a doc comment directly above a declaration binds to it.
TEST(DocComment, BindsCommentDirectlyAbove) {
  AnalyzedSource src{R"(program p
  !> documents alpha
  integer :: alpha
end program p
)"};
  ASSERT_TRUE(src.analyzed());

  Symbol *alpha{src.findSymbol("alpha")};
  ASSERT_NE(alpha, nullptr);

  const parser::SourceComment *doc{src.context().GetDocComment(*alpha)};
  ASSERT_NE(doc, nullptr);
  EXPECT_NE(std::string(doc->source.begin(), doc->source.size())
                .find("documents alpha"),
      std::string::npos);
}

// A plain '!' comment is not documentation and must not bind.
TEST(DocComment, IgnoresNonDocComments) {
  AnalyzedSource src{R"(program p
  ! just an ordinary comment
  integer :: alpha
end program p
)"};
  ASSERT_TRUE(src.analyzed());

  Symbol *alpha{src.findSymbol("alpha")};
  ASSERT_NE(alpha, nullptr);
  EXPECT_EQ(src.context().GetDocComment(*alpha), nullptr);
}

// The binding must be adjacent.  A doc comment that documents an earlier
// declaration must not leak onto a later, undocumented one -- otherwise every
// symbol following the first doc comment in a file inherits it.
TEST(DocComment, DoesNotBindDocCommentOfAnEarlierDeclaration) {
  AnalyzedSource src{R"(program p
  !> documents alpha
  integer :: alpha
  integer :: beta
end program p
)"};
  ASSERT_TRUE(src.analyzed());

  Symbol *beta{src.findSymbol("beta")};
  ASSERT_NE(beta, nullptr);
  EXPECT_EQ(src.context().GetDocComment(*beta), nullptr);
}

// A blank line breaks the association even with nothing else between.
TEST(DocComment, DoesNotBindAcrossABlankLine) {
  AnalyzedSource src{R"(program p
  !> floating doc comment

  integer :: alpha
end program p
)"};
  ASSERT_TRUE(src.analyzed());

  Symbol *alpha{src.findSymbol("alpha")};
  ASSERT_NE(alpha, nullptr);
  EXPECT_EQ(src.context().GetDocComment(*alpha), nullptr);
}

// A comment trailing a statement does not turn that statement's line into a
// comment line, so it cannot bridge a doc comment down to the next declaration.
TEST(DocComment, TrailingCommentDoesNotBridgeToTheNextDeclaration) {
  AnalyzedSource src{R"(program p
  !> documents alpha
  integer :: alpha ! trailing comment
  integer :: beta
end program p
)"};
  ASSERT_TRUE(src.analyzed());

  Symbol *beta{src.findSymbol("beta")};
  ASSERT_NE(beta, nullptr);
  EXPECT_EQ(src.context().GetDocComment(*beta), nullptr);
}

// Conversely, an ordinary whole-line comment between the doc comment and the
// declaration keeps the run unbroken, so the binding survives.
TEST(DocComment, OrdinaryCommentLineDoesNotBreakTheRun) {
  AnalyzedSource src{R"(program p
  !> documents alpha
  ! an ordinary note
  integer :: alpha
end program p
)"};
  ASSERT_TRUE(src.analyzed());

  Symbol *alpha{src.findSymbol("alpha")};
  ASSERT_NE(alpha, nullptr);

  const parser::SourceComment *doc{src.context().GetDocComment(*alpha)};
  ASSERT_NE(doc, nullptr);
  EXPECT_NE(std::string(doc->source.begin(), doc->source.size())
                .find("documents alpha"),
      std::string::npos);
}

// Each declaration keeps its own doc comment when several are stacked.
TEST(DocComment, BindsTheNearestDocCommentPerDeclaration) {
  AnalyzedSource src{R"(program p
  !> documents alpha
  integer :: alpha
  !> documents beta
  integer :: beta
end program p
)"};
  ASSERT_TRUE(src.analyzed());

  Symbol *alpha{src.findSymbol("alpha")};
  Symbol *beta{src.findSymbol("beta")};
  ASSERT_NE(alpha, nullptr);
  ASSERT_NE(beta, nullptr);

  const parser::SourceComment *alphaDoc{src.context().GetDocComment(*alpha)};
  const parser::SourceComment *betaDoc{src.context().GetDocComment(*beta)};
  ASSERT_NE(alphaDoc, nullptr);
  ASSERT_NE(betaDoc, nullptr);
  EXPECT_NE(std::string(alphaDoc->source.begin(), alphaDoc->source.size())
                .find("documents alpha"),
      std::string::npos);
  EXPECT_NE(std::string(betaDoc->source.begin(), betaDoc->source.size())
                .find("documents beta"),
      std::string::npos);
}

// The result is cached on the Symbol, so repeated queries are stable.
TEST(DocComment, CachesBindingOnTheSymbol) {
  AnalyzedSource src{R"(program p
  !> documents alpha
  integer :: alpha
end program p
)"};
  ASSERT_TRUE(src.analyzed());

  Symbol *alpha{src.findSymbol("alpha")};
  ASSERT_NE(alpha, nullptr);

  const parser::SourceComment *first{src.context().GetDocComment(*alpha)};
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(alpha->docComment(), first);
  EXPECT_EQ(src.context().GetDocComment(*alpha), first);
}

} // namespace
} // namespace Fortran::semantics
