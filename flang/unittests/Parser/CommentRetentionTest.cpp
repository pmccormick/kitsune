//===-- flang/unittests/Parser/CommentRetentionTest.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Comments are normally dropped on the floor as the prescanner builds the
// cooked character stream.  The comment-retention support captures them on the
// way out into CookedSource, where they stay addressable for the lifetime of
// the compilation, and exposes them through CookedSource::Comments() plus the
// position-based lookup helpers on AllCookedSources.
//
// These tests pin down the two properties that support rests on: that comments
// are captured at all, and that capturing them does not perturb the cooked
// stream the parser actually sees.
//
//===----------------------------------------------------------------------===//

#include "flang/Parser/parsing.h"
#include "flang/Parser/options.h"
#include "flang/Parser/provenance.h"
#include "gtest/gtest.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

namespace Fortran::parser {
namespace {

// Prescans a snippet from a temporary file and owns the resulting state.
//
// Prescanning needs a real path, and every CharBlock in a SourceComment points
// directly into the SourceFile's buffer rather than owning its text, so the
// AllSources/AllCookedSources pair must outlive any inspection of the comments.
// Holding all three together here is what keeps those pointers valid.
class PrescannedSource {
public:
  explicit PrescannedSource(llvm::StringRef source, bool isFixedForm = false) {
    llvm::SmallString<128> path;
    int fd{-1};
    std::error_code ec{llvm::sys::fs::createTemporaryFile(
        "flang-comment-retention", isFixedForm ? "f" : "f90", fd, path)};
    EXPECT_FALSE(ec) << "could not create temporary file: " << ec.message();
    {
      llvm::raw_fd_ostream os{fd, /*shouldClose=*/true};
      os << source;
    }
    path_ = std::string{path};

    Options options;
    options.isFixedForm = isFixedForm;
    sourceFile_ = parsing_.Prescan(path_, options);
  }

  ~PrescannedSource() { llvm::sys::fs::remove(path_); }

  bool prescanSucceeded() const { return sourceFile_ != nullptr; }
  const CookedSource &cooked() const { return parsing_.cooked(); }
  AllCookedSources &allCooked() { return allCooked_; }

  const std::vector<SourceComment> &comments() const {
    return parsing_.cooked().Comments();
  }

  // Locates a token in the cooked stream and returns it as a CharBlock, which
  // is how a consumer hands a parse-tree node's source position to the lookup
  // helpers below.
  std::optional<CharBlock> findInCooked(llvm::StringRef token) const {
    CharBlock all{parsing_.cooked().AsCharBlock()};
    llvm::StringRef text{all.begin(), all.size()};
    std::size_t pos{text.find(token)};
    if (pos == llvm::StringRef::npos) {
      return std::nullopt;
    }
    return CharBlock{all.begin() + pos, token.size()};
  }

  // The retained comment texts, in capture order, as plain strings.
  std::vector<std::string> commentTexts() const {
    std::vector<std::string> texts;
    for (const SourceComment &c : comments()) {
      texts.emplace_back(c.source.begin(), c.source.size());
    }
    return texts;
  }

private:
  std::string path_;
  AllSources allSources_;
  AllCookedSources allCooked_{allSources_};
  Parsing parsing_{allCooked_};
  const SourceFile *sourceFile_{nullptr};
};

// Whole-line comments are captured from the start of the line, so the retained
// text includes any leading indentation.
TEST(CommentRetention, CapturesWholeLineComments) {
  PrescannedSource src{R"(! leading comment
program p
  ! indented comment
end program p
)"};
  ASSERT_TRUE(src.prescanSucceeded());

  std::vector<std::string> texts{src.commentTexts()};
  ASSERT_EQ(texts.size(), 2u);
  EXPECT_EQ(texts[0], "! leading comment");
  EXPECT_EQ(texts[1], "  ! indented comment");
}

// Trailing comments are captured from the '!' itself, not from the start of
// the line, so the statement text ahead of them is not included.
TEST(CommentRetention, CapturesTrailingComments) {
  PrescannedSource src{R"(program p
  integer :: i ! trailing comment
end program p
)"};
  ASSERT_TRUE(src.prescanSucceeded());

  std::vector<std::string> texts{src.commentTexts()};
  ASSERT_EQ(texts.size(), 1u);
  EXPECT_EQ(texts[0], "! trailing comment");
}

// '!>' and '!!' mark a comment as documentation; a plain '!' does not.  The
// classifier looks at the first '!' in the retained text, so indentation ahead
// of the marker is irrelevant.
TEST(CommentRetention, ClassifiesDocComments) {
  PrescannedSource src{R"(!> angle doc comment
!! bang doc comment
! ordinary comment
  !> indented doc comment
program p
end program p
)"};
  ASSERT_TRUE(src.prescanSucceeded());

  const std::vector<SourceComment> &comments{src.comments()};
  ASSERT_EQ(comments.size(), 4u);
  EXPECT_TRUE(comments[0].isDocComment);
  EXPECT_TRUE(comments[1].isDocComment);
  EXPECT_FALSE(comments[2].isDocComment);
  EXPECT_TRUE(comments[3].isDocComment);
}

// A '!' inside a character literal is not a comment and must not be captured,
// nor may it truncate the literal in the cooked stream.
TEST(CommentRetention, IgnoresBangInsideCharacterLiteral) {
  PrescannedSource src{R"(program p
  character(*), parameter :: s = 'not ! a comment'
end program p
)"};
  ASSERT_TRUE(src.prescanSucceeded());

  EXPECT_TRUE(src.comments().empty());
}

// Retention must be observationally invisible to the parser: the cooked stream
// is what the parser consumes, and no comment text may survive into it.
TEST(CommentRetention, CommentsAreAbsentFromCookedStream) {
  PrescannedSource src{R"(! secret one
program p
  integer :: i ! secret two
end program p
)"};
  ASSERT_TRUE(src.prescanSucceeded());

  ASSERT_EQ(src.comments().size(), 2u);
  llvm::StringRef cooked{
      src.cooked().AsCharBlock().begin(), src.cooked().AsCharBlock().size()};
  EXPECT_EQ(cooked.find("secret"), llvm::StringRef::npos);
}

// Continuation lines may be interrupted by comment lines; those are skipped
// through a separate path in the prescanner and must still be captured.
TEST(CommentRetention, CapturesCommentsBetweenContinuationLines) {
  PrescannedSource src{R"(program p
  integer :: i = 1 + &
    ! comment between continuations
    2
end program p
)"};
  ASSERT_TRUE(src.prescanSucceeded());

  std::vector<std::string> texts{src.commentTexts()};
  ASSERT_EQ(texts.size(), 1u);
  EXPECT_NE(texts[0].find("comment between continuations"), std::string::npos);
}

// A source with no comments must retain nothing, so that consumers can
// distinguish "no comments" from "retention did not run".
TEST(CommentRetention, RetainsNothingWhenThereAreNoComments) {
  PrescannedSource src{R"(program p
  integer :: i
end program p
)"};
  ASSERT_TRUE(src.prescanSucceeded());

  EXPECT_TRUE(src.comments().empty());
}

// GetPrecedingComments returns only the run of comment lines directly above an
// entity.  The contiguity rule is what makes a doc-comment block attributable:
// an unrelated comment further up, separated by a blank line, must not be
// swept in.
TEST(CommentRetention, GetPrecedingCommentsReturnsContiguousBlockOnly) {
  PrescannedSource src{R"(program p
! unrelated comment

!> first doc line
!> second doc line
  integer :: myvariable
end program p
)"};
  ASSERT_TRUE(src.prescanSucceeded());

  std::optional<CharBlock> entity{src.findInCooked("myvariable")};
  ASSERT_TRUE(entity.has_value());

  std::vector<SourceComment> preceding{
      src.allCooked().GetPrecedingComments(*entity)};
  ASSERT_EQ(preceding.size(), 2u);
  EXPECT_EQ(std::string(preceding[0].source.begin(), preceding[0].source.size()),
      "!> first doc line");
  EXPECT_EQ(std::string(preceding[1].source.begin(), preceding[1].source.size()),
      "!> second doc line");
}

// An entity with nothing above it has no preceding comments.
TEST(CommentRetention, GetPrecedingCommentsIsEmptyWhenNoneAdjoin) {
  PrescannedSource src{R"(! comment far above

program p
  integer :: lonely
end program p
)"};
  ASSERT_TRUE(src.prescanSucceeded());

  std::optional<CharBlock> entity{src.findInCooked("lonely")};
  ASSERT_TRUE(entity.has_value());
  EXPECT_TRUE(src.allCooked().GetPrecedingComments(*entity).empty());
}

// GetFollowingComment looks at the line after the entity ends.
TEST(CommentRetention, GetFollowingCommentFindsCommentOnNextLine) {
  PrescannedSource src{R"(program p
  integer :: beta
  ! following comment
end program p
)"};
  ASSERT_TRUE(src.prescanSucceeded());

  std::optional<CharBlock> entity{src.findInCooked("beta")};
  ASSERT_TRUE(entity.has_value());

  std::optional<SourceComment> following{
      src.allCooked().GetFollowingComment(*entity)};
  ASSERT_TRUE(following.has_value());
  EXPECT_NE(std::string(following->source.begin(), following->source.size())
                .find("following comment"),
      std::string::npos);
}

// A trailing comment sits on the entity's own line, not the following one, so
// it is deliberately not what GetFollowingComment reports.
TEST(CommentRetention, GetFollowingCommentIgnoresSameLineTrailingComment) {
  PrescannedSource src{R"(program p
  integer :: gamma ! trailing comment
  integer :: delta
end program p
)"};
  ASSERT_TRUE(src.prescanSucceeded());

  std::optional<CharBlock> entity{src.findInCooked("gamma")};
  ASSERT_TRUE(entity.has_value());
  EXPECT_FALSE(src.allCooked().GetFollowingComment(*entity).has_value());
}

// GetCommentsInRange spans the lines between two provenance points inclusive,
// so comments outside that span are excluded on both sides.
TEST(CommentRetention, GetCommentsInRangeIsBoundedByBothEndpoints) {
  PrescannedSource src{R"(program p
  ! before the range
  integer :: alpha
  ! inside the range
  integer :: omega
  ! after the range
end program p
)"};
  ASSERT_TRUE(src.prescanSucceeded());

  std::optional<CharBlock> first{src.findInCooked("alpha")};
  std::optional<CharBlock> last{src.findInCooked("omega")};
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(last.has_value());

  std::optional<ProvenanceRange> firstRange{
      src.allCooked().GetProvenanceRange(*first)};
  std::optional<ProvenanceRange> lastRange{
      src.allCooked().GetProvenanceRange(*last)};
  ASSERT_TRUE(firstRange.has_value());
  ASSERT_TRUE(lastRange.has_value());

  std::vector<SourceComment> inRange{src.allCooked().GetCommentsInRange(
      firstRange->start(), lastRange->start())};
  ASSERT_EQ(inRange.size(), 1u);
  EXPECT_NE(std::string(inRange[0].source.begin(), inRange[0].source.size())
                .find("inside the range"),
      std::string::npos);
}

} // namespace
} // namespace Fortran::parser
