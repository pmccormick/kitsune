# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o
# RUN: ld.lld -r %t1.o -o %t
# RUN: llvm-readobj --symbols -r %t | FileCheck %s

# WARN: warning: -d, -dc, -dp, and --[no-]define-common will be removed. See https://github.com/llvm/llvm-project/issues/53660

# CHECK:        Symbol {
# CHECK:          Name: common
# CHECK-NEXT:     Value: 0x4
# CHECK-NEXT:     Size: 4
# CHECK-NEXT:     Binding: Global
# CHECK-NEXT:     Type: Object
# CHECK-NEXT:     Other: 0
# CHECK-NEXT:     Section: Common (0xFFF2)
# CHECK-NEXT:   }

.comm common,4,4
