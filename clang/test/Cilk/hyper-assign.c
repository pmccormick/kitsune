// RUN: %clang_cc1 %s -x c -fopencilk -verify -S -emit-llvm -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang_cc1 %s -x c++ -fopencilk -verify -S -emit-llvm -disable-llvm-passes -o - | FileCheck %s
// expected-no-diagnostics

extern long _Hyperobject x, _Hyperobject y;

long chain_assign()
{
  // CHECK: %[[Y1PTR:.+]] = call ptr @llvm.hyper.lookup(ptr @y)
  // CHECK: %[[Y1VAL:.+]] = load i64, ptr %[[Y1PTR]]
  // CHECK: call ptr @llvm.hyper.lookup(ptr @x)
  // CHECK: store i64 %[[Y1VAL]]
  // CHECK: call ptr @llvm.hyper.lookup(ptr @y)
  // CHECK: call ptr @llvm.hyper.lookup(ptr @x)
  return x = y = x = y;
}

long simple_assign(long val)
{
  // CHECK: call ptr @llvm.hyper.lookup(ptr @x)
  // CHECK-NOT: call ptr @llvm.hyper.lookup(ptr @x)
  // CHECK: store i64
  return x = val;
}

long subtract()
{
  // The order is not fixed here.
  // CHECK: {{.+}} = call ptr @llvm.hyper.lookup(ptr @y)
  // CHECK: load i64
  // CHECK: add nsw i64 %[[Y:.+]], 1
  // CHECK: store i64
  // CHECK: call ptr @llvm.hyper.lookup(ptr @x)
  // CHECK: load i64
  // CHECK: sub nsw
  // CHECK: store i64
  return x -= y++;
}
