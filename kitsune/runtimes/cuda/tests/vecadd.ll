;
;
;
target triple = "nvptx64-unknown-cuda"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

define i32 @addElements(i32 %a, i32 %b) {
entry:
  %add = add nsw i32 %a, %b
  ret i32 %add
}

define void @vecadd(i32* %c, i32* %a, i32* %b) {
entry:
  ; do some standard cuda stuff:
  ; 
  ;   CUDA: int tid = blockIdx.x * blockDim.x + threadIdx.x
  ;
  ; note that the names here refer directly to PTX special registers 
  ; (ntid, ctaid, tid).  you can find details on these in the PTX 
  ; reference; as they are not really well defined in the NVVM 
  ; documentation. 
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %mul = mul i32 %0, %1
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %tid = add i32 %mul, %2
  ; 
  %idxprom = sext i32 %tid to i64
  %3 = getelementptr inbounds i32, i32* %a, i64 %idxprom
  %4 = load i32, i32* %3, align 4
  %5 = getelementptr inbounds i32, i32* %b, i64 %idxprom
  %6 = load i32, i32* %5, align 4
  %esum = call i32 @addElements(i32 %4, i32 %6)
  %indx = getelementptr inbounds i32, i32* %c, i64 %idxprom
  store i32 %esum, i32* %indx, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone

!nvvm.annotations = !{!1}
!1 = !{void (i32*, i32*, i32*)* @vecadd, !"kernel", i32 1}
