-- The CXX compiler identification is GNU 11.4.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done (0.2s)
-- Generating done (0.0s)
-- Build files have been written to: /content/alien-intelligence/build
[  6%] Building CXX object CMakeFiles/ai2_core.dir/src/lssc.cpp.o
[  6%] Building CXX object CMakeFiles/ai2_core.dir/src/slie.cpp.o
[ 10%] Building CXX object CMakeFiles/ai2_core.dir/src/stre.cpp.o
[ 13%] Building CXX object CMakeFiles/ai2_core.dir/src/uq.cpp.o
[ 17%] Building CXX object CMakeFiles/ai2_core.dir/src/ataa.cpp.o
[ 20%] Building CXX object CMakeFiles/ai2_core.dir/src/ssog.cpp.o
[ 24%] Linking CXX static library libai2_core.a
[ 24%] Built target ai2_core
[ 27%] Building CXX object CMakeFiles/test_slie.dir/tests/test_slie.cpp.o
[ 31%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/tokenizer.cpp.o
[ 34%] Linking CXX executable test_slie
[ 34%] Built target test_slie
[ 37%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/dataloader.cpp.o
[ 41%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/model.cpp.o
[ 44%] Building CXX object CMakeFiles/test_lssc.dir/tests/test_lssc.cpp.o
[ 48%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/optimizer.cpp.o
[ 51%] Linking CXX executable test_lssc
[ 51%] Built target test_lssc
[ 55%] Building CXX object CMakeFiles/test_stre.dir/tests/test_stre.cpp.o
[ 58%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/trainer.cpp.o
[ 62%] Linking CXX executable test_stre
[ 62%] Built target test_stre
[ 65%] Building CXX object CMakeFiles/test_uq.dir/tests/test_uq.cpp.o
[ 68%] Linking CXX executable test_uq
[ 68%] Built target test_uq
[ 72%] Linking CXX static library libai2_train_lib.a
[ 75%] Building CXX object CMakeFiles/test_ataa.dir/tests/test_ataa.cpp.o
[ 75%] Built target ai2_train_lib
[ 79%] Building CXX object CMakeFiles/test_ssog.dir/tests/test_ssog.cpp.o
[ 82%] Linking CXX executable test_ataa
[ 86%] Linking CXX executable test_ssog
[ 86%] Built target test_ataa
[ 89%] Building CXX object CMakeFiles/test_math.dir/tests/test_math.cpp.o
[ 89%] Built target test_ssog
[ 93%] Building CXX object CMakeFiles/ai2_train.dir/src/main_train.cpp.o
[ 96%] Linking CXX executable ai2_train
[ 96%] Built target ai2_train
[100%] Linking CXX executable test_math
[100%] Built target test_math
=== Alien Intelligence (AI²) Comprehensive Mathematical Tests ===

[Types/Utilities Mathematical Tests]

[SLIE Mathematical Tests]
    [debug] token0_1 L2^2=0.0970324
    [debug] same-token L2 diff=0 max_diff[0]=0 a[i]=0.00813751 b[i]=0.00813751
    [debug] pos_diff L2^2=0.003907
    [debug] e_pos0[0..7]=0.0248757,-0.0077751,-0.00222176,0.00908489,-0.00269023,0.0119504,0.0620854,0.036686, e_pos1[0..7]=0.0246679,0.00313438,0.03809,0.0443968,-0.0176159,0.0328002,0.0519693,0.0241902,
    [debug] sketch_features size=16 expected=16

[LSSC Mathematical Tests]

[STRE Mathematical Tests]
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=16 first_feat[0]=1 result[0][0]=0
    [debug] n_nodes=1 lap[0][0]=0 total=0
    [debug stre] sheaf_laplacian_all: n=4 nodes_.size()=4 rest_maps_.size()=4 d_node=16 first_feat[0]=-0.641406 result[0][0]=0

[UQ Mathematical Tests]

[ATAA Mathematical Tests]

[SSOG Mathematical Tests]
    [debug ssog] base_dist call=1 mixture[0]=0.352496 W_out[0][0]=-0.160737 b_out[0]=0 logit[0]=-0.143099 logit[1]=0.110007 max-min=0.524427 n_vocab=50 d_model=16
    [debug ssog] base_dist call=2 mixture[0]=0.352496 W_out[0][0]=-0.160737 b_out[0]=0 logit[0]=-0.143099 logit[1]=0.110007 max-min=0.524427 n_vocab=50 d_model=16
    [debug ssog] base_dist call=3 mixture[0]=0.352496 W_out[0][0]=-0.160737 b_out[0]=0 logit[0]=-0.143099 logit[1]=0.110007 max-min=0.524427 n_vocab=50 d_model=16
    [debug ssog] base_dist call=4 mixture[0]=0.352496 W_out[0][0]=-0.160737 b_out[0]=0 logit[0]=-0.143099 logit[1]=0.110007 max-min=0.524427 n_vocab=50 d_model=16

[Optimizer Mathematical Tests]

[Gradient Mathematical Tests]
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.173153 result[0][0]=0
    [debug ssog] base_dist call=5 mixture[0]=-0.23938 W_out[0][0]=0.371773 b_out[0]=0 logit[0]=-0.360955 logit[1]=0.0803731 max-min=1.08009 n_vocab=16 d_model=8
    [debug ssog] base_dist call=6 mixture[0]=-0.132779 W_out[0][0]=0.371773 b_out[0]=0 logit[0]=0.107941 logit[1]=0.255436 max-min=0.394355 n_vocab=16 d_model=8
    [debug ssog] base_dist call=7 mixture[0]=-0.326068 W_out[0][0]=0.371773 b_out[0]=0 logit[0]=0.0259704 logit[1]=-0.34685 max-min=0.833478 n_vocab=16 d_model=8
    [debug] grad_norm=2.52491 grad_max=1.07206 param_W[0]=0.371773 param_b[0]=0
    [debug] BEFORE step: logits.size=1 seq[0].size=3 vocab=16 -2.77259 -2.77259 -2.77259 -2.77259 -2.77259 -2.77259 -2.77259 -2.77259 target=2
    [debug model] sync: b_out[0] 0 -> -0.01875 (param=-0.01875) W_out[0][0] 0.371773 -> 0.377146 (param=0.377146) n_vocab=16 d_model=8 W_out.size=16 b_out.size=16
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.173153 result[0][0]=0
    [debug ssog] base_dist call=8 mixture[0]=-0.23938 W_out[0][0]=0.377146 b_out[0]=-0.01875 logit[0]=-0.381647 logit[1]=0.0596804 max-min=1.0095 n_vocab=16 d_model=8
    [debug ssog] base_dist call=9 mixture[0]=-0.132779 W_out[0][0]=0.377146 b_out[0]=-0.01875 logit[0]=0.0883484 logit[1]=0.235843 max-min=0.51612 n_vocab=16 d_model=8
    [debug ssog] base_dist call=10 mixture[0]=-0.326068 W_out[0][0]=0.377146 b_out[0]=-0.01875 logit[0]=0.0067593 logit[1]=-0.366061 max-min=0.982307 n_vocab=16 d_model=8
    [debug] AFTER  step: logits.size=1 seq[0].size=3 vocab=16 -2.77259 -2.77259 -2.77259 -2.77259 -2.77259 -2.77259 -2.77259 -2.77259 target=2
 target=2
    [debug] loss before=2.77259 after=2.77259 change=0 param_W[0]=0.377146 param_b[0]=-0.01875
  FAIL: Loss should change after SGD step, was 2.772589 now 2.772589 (grad_norm=2.524915)

[Pipeline Integration Test]
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.0588751 result[0][0]=0
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.101983 result[0][0]=0
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=-0.024458 result[0][0]=0
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.0163873 result[0][0]=0
    [debug model] sync: b_out[0] 0 -> -0.01 (param=-0.01) W_out[0][0] 0.371773 -> 0.381773 (param=0.381773) n_vocab=16 d_model=8 W_out.size=16 b_out.size=16
    [debug] step=1 loss=2.772589 |g|=11.8163 b_out[0]: 0.0000 -> -0.0100
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.0589 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.1020 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=-0.0245 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.0164 result[0][0]=0.0000
    [debug model] sync: b_out[0] -0.0100 -> -0.0200 (param=-0.0200) W_out[0][0] 0.3818 -> 0.3918 (param=0.3918) n_vocab=16 d_model=8 W_out.size=16 b_out.size=16
    [debug] step=2 loss=2.772589 |g|=11.8163 b_out[0]: -0.0100 -> -0.0200
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.0589 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.1020 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=-0.0245 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.0164 result[0][0]=0.0000
    [debug model] sync: b_out[0] -0.0200 -> -0.0300 (param=-0.0300) W_out[0][0] 0.3918 -> 0.4018 (param=0.4018) n_vocab=16 d_model=8 W_out.size=16 b_out.size=16
    [debug] step=3 loss=2.772589 |g|=11.8163 b_out[0]: -0.0200 -> -0.0300
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.0589 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.1020 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=-0.0245 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.0164 result[0][0]=0.0000
    [debug model] sync: b_out[0] -0.0300 -> -0.0400 (param=-0.0400) W_out[0][0] 0.4018 -> 0.4118 (param=0.4118) n_vocab=16 d_model=8 W_out.size=16 b_out.size=16
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.0589 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.1020 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=-0.0245 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.0164 result[0][0]=0.0000
    [debug model] sync: b_out[0] -0.0400 -> -0.0500 (param=-0.0500) W_out[0][0] 0.4118 -> 0.4218 (param=0.4218) n_vocab=16 d_model=8 W_out.size=16 b_out.size=16
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.0589 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.1020 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=-0.0245 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.0164 result[0][0]=0.0000
    [debug model] sync: b_out[0] -0.0500 -> -0.0600 (param=-0.0600) W_out[0][0] 0.4218 -> 0.4318 (param=0.4318) n_vocab=16 d_model=8 W_out.size=16 b_out.size=16
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.0589 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.1020 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=-0.0245 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.0164 result[0][0]=0.0000
    [debug model] sync: b_out[0] -0.0600 -> -0.0700 (param=-0.0700) W_out[0][0] 0.4318 -> 0.4418 (param=0.4418) n_vocab=16 d_model=8 W_out.size=16 b_out.size=16
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.0589 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.1020 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=-0.0245 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.0164 result[0][0]=0.0000
    [debug model] sync: b_out[0] -0.0700 -> -0.0800 (param=-0.0800) W_out[0][0] 0.4418 -> 0.4518 (param=0.4518) n_vocab=16 d_model=8 W_out.size=16 b_out.size=16
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.0589 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.1020 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=-0.0245 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.0164 result[0][0]=0.0000
    [debug model] sync: b_out[0] -0.0800 -> -0.0900 (param=-0.0900) W_out[0][0] 0.4518 -> 0.4618 (param=0.4618) n_vocab=16 d_model=8 W_out.size=16 b_out.size=16
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.0589 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.1020 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=-0.0245 result[0][0]=0.0000
    [debug stre] sheaf_laplacian_all: n=1 nodes_.size()=1 rest_maps_.size()=1 d_node=4 first_feat[0]=0.0164 result[0][0]=0.0000
    [debug model] sync: b_out[0] -0.0900 -> -0.1000 (param=-0.1000) W_out[0][0] 0.4618 -> 0.4718 (param=0.4718) n_vocab=16 d_model=8 W_out.size=16 b_out.size=16
    [debug] step=10 loss=2.772589 |g|=11.8163 b_out[0]: -0.0900 -> -0.1000
  FAIL: Loss should change during training. All values = 2.772589
  Loss trajectory: 2.7726 2.7726 2.7726 2.7726 2.7726 2.7726 2.7726 2.7726 2.7726 2.7726 

=== Results ===
  Total: 277
  Passed: 275
  Failed: 2

