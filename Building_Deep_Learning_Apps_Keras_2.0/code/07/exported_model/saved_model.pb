ζω
Σ¨
:
Add
x"T
y"T
z"T"
Ttype:
2	
A
AddV2
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignAdd
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
8
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
k
NotEqual
x"T
y"T
z
"
Ttype:
2	
"$
incompatible_shape_errorbool(
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.15.02v1.15.0-rc3-22-g590d6ee’΄
p
layer_1_inputPlaceholder*
dtype0*
shape:?????????	*'
_output_shapes
:?????????	
m
layer_1/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"	   2   
_
layer_1/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *dF£Ύ
_
layer_1/random_uniform/maxConst*
valueB
 *dF£>*
_output_shapes
: *
dtype0
¨
$layer_1/random_uniform/RandomUniformRandomUniformlayer_1/random_uniform/shape*
T0*
seed2*
_output_shapes

:	2*
seed±?ε)*
dtype0
z
layer_1/random_uniform/subSublayer_1/random_uniform/maxlayer_1/random_uniform/min*
_output_shapes
: *
T0

layer_1/random_uniform/mulMul$layer_1/random_uniform/RandomUniformlayer_1/random_uniform/sub*
_output_shapes

:	2*
T0
~
layer_1/random_uniformAddlayer_1/random_uniform/mullayer_1/random_uniform/min*
_output_shapes

:	2*
T0

layer_1/kernel
VariableV2*
shape
:	2*
	container *
dtype0*
_output_shapes

:	2*
shared_name 
Ό
layer_1/kernel/AssignAssignlayer_1/kernellayer_1/random_uniform*
T0*
_output_shapes

:	2*!
_class
loc:@layer_1/kernel*
validate_shape(*
use_locking(
{
layer_1/kernel/readIdentitylayer_1/kernel*!
_class
loc:@layer_1/kernel*
_output_shapes

:	2*
T0
Z
layer_1/ConstConst*
valueB2*    *
_output_shapes
:2*
dtype0
x
layer_1/bias
VariableV2*
	container *
_output_shapes
:2*
dtype0*
shared_name *
shape:2
©
layer_1/bias/AssignAssignlayer_1/biaslayer_1/Const*
validate_shape(*
_output_shapes
:2*
use_locking(*
_class
loc:@layer_1/bias*
T0
q
layer_1/bias/readIdentitylayer_1/bias*
_class
loc:@layer_1/bias*
T0*
_output_shapes
:2

layer_1/MatMulMatMullayer_1_inputlayer_1/kernel/read*'
_output_shapes
:?????????2*
transpose_a( *
transpose_b( *
T0

layer_1/BiasAddBiasAddlayer_1/MatMullayer_1/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:?????????2
W
layer_1/ReluRelulayer_1/BiasAdd*
T0*'
_output_shapes
:?????????2
m
layer_2/random_uniform/shapeConst*
_output_shapes
:*
valueB"2   d   *
dtype0
_
layer_2/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜLΎ
_
layer_2/random_uniform/maxConst*
dtype0*
valueB
 *ΝΜL>*
_output_shapes
: 
§
$layer_2/random_uniform/RandomUniformRandomUniformlayer_2/random_uniform/shape*
_output_shapes

:2d*
T0*
seed2Ι2*
seed±?ε)*
dtype0
z
layer_2/random_uniform/subSublayer_2/random_uniform/maxlayer_2/random_uniform/min*
T0*
_output_shapes
: 

layer_2/random_uniform/mulMul$layer_2/random_uniform/RandomUniformlayer_2/random_uniform/sub*
T0*
_output_shapes

:2d
~
layer_2/random_uniformAddlayer_2/random_uniform/mullayer_2/random_uniform/min*
T0*
_output_shapes

:2d

layer_2/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes

:2d*
shape
:2d*
	container 
Ό
layer_2/kernel/AssignAssignlayer_2/kernellayer_2/random_uniform*!
_class
loc:@layer_2/kernel*
validate_shape(*
_output_shapes

:2d*
use_locking(*
T0
{
layer_2/kernel/readIdentitylayer_2/kernel*
_output_shapes

:2d*
T0*!
_class
loc:@layer_2/kernel
Z
layer_2/ConstConst*
dtype0*
valueBd*    *
_output_shapes
:d
x
layer_2/bias
VariableV2*
shape:d*
dtype0*
_output_shapes
:d*
shared_name *
	container 
©
layer_2/bias/AssignAssignlayer_2/biaslayer_2/Const*
T0*
_output_shapes
:d*
validate_shape(*
_class
loc:@layer_2/bias*
use_locking(
q
layer_2/bias/readIdentitylayer_2/bias*
_output_shapes
:d*
_class
loc:@layer_2/bias*
T0

layer_2/MatMulMatMullayer_1/Relulayer_2/kernel/read*
transpose_b( *'
_output_shapes
:?????????d*
T0*
transpose_a( 

layer_2/BiasAddBiasAddlayer_2/MatMullayer_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:?????????d
W
layer_2/ReluRelulayer_2/BiasAdd*'
_output_shapes
:?????????d*
T0
m
layer_3/random_uniform/shapeConst*
dtype0*
valueB"d   2   *
_output_shapes
:
_
layer_3/random_uniform/minConst*
valueB
 *ΝΜLΎ*
_output_shapes
: *
dtype0
_
layer_3/random_uniform/maxConst*
_output_shapes
: *
valueB
 *ΝΜL>*
dtype0
¨
$layer_3/random_uniform/RandomUniformRandomUniformlayer_3/random_uniform/shape*
seed2ΰ½*
seed±?ε)*
T0*
dtype0*
_output_shapes

:d2
z
layer_3/random_uniform/subSublayer_3/random_uniform/maxlayer_3/random_uniform/min*
T0*
_output_shapes
: 

layer_3/random_uniform/mulMul$layer_3/random_uniform/RandomUniformlayer_3/random_uniform/sub*
_output_shapes

:d2*
T0
~
layer_3/random_uniformAddlayer_3/random_uniform/mullayer_3/random_uniform/min*
T0*
_output_shapes

:d2

layer_3/kernel
VariableV2*
shared_name *
shape
:d2*
	container *
dtype0*
_output_shapes

:d2
Ό
layer_3/kernel/AssignAssignlayer_3/kernellayer_3/random_uniform*
T0*
_output_shapes

:d2*
use_locking(*
validate_shape(*!
_class
loc:@layer_3/kernel
{
layer_3/kernel/readIdentitylayer_3/kernel*!
_class
loc:@layer_3/kernel*
T0*
_output_shapes

:d2
Z
layer_3/ConstConst*
_output_shapes
:2*
dtype0*
valueB2*    
x
layer_3/bias
VariableV2*
shape:2*
shared_name *
	container *
_output_shapes
:2*
dtype0
©
layer_3/bias/AssignAssignlayer_3/biaslayer_3/Const*
use_locking(*
T0*
validate_shape(*
_class
loc:@layer_3/bias*
_output_shapes
:2
q
layer_3/bias/readIdentitylayer_3/bias*
T0*
_output_shapes
:2*
_class
loc:@layer_3/bias

layer_3/MatMulMatMullayer_2/Relulayer_3/kernel/read*
transpose_a( *
T0*
transpose_b( *'
_output_shapes
:?????????2

layer_3/BiasAddBiasAddlayer_3/MatMullayer_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:?????????2
W
layer_3/ReluRelulayer_3/BiasAdd*'
_output_shapes
:?????????2*
T0
r
!output_layer/random_uniform/shapeConst*
valueB"2      *
dtype0*
_output_shapes
:
d
output_layer/random_uniform/minConst*
valueB
 *S―Ύ*
_output_shapes
: *
dtype0
d
output_layer/random_uniform/maxConst*
valueB
 *S―>*
dtype0*
_output_shapes
: 
²
)output_layer/random_uniform/RandomUniformRandomUniform!output_layer/random_uniform/shape*
T0*
_output_shapes

:2*
seed±?ε)*
seed2’¬Σ*
dtype0

output_layer/random_uniform/subSuboutput_layer/random_uniform/maxoutput_layer/random_uniform/min*
_output_shapes
: *
T0

output_layer/random_uniform/mulMul)output_layer/random_uniform/RandomUniformoutput_layer/random_uniform/sub*
_output_shapes

:2*
T0

output_layer/random_uniformAddoutput_layer/random_uniform/muloutput_layer/random_uniform/min*
_output_shapes

:2*
T0

output_layer/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes

:2*
shape
:2*
	container 
Π
output_layer/kernel/AssignAssignoutput_layer/kerneloutput_layer/random_uniform*
validate_shape(*
T0*
_output_shapes

:2*&
_class
loc:@output_layer/kernel*
use_locking(

output_layer/kernel/readIdentityoutput_layer/kernel*&
_class
loc:@output_layer/kernel*
T0*
_output_shapes

:2
_
output_layer/ConstConst*
dtype0*
_output_shapes
:*
valueB*    
}
output_layer/bias
VariableV2*
shared_name *
	container *
_output_shapes
:*
dtype0*
shape:
½
output_layer/bias/AssignAssignoutput_layer/biasoutput_layer/Const*
validate_shape(*$
_class
loc:@output_layer/bias*
use_locking(*
T0*
_output_shapes
:

output_layer/bias/readIdentityoutput_layer/bias*
T0*$
_class
loc:@output_layer/bias*
_output_shapes
:

output_layer/MatMulMatMullayer_3/Reluoutput_layer/kernel/read*
transpose_b( *
T0*'
_output_shapes
:?????????*
transpose_a( 

output_layer/BiasAddBiasAddoutput_layer/MatMuloutput_layer/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:?????????
]
iterations/initial_valueConst*
valueB
 *    *
_output_shapes
: *
dtype0
n

iterations
VariableV2*
shared_name *
shape: *
	container *
dtype0*
_output_shapes
: 
ͺ
iterations/AssignAssign
iterationsiterations/initial_value*
_class
loc:@iterations*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
g
iterations/readIdentity
iterations*
_class
loc:@iterations*
T0*
_output_shapes
: 
U
lr/initial_valueConst*
_output_shapes
: *
valueB
 *o:*
dtype0
f
lr
VariableV2*
shared_name *
	container *
dtype0*
shape: *
_output_shapes
: 

	lr/AssignAssignlrlr/initial_value*
use_locking(*
T0*
_class
	loc:@lr*
validate_shape(*
_output_shapes
: 
O
lr/readIdentitylr*
T0*
_output_shapes
: *
_class
	loc:@lr
Y
beta_1/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *fff?
j
beta_1
VariableV2*
	container *
shared_name *
_output_shapes
: *
dtype0*
shape: 

beta_1/AssignAssignbeta_1beta_1/initial_value*
_class
loc:@beta_1*
validate_shape(*
_output_shapes
: *
T0*
use_locking(
[
beta_1/readIdentitybeta_1*
T0*
_output_shapes
: *
_class
loc:@beta_1
Y
beta_2/initial_valueConst*
dtype0*
valueB
 *wΎ?*
_output_shapes
: 
j
beta_2
VariableV2*
shared_name *
shape: *
dtype0*
_output_shapes
: *
	container 

beta_2/AssignAssignbeta_2beta_2/initial_value*
_class
loc:@beta_2*
validate_shape(*
use_locking(*
_output_shapes
: *
T0
[
beta_2/readIdentitybeta_2*
T0*
_class
loc:@beta_2*
_output_shapes
: 
X
decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
i
decay
VariableV2*
shared_name *
	container *
dtype0*
shape: *
_output_shapes
: 

decay/AssignAssigndecaydecay/initial_value*
use_locking(*
_class

loc:@decay*
validate_shape(*
_output_shapes
: *
T0
X

decay/readIdentitydecay*
T0*
_class

loc:@decay*
_output_shapes
: 
v
output_layer_sample_weightsPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????

output_layer_targetPlaceholder*0
_output_shapes
:??????????????????*
dtype0*%
shape:??????????????????
p
subSuboutput_layer/BiasAddoutput_layer_target*
T0*0
_output_shapes
:??????????????????
P
SquareSquaresub*0
_output_shapes
:??????????????????*
T0
X
Mean/reduction_indicesConst*
value	B :*
_output_shapes
: *
dtype0
w
MeanMeanSquareMean/reduction_indices*
	keep_dims( *#
_output_shapes
:?????????*

Tidx0*
T0
[
Mean_1/reduction_indicesConst*
valueB *
_output_shapes
: *
dtype0
y
Mean_1MeanMeanMean_1/reduction_indices*

Tidx0*
	keep_dims( *#
_output_shapes
:?????????*
T0
]
mulMulMean_1output_layer_sample_weights*#
_output_shapes
:?????????*
T0
O

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    

NotEqualNotEqualoutput_layer_sample_weights
NotEqual/y*
T0*#
_output_shapes
:?????????*
incompatible_shape_error(
c
CastCastNotEqual*

SrcT0
*#
_output_shapes
:?????????*
Truncate( *

DstT0
O
ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Y
Mean_2MeanCastConst*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
M
truedivRealDivmulMean_2*#
_output_shapes
:?????????*
T0
Q
Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
^
Mean_3MeantruedivConst_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
L
mul_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
>
mul_1Mulmul_1/xMean_3*
_output_shapes
: *
T0
l
gradients/ShapeConst*
_output_shapes
: *
valueB *
_class

loc:@mul_1*
dtype0
r
gradients/grad_ys_0Const*
valueB
 *  ?*
_class

loc:@mul_1*
dtype0*
_output_shapes
: 

gradients/FillFillgradients/Shapegradients/grad_ys_0*
_class

loc:@mul_1*
_output_shapes
: *
T0*

index_type0
r
gradients/mul_1_grad/MulMulgradients/FillMean_3*
_class

loc:@mul_1*
T0*
_output_shapes
: 
u
gradients/mul_1_grad/Mul_1Mulgradients/Fillmul_1/x*
_class

loc:@mul_1*
_output_shapes
: *
T0

#gradients/Mean_3_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0*
_class
loc:@Mean_3
·
gradients/Mean_3_grad/ReshapeReshapegradients/mul_1_grad/Mul_1#gradients/Mean_3_grad/Reshape/shape*
T0*
_class
loc:@Mean_3*
_output_shapes
:*
Tshape0
}
gradients/Mean_3_grad/ShapeShapetruediv*
out_type0*
_class
loc:@Mean_3*
T0*
_output_shapes
:
Ή
gradients/Mean_3_grad/TileTilegradients/Mean_3_grad/Reshapegradients/Mean_3_grad/Shape*

Tmultiples0*
T0*
_class
loc:@Mean_3*#
_output_shapes
:?????????

gradients/Mean_3_grad/Shape_1Shapetruediv*
_class
loc:@Mean_3*
T0*
out_type0*
_output_shapes
:
{
gradients/Mean_3_grad/Shape_2Const*
_output_shapes
: *
valueB *
_class
loc:@Mean_3*
dtype0

gradients/Mean_3_grad/ConstConst*
_class
loc:@Mean_3*
dtype0*
valueB: *
_output_shapes
:
·
gradients/Mean_3_grad/ProdProdgradients/Mean_3_grad/Shape_1gradients/Mean_3_grad/Const*
T0*
	keep_dims( *
_output_shapes
: *

Tidx0*
_class
loc:@Mean_3

gradients/Mean_3_grad/Const_1Const*
_class
loc:@Mean_3*
dtype0*
valueB: *
_output_shapes
:
»
gradients/Mean_3_grad/Prod_1Prodgradients/Mean_3_grad/Shape_2gradients/Mean_3_grad/Const_1*
_class
loc:@Mean_3*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
|
gradients/Mean_3_grad/Maximum/yConst*
dtype0*
_class
loc:@Mean_3*
_output_shapes
: *
value	B :
£
gradients/Mean_3_grad/MaximumMaximumgradients/Mean_3_grad/Prod_1gradients/Mean_3_grad/Maximum/y*
_output_shapes
: *
_class
loc:@Mean_3*
T0
‘
gradients/Mean_3_grad/floordivFloorDivgradients/Mean_3_grad/Prodgradients/Mean_3_grad/Maximum*
T0*
_class
loc:@Mean_3*
_output_shapes
: 

gradients/Mean_3_grad/CastCastgradients/Mean_3_grad/floordiv*

DstT0*
Truncate( *

SrcT0*
_output_shapes
: *
_class
loc:@Mean_3
©
gradients/Mean_3_grad/truedivRealDivgradients/Mean_3_grad/Tilegradients/Mean_3_grad/Cast*
_class
loc:@Mean_3*
T0*#
_output_shapes
:?????????
{
gradients/truediv_grad/ShapeShapemul*
_output_shapes
:*
_class
loc:@truediv*
out_type0*
T0
}
gradients/truediv_grad/Shape_1Const*
_output_shapes
: *
_class
loc:@truediv*
dtype0*
valueB 
ά
,gradients/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_grad/Shapegradients/truediv_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
_class
loc:@truediv*
T0

gradients/truediv_grad/RealDivRealDivgradients/Mean_3_grad/truedivMean_2*
T0*
_class
loc:@truediv*#
_output_shapes
:?????????
Λ
gradients/truediv_grad/SumSumgradients/truediv_grad/RealDiv,gradients/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*
_class
loc:@truediv
»
gradients/truediv_grad/ReshapeReshapegradients/truediv_grad/Sumgradients/truediv_grad/Shape*
Tshape0*#
_output_shapes
:?????????*
_class
loc:@truediv*
T0
p
gradients/truediv_grad/NegNegmul*
_class
loc:@truediv*
T0*#
_output_shapes
:?????????

 gradients/truediv_grad/RealDiv_1RealDivgradients/truediv_grad/NegMean_2*
_class
loc:@truediv*#
_output_shapes
:?????????*
T0

 gradients/truediv_grad/RealDiv_2RealDiv gradients/truediv_grad/RealDiv_1Mean_2*#
_output_shapes
:?????????*
_class
loc:@truediv*
T0
¬
gradients/truediv_grad/mulMulgradients/Mean_3_grad/truediv gradients/truediv_grad/RealDiv_2*
_class
loc:@truediv*
T0*#
_output_shapes
:?????????
Λ
gradients/truediv_grad/Sum_1Sumgradients/truediv_grad/mul.gradients/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
T0*
_class
loc:@truediv*
	keep_dims( 
΄
 gradients/truediv_grad/Reshape_1Reshapegradients/truediv_grad/Sum_1gradients/truediv_grad/Shape_1*
_output_shapes
: *
_class
loc:@truediv*
Tshape0*
T0
v
gradients/mul_grad/ShapeShapeMean_1*
T0*
out_type0*
_class

loc:@mul*
_output_shapes
:

gradients/mul_grad/Shape_1Shapeoutput_layer_sample_weights*
T0*
_class

loc:@mul*
out_type0*
_output_shapes
:
Μ
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
_class

loc:@mul*
T0
 
gradients/mul_grad/MulMulgradients/truediv_grad/Reshapeoutput_layer_sample_weights*
_class

loc:@mul*
T0*#
_output_shapes
:?????????
·
gradients/mul_grad/SumSumgradients/mul_grad/Mul(gradients/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
_class

loc:@mul*
	keep_dims( *

Tidx0
«
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
_class

loc:@mul*
Tshape0*#
_output_shapes
:?????????

gradients/mul_grad/Mul_1MulMean_1gradients/truediv_grad/Reshape*
_class

loc:@mul*#
_output_shapes
:?????????*
T0
½
gradients/mul_grad/Sum_1Sumgradients/mul_grad/Mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( *
_class

loc:@mul
±
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????*
_class

loc:@mul
z
gradients/Mean_1_grad/ShapeShapeMean*
T0*
out_type0*
_class
loc:@Mean_1*
_output_shapes
:
w
gradients/Mean_1_grad/SizeConst*
dtype0*
_class
loc:@Mean_1*
value	B :*
_output_shapes
: 

gradients/Mean_1_grad/addAddV2Mean_1/reduction_indicesgradients/Mean_1_grad/Size*
_class
loc:@Mean_1*
_output_shapes
: *
T0

gradients/Mean_1_grad/modFloorModgradients/Mean_1_grad/addgradients/Mean_1_grad/Size*
T0*
_class
loc:@Mean_1*
_output_shapes
: 

gradients/Mean_1_grad/Shape_1Const*
_class
loc:@Mean_1*
_output_shapes
:*
valueB: *
dtype0
~
!gradients/Mean_1_grad/range/startConst*
_output_shapes
: *
value	B : *
dtype0*
_class
loc:@Mean_1
~
!gradients/Mean_1_grad/range/deltaConst*
_output_shapes
: *
value	B :*
_class
loc:@Mean_1*
dtype0
Ι
gradients/Mean_1_grad/rangeRange!gradients/Mean_1_grad/range/startgradients/Mean_1_grad/Size!gradients/Mean_1_grad/range/delta*
_class
loc:@Mean_1*

Tidx0*
_output_shapes
:
}
 gradients/Mean_1_grad/Fill/valueConst*
value	B :*
_output_shapes
: *
_class
loc:@Mean_1*
dtype0
³
gradients/Mean_1_grad/FillFillgradients/Mean_1_grad/Shape_1 gradients/Mean_1_grad/Fill/value*
_class
loc:@Mean_1*
_output_shapes
: *

index_type0*
T0
ξ
#gradients/Mean_1_grad/DynamicStitchDynamicStitchgradients/Mean_1_grad/rangegradients/Mean_1_grad/modgradients/Mean_1_grad/Shapegradients/Mean_1_grad/Fill*
_class
loc:@Mean_1*
N*
T0*
_output_shapes
:
|
gradients/Mean_1_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
_class
loc:@Mean_1*
value	B :
?
gradients/Mean_1_grad/MaximumMaximum#gradients/Mean_1_grad/DynamicStitchgradients/Mean_1_grad/Maximum/y*
_output_shapes
:*
_class
loc:@Mean_1*
T0
¦
gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Shapegradients/Mean_1_grad/Maximum*
T0*
_class
loc:@Mean_1*
_output_shapes
:
ΐ
gradients/Mean_1_grad/ReshapeReshapegradients/mul_grad/Reshape#gradients/Mean_1_grad/DynamicStitch*
T0*#
_output_shapes
:?????????*
_class
loc:@Mean_1*
Tshape0
Ό
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/floordiv*#
_output_shapes
:?????????*

Tmultiples0*
_class
loc:@Mean_1*
T0
|
gradients/Mean_1_grad/Shape_2ShapeMean*
T0*
_class
loc:@Mean_1*
_output_shapes
:*
out_type0
~
gradients/Mean_1_grad/Shape_3ShapeMean_1*
_class
loc:@Mean_1*
_output_shapes
:*
out_type0*
T0

gradients/Mean_1_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0*
_class
loc:@Mean_1
·
gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const*
T0*
	keep_dims( *

Tidx0*
_class
loc:@Mean_1*
_output_shapes
: 

gradients/Mean_1_grad/Const_1Const*
valueB: *
_class
loc:@Mean_1*
_output_shapes
:*
dtype0
»
gradients/Mean_1_grad/Prod_1Prodgradients/Mean_1_grad/Shape_3gradients/Mean_1_grad/Const_1*
_output_shapes
: *
	keep_dims( *
T0*

Tidx0*
_class
loc:@Mean_1
~
!gradients/Mean_1_grad/Maximum_1/yConst*
dtype0*
_class
loc:@Mean_1*
value	B :*
_output_shapes
: 
§
gradients/Mean_1_grad/Maximum_1Maximumgradients/Mean_1_grad/Prod_1!gradients/Mean_1_grad/Maximum_1/y*
T0*
_output_shapes
: *
_class
loc:@Mean_1
₯
 gradients/Mean_1_grad/floordiv_1FloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum_1*
T0*
_output_shapes
: *
_class
loc:@Mean_1

gradients/Mean_1_grad/CastCast gradients/Mean_1_grad/floordiv_1*

SrcT0*
_class
loc:@Mean_1*
Truncate( *
_output_shapes
: *

DstT0
©
gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*
T0*
_class
loc:@Mean_1*#
_output_shapes
:?????????
x
gradients/Mean_grad/ShapeShapeSquare*
_class
	loc:@Mean*
T0*
_output_shapes
:*
out_type0
s
gradients/Mean_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :*
_class
	loc:@Mean

gradients/Mean_grad/addAddV2Mean/reduction_indicesgradients/Mean_grad/Size*
_class
	loc:@Mean*
T0*
_output_shapes
: 

gradients/Mean_grad/modFloorModgradients/Mean_grad/addgradients/Mean_grad/Size*
_class
	loc:@Mean*
_output_shapes
: *
T0
w
gradients/Mean_grad/Shape_1Const*
dtype0*
_output_shapes
: *
_class
	loc:@Mean*
valueB 
z
gradients/Mean_grad/range/startConst*
_output_shapes
: *
dtype0*
_class
	loc:@Mean*
value	B : 
z
gradients/Mean_grad/range/deltaConst*
_output_shapes
: *
value	B :*
_class
	loc:@Mean*
dtype0
Ώ
gradients/Mean_grad/rangeRangegradients/Mean_grad/range/startgradients/Mean_grad/Sizegradients/Mean_grad/range/delta*

Tidx0*
_class
	loc:@Mean*
_output_shapes
:
y
gradients/Mean_grad/Fill/valueConst*
_class
	loc:@Mean*
_output_shapes
: *
value	B :*
dtype0
©
gradients/Mean_grad/FillFillgradients/Mean_grad/Shape_1gradients/Mean_grad/Fill/value*
T0*
_output_shapes
: *

index_type0*
_class
	loc:@Mean
β
!gradients/Mean_grad/DynamicStitchDynamicStitchgradients/Mean_grad/rangegradients/Mean_grad/modgradients/Mean_grad/Shapegradients/Mean_grad/Fill*
_output_shapes
:*
T0*
N*
_class
	loc:@Mean
x
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0*
_class
	loc:@Mean
¦
gradients/Mean_grad/MaximumMaximum!gradients/Mean_grad/DynamicStitchgradients/Mean_grad/Maximum/y*
_class
	loc:@Mean*
_output_shapes
:*
T0

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Shapegradients/Mean_grad/Maximum*
T0*
_class
	loc:@Mean*
_output_shapes
:
Κ
gradients/Mean_grad/ReshapeReshapegradients/Mean_1_grad/truediv!gradients/Mean_grad/DynamicStitch*
Tshape0*0
_output_shapes
:??????????????????*
T0*
_class
	loc:@Mean
Α
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:??????????????????*
_class
	loc:@Mean
z
gradients/Mean_grad/Shape_2ShapeSquare*
T0*
_output_shapes
:*
_class
	loc:@Mean*
out_type0
x
gradients/Mean_grad/Shape_3ShapeMean*
_class
	loc:@Mean*
out_type0*
_output_shapes
:*
T0
|
gradients/Mean_grad/ConstConst*
_output_shapes
:*
_class
	loc:@Mean*
dtype0*
valueB: 
―
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_2gradients/Mean_grad/Const*

Tidx0*
_class
	loc:@Mean*
_output_shapes
: *
	keep_dims( *
T0
~
gradients/Mean_grad/Const_1Const*
dtype0*
_class
	loc:@Mean*
valueB: *
_output_shapes
:
³
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_3gradients/Mean_grad/Const_1*

Tidx0*
_class
	loc:@Mean*
	keep_dims( *
_output_shapes
: *
T0
z
gradients/Mean_grad/Maximum_1/yConst*
value	B :*
_class
	loc:@Mean*
_output_shapes
: *
dtype0

gradients/Mean_grad/Maximum_1Maximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum_1/y*
_output_shapes
: *
_class
	loc:@Mean*
T0

gradients/Mean_grad/floordiv_1FloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum_1*
_class
	loc:@Mean*
_output_shapes
: *
T0

gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv_1*

DstT0*
_output_shapes
: *
_class
	loc:@Mean*

SrcT0*
Truncate( 
?
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*0
_output_shapes
:??????????????????*
_class
	loc:@Mean*
T0

gradients/Square_grad/ConstConst^gradients/Mean_grad/truediv*
_output_shapes
: *
dtype0*
valueB
 *   @*
_class
loc:@Square

gradients/Square_grad/MulMulsubgradients/Square_grad/Const*
T0*0
_output_shapes
:??????????????????*
_class
loc:@Square
°
gradients/Square_grad/Mul_1Mulgradients/Mean_grad/truedivgradients/Square_grad/Mul*
T0*0
_output_shapes
:??????????????????*
_class
loc:@Square

gradients/sub_grad/ShapeShapeoutput_layer/BiasAdd*
out_type0*
_class

loc:@sub*
T0*
_output_shapes
:

gradients/sub_grad/Shape_1Shapeoutput_layer_target*
_class

loc:@sub*
T0*
out_type0*
_output_shapes
:
Μ
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
_class

loc:@sub*2
_output_shapes 
:?????????:?????????*
T0
Ό
gradients/sub_grad/SumSumgradients/Square_grad/Mul_1(gradients/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
_class

loc:@sub*

Tidx0*
T0
―
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????*
_class

loc:@sub

gradients/sub_grad/NegNeggradients/Square_grad/Mul_1*
T0*0
_output_shapes
:??????????????????*
_class

loc:@sub
»
gradients/sub_grad/Sum_1Sumgradients/sub_grad/Neg*gradients/sub_grad/BroadcastGradientArgs:1*
T0*
_class

loc:@sub*

Tidx0*
_output_shapes
:*
	keep_dims( 
Ύ
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Sum_1gradients/sub_grad/Shape_1*
T0*
Tshape0*
_class

loc:@sub*0
_output_shapes
:??????????????????
Ώ
/gradients/output_layer/BiasAdd_grad/BiasAddGradBiasAddGradgradients/sub_grad/Reshape*
_output_shapes
:*
data_formatNHWC*'
_class
loc:@output_layer/BiasAdd*
T0
ι
)gradients/output_layer/MatMul_grad/MatMulMatMulgradients/sub_grad/Reshapeoutput_layer/kernel/read*'
_output_shapes
:?????????2*
transpose_a( *
transpose_b(*&
_class
loc:@output_layer/MatMul*
T0
Φ
+gradients/output_layer/MatMul_grad/MatMul_1MatMullayer_3/Relugradients/sub_grad/Reshape*
transpose_a(*
T0*&
_class
loc:@output_layer/MatMul*
_output_shapes

:2*
transpose_b( 
Ό
$gradients/layer_3/Relu_grad/ReluGradReluGrad)gradients/output_layer/MatMul_grad/MatMullayer_3/Relu*'
_output_shapes
:?????????2*
_class
loc:@layer_3/Relu*
T0
Ώ
*gradients/layer_3/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients/layer_3/Relu_grad/ReluGrad*
data_formatNHWC*"
_class
loc:@layer_3/BiasAdd*
_output_shapes
:2*
T0
δ
$gradients/layer_3/MatMul_grad/MatMulMatMul$gradients/layer_3/Relu_grad/ReluGradlayer_3/kernel/read*
T0*
transpose_b(*'
_output_shapes
:?????????d*
transpose_a( *!
_class
loc:@layer_3/MatMul
Φ
&gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_2/Relu$gradients/layer_3/Relu_grad/ReluGrad*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:d2*!
_class
loc:@layer_3/MatMul
·
$gradients/layer_2/Relu_grad/ReluGradReluGrad$gradients/layer_3/MatMul_grad/MatMullayer_2/Relu*
_class
loc:@layer_2/Relu*
T0*'
_output_shapes
:?????????d
Ώ
*gradients/layer_2/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients/layer_2/Relu_grad/ReluGrad*"
_class
loc:@layer_2/BiasAdd*
_output_shapes
:d*
data_formatNHWC*
T0
δ
$gradients/layer_2/MatMul_grad/MatMulMatMul$gradients/layer_2/Relu_grad/ReluGradlayer_2/kernel/read*
transpose_a( *
transpose_b(*!
_class
loc:@layer_2/MatMul*
T0*'
_output_shapes
:?????????2
Φ
&gradients/layer_2/MatMul_grad/MatMul_1MatMullayer_1/Relu$gradients/layer_2/Relu_grad/ReluGrad*
T0*
_output_shapes

:2d*!
_class
loc:@layer_2/MatMul*
transpose_a(*
transpose_b( 
·
$gradients/layer_1/Relu_grad/ReluGradReluGrad$gradients/layer_2/MatMul_grad/MatMullayer_1/Relu*'
_output_shapes
:?????????2*
T0*
_class
loc:@layer_1/Relu
Ώ
*gradients/layer_1/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients/layer_1/Relu_grad/ReluGrad*
_output_shapes
:2*
data_formatNHWC*
T0*"
_class
loc:@layer_1/BiasAdd
δ
$gradients/layer_1/MatMul_grad/MatMulMatMul$gradients/layer_1/Relu_grad/ReluGradlayer_1/kernel/read*'
_output_shapes
:?????????	*
transpose_b(*
T0*
transpose_a( *!
_class
loc:@layer_1/MatMul
Χ
&gradients/layer_1/MatMul_grad/MatMul_1MatMullayer_1_input$gradients/layer_1/Relu_grad/ReluGrad*
_output_shapes

:	2*
T0*
transpose_a(*!
_class
loc:@layer_1/MatMul*
transpose_b( 
T
AssignAdd/valueConst*
dtype0*
_output_shapes
: *
valueB
 *  ?

	AssignAdd	AssignAdd
iterationsAssignAdd/value*
use_locking( *
_output_shapes
: *
T0*
_class
loc:@iterations
J
add/yConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
E
addAddV2iterations/readadd/y*
_output_shapes
: *
T0
=
PowPowbeta_2/readadd*
_output_shapes
: *
T0
L
sub_1/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
;
sub_1Subsub_1/xPow*
T0*
_output_shapes
: 
L
Const_2Const*
_output_shapes
: *
valueB
 *    *
dtype0
L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  
Q
clip_by_value/MinimumMinimumsub_1Const_3*
_output_shapes
: *
T0
Y
clip_by_valueMaximumclip_by_value/MinimumConst_2*
_output_shapes
: *
T0
<
SqrtSqrtclip_by_value*
T0*
_output_shapes
: 
?
Pow_1Powbeta_1/readadd*
_output_shapes
: *
T0
L
sub_2/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
=
sub_2Subsub_2/xPow_1*
_output_shapes
: *
T0
B
	truediv_1RealDivSqrtsub_2*
T0*
_output_shapes
: 
A
mul_2Mullr/read	truediv_1*
_output_shapes
: *
T0
\
Const_4Const*
dtype0*
valueB	2*    *
_output_shapes

:	2
|
Variable
VariableV2*
shared_name *
shape
:	2*
_output_shapes

:	2*
dtype0*
	container 

Variable/AssignAssignVariableConst_4*
T0*
validate_shape(*
_output_shapes

:	2*
_class
loc:@Variable*
use_locking(
i
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes

:	2
T
Const_5Const*
dtype0*
valueB2*    *
_output_shapes
:2
v

Variable_1
VariableV2*
shared_name *
shape:2*
dtype0*
_output_shapes
:2*
	container 

Variable_1/AssignAssign
Variable_1Const_5*
_class
loc:@Variable_1*
validate_shape(*
T0*
use_locking(*
_output_shapes
:2
k
Variable_1/readIdentity
Variable_1*
_output_shapes
:2*
_class
loc:@Variable_1*
T0
\
Const_6Const*
valueB2d*    *
dtype0*
_output_shapes

:2d
~

Variable_2
VariableV2*
_output_shapes

:2d*
	container *
shared_name *
shape
:2d*
dtype0
‘
Variable_2/AssignAssign
Variable_2Const_6*
T0*
_output_shapes

:2d*
validate_shape(*
_class
loc:@Variable_2*
use_locking(
o
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2*
_output_shapes

:2d
T
Const_7Const*
dtype0*
_output_shapes
:d*
valueBd*    
v

Variable_3
VariableV2*
	container *
shape:d*
_output_shapes
:d*
shared_name *
dtype0

Variable_3/AssignAssign
Variable_3Const_7*
use_locking(*
_class
loc:@Variable_3*
validate_shape(*
T0*
_output_shapes
:d
k
Variable_3/readIdentity
Variable_3*
T0*
_output_shapes
:d*
_class
loc:@Variable_3
\
Const_8Const*
valueBd2*    *
_output_shapes

:d2*
dtype0
~

Variable_4
VariableV2*
shared_name *
	container *
shape
:d2*
dtype0*
_output_shapes

:d2
‘
Variable_4/AssignAssign
Variable_4Const_8*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:d2
o
Variable_4/readIdentity
Variable_4*
_output_shapes

:d2*
_class
loc:@Variable_4*
T0
T
Const_9Const*
valueB2*    *
_output_shapes
:2*
dtype0
v

Variable_5
VariableV2*
shared_name *
	container *
dtype0*
_output_shapes
:2*
shape:2

Variable_5/AssignAssign
Variable_5Const_9*
validate_shape(*
use_locking(*
_class
loc:@Variable_5*
_output_shapes
:2*
T0
k
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0*
_output_shapes
:2
]
Const_10Const*
_output_shapes

:2*
valueB2*    *
dtype0
~

Variable_6
VariableV2*
shape
:2*
_output_shapes

:2*
dtype0*
shared_name *
	container 
’
Variable_6/AssignAssign
Variable_6Const_10*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*
_output_shapes

:2
o
Variable_6/readIdentity
Variable_6*
_output_shapes

:2*
T0*
_class
loc:@Variable_6
U
Const_11Const*
dtype0*
_output_shapes
:*
valueB*    
v

Variable_7
VariableV2*
shape:*
_output_shapes
:*
	container *
shared_name *
dtype0

Variable_7/AssignAssign
Variable_7Const_11*
_output_shapes
:*
_class
loc:@Variable_7*
use_locking(*
validate_shape(*
T0
k
Variable_7/readIdentity
Variable_7*
_class
loc:@Variable_7*
_output_shapes
:*
T0
]
Const_12Const*
_output_shapes

:	2*
dtype0*
valueB	2*    
~

Variable_8
VariableV2*
dtype0*
_output_shapes

:	2*
	container *
shared_name *
shape
:	2
’
Variable_8/AssignAssign
Variable_8Const_12*
_output_shapes

:	2*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_8
o
Variable_8/readIdentity
Variable_8*
_class
loc:@Variable_8*
T0*
_output_shapes

:	2
U
Const_13Const*
_output_shapes
:2*
dtype0*
valueB2*    
v

Variable_9
VariableV2*
shape:2*
	container *
shared_name *
dtype0*
_output_shapes
:2

Variable_9/AssignAssign
Variable_9Const_13*
_class
loc:@Variable_9*
_output_shapes
:2*
use_locking(*
validate_shape(*
T0
k
Variable_9/readIdentity
Variable_9*
_output_shapes
:2*
_class
loc:@Variable_9*
T0
]
Const_14Const*
dtype0*
valueB2d*    *
_output_shapes

:2d

Variable_10
VariableV2*
shared_name *
shape
:2d*
dtype0*
	container *
_output_shapes

:2d
₯
Variable_10/AssignAssignVariable_10Const_14*
use_locking(*
_class
loc:@Variable_10*
_output_shapes

:2d*
validate_shape(*
T0
r
Variable_10/readIdentityVariable_10*
_output_shapes

:2d*
T0*
_class
loc:@Variable_10
U
Const_15Const*
valueBd*    *
_output_shapes
:d*
dtype0
w
Variable_11
VariableV2*
_output_shapes
:d*
shape:d*
dtype0*
shared_name *
	container 
‘
Variable_11/AssignAssignVariable_11Const_15*
_output_shapes
:d*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(
n
Variable_11/readIdentityVariable_11*
T0*
_output_shapes
:d*
_class
loc:@Variable_11
]
Const_16Const*
dtype0*
_output_shapes

:d2*
valueBd2*    

Variable_12
VariableV2*
dtype0*
shared_name *
_output_shapes

:d2*
	container *
shape
:d2
₯
Variable_12/AssignAssignVariable_12Const_16*
_output_shapes

:d2*
T0*
validate_shape(*
_class
loc:@Variable_12*
use_locking(
r
Variable_12/readIdentityVariable_12*
_output_shapes

:d2*
_class
loc:@Variable_12*
T0
U
Const_17Const*
dtype0*
_output_shapes
:2*
valueB2*    
w
Variable_13
VariableV2*
	container *
shape:2*
dtype0*
shared_name *
_output_shapes
:2
‘
Variable_13/AssignAssignVariable_13Const_17*
T0*
use_locking(*
_class
loc:@Variable_13*
validate_shape(*
_output_shapes
:2
n
Variable_13/readIdentityVariable_13*
T0*
_class
loc:@Variable_13*
_output_shapes
:2
]
Const_18Const*
dtype0*
_output_shapes

:2*
valueB2*    

Variable_14
VariableV2*
	container *
dtype0*
shape
:2*
_output_shapes

:2*
shared_name 
₯
Variable_14/AssignAssignVariable_14Const_18*
use_locking(*
validate_shape(*
T0*
_output_shapes

:2*
_class
loc:@Variable_14
r
Variable_14/readIdentityVariable_14*
_output_shapes

:2*
T0*
_class
loc:@Variable_14
U
Const_19Const*
valueB*    *
dtype0*
_output_shapes
:
w
Variable_15
VariableV2*
shared_name *
dtype0*
	container *
shape:*
_output_shapes
:
‘
Variable_15/AssignAssignVariable_15Const_19*
T0*
_output_shapes
:*
use_locking(*
_class
loc:@Variable_15*
validate_shape(
n
Variable_15/readIdentityVariable_15*
_output_shapes
:*
_class
loc:@Variable_15*
T0
Q
mul_3Mulbeta_1/readVariable/read*
_output_shapes

:	2*
T0
L
sub_3/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
C
sub_3Subsub_3/xbeta_1/read*
_output_shapes
: *
T0
d
mul_4Mulsub_3&gradients/layer_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:	2
E
add_1AddV2mul_3mul_4*
T0*
_output_shapes

:	2
S
mul_5Mulbeta_2/readVariable_8/read*
_output_shapes

:	2*
T0
L
sub_4/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
C
sub_4Subsub_4/xbeta_2/read*
T0*
_output_shapes
: 
c
Square_1Square&gradients/layer_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:	2
F
mul_6Mulsub_4Square_1*
_output_shapes

:	2*
T0
E
add_2AddV2mul_5mul_6*
_output_shapes

:	2*
T0
C
mul_7Mulmul_2add_1*
_output_shapes

:	2*
T0
M
Const_20Const*
valueB
 *    *
dtype0*
_output_shapes
: 
M
Const_21Const*
valueB
 *  *
dtype0*
_output_shapes
: 
\
clip_by_value_1/MinimumMinimumadd_2Const_21*
_output_shapes

:	2*
T0
f
clip_by_value_1Maximumclip_by_value_1/MinimumConst_20*
_output_shapes

:	2*
T0
H
Sqrt_1Sqrtclip_by_value_1*
_output_shapes

:	2*
T0
L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+2
H
add_3AddV2Sqrt_1add_3/y*
_output_shapes

:	2*
T0
K
	truediv_2RealDivmul_7add_3*
T0*
_output_shapes

:	2
U
sub_5Sublayer_1/kernel/read	truediv_2*
T0*
_output_shapes

:	2

AssignAssignVariableadd_1*
T0*
_class
loc:@Variable*
_output_shapes

:	2*
use_locking(*
validate_shape(

Assign_1Assign
Variable_8add_2*
validate_shape(*
_class
loc:@Variable_8*
_output_shapes

:	2*
T0*
use_locking(

Assign_2Assignlayer_1/kernelsub_5*
validate_shape(*
use_locking(*
_output_shapes

:	2*
T0*!
_class
loc:@layer_1/kernel
O
mul_8Mulbeta_1/readVariable_1/read*
_output_shapes
:2*
T0
L
sub_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
C
sub_6Subsub_6/xbeta_1/read*
T0*
_output_shapes
: 
d
mul_9Mulsub_6*gradients/layer_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:2*
T0
A
add_4AddV2mul_8mul_9*
_output_shapes
:2*
T0
P
mul_10Mulbeta_2/readVariable_9/read*
_output_shapes
:2*
T0
L
sub_7/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
C
sub_7Subsub_7/xbeta_2/read*
T0*
_output_shapes
: 
c
Square_2Square*gradients/layer_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:2*
T0
C
mul_11Mulsub_7Square_2*
_output_shapes
:2*
T0
C
add_5AddV2mul_10mul_11*
T0*
_output_shapes
:2
@
mul_12Mulmul_2add_4*
T0*
_output_shapes
:2
M
Const_22Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_23Const*
_output_shapes
: *
dtype0*
valueB
 *  
X
clip_by_value_2/MinimumMinimumadd_5Const_23*
_output_shapes
:2*
T0
b
clip_by_value_2Maximumclip_by_value_2/MinimumConst_22*
_output_shapes
:2*
T0
D
Sqrt_2Sqrtclip_by_value_2*
_output_shapes
:2*
T0
L
add_6/yConst*
dtype0*
valueB
 *wΜ+2*
_output_shapes
: 
D
add_6AddV2Sqrt_2add_6/y*
T0*
_output_shapes
:2
H
	truediv_3RealDivmul_12add_6*
T0*
_output_shapes
:2
O
sub_8Sublayer_1/bias/read	truediv_3*
_output_shapes
:2*
T0

Assign_3Assign
Variable_1add_4*
use_locking(*
_class
loc:@Variable_1*
_output_shapes
:2*
T0*
validate_shape(

Assign_4Assign
Variable_9add_5*
_class
loc:@Variable_9*
use_locking(*
_output_shapes
:2*
validate_shape(*
T0

Assign_5Assignlayer_1/biassub_8*
validate_shape(*
use_locking(*
_output_shapes
:2*
T0*
_class
loc:@layer_1/bias
T
mul_13Mulbeta_1/readVariable_2/read*
_output_shapes

:2d*
T0
L
sub_9/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
C
sub_9Subsub_9/xbeta_1/read*
_output_shapes
: *
T0
e
mul_14Mulsub_9&gradients/layer_2/MatMul_grad/MatMul_1*
_output_shapes

:2d*
T0
G
add_7AddV2mul_13mul_14*
_output_shapes

:2d*
T0
U
mul_15Mulbeta_2/readVariable_10/read*
_output_shapes

:2d*
T0
M
sub_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
E
sub_10Subsub_10/xbeta_2/read*
T0*
_output_shapes
: 
c
Square_3Square&gradients/layer_2/MatMul_grad/MatMul_1*
_output_shapes

:2d*
T0
H
mul_16Mulsub_10Square_3*
T0*
_output_shapes

:2d
G
add_8AddV2mul_15mul_16*
T0*
_output_shapes

:2d
D
mul_17Mulmul_2add_7*
T0*
_output_shapes

:2d
M
Const_24Const*
_output_shapes
: *
valueB
 *    *
dtype0
M
Const_25Const*
valueB
 *  *
dtype0*
_output_shapes
: 
\
clip_by_value_3/MinimumMinimumadd_8Const_25*
_output_shapes

:2d*
T0
f
clip_by_value_3Maximumclip_by_value_3/MinimumConst_24*
T0*
_output_shapes

:2d
H
Sqrt_3Sqrtclip_by_value_3*
T0*
_output_shapes

:2d
L
add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+2
H
add_9AddV2Sqrt_3add_9/y*
_output_shapes

:2d*
T0
L
	truediv_4RealDivmul_17add_9*
T0*
_output_shapes

:2d
V
sub_11Sublayer_2/kernel/read	truediv_4*
_output_shapes

:2d*
T0

Assign_6Assign
Variable_2add_7*
validate_shape(*
T0*
use_locking(*
_output_shapes

:2d*
_class
loc:@Variable_2

Assign_7AssignVariable_10add_8*
use_locking(*
_class
loc:@Variable_10*
validate_shape(*
_output_shapes

:2d*
T0

Assign_8Assignlayer_2/kernelsub_11*!
_class
loc:@layer_2/kernel*
T0*
validate_shape(*
_output_shapes

:2d*
use_locking(
P
mul_18Mulbeta_1/readVariable_3/read*
T0*
_output_shapes
:d
M
sub_12/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
E
sub_12Subsub_12/xbeta_1/read*
T0*
_output_shapes
: 
f
mul_19Mulsub_12*gradients/layer_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:d*
T0
D
add_10AddV2mul_18mul_19*
_output_shapes
:d*
T0
Q
mul_20Mulbeta_2/readVariable_11/read*
_output_shapes
:d*
T0
M
sub_13/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
E
sub_13Subsub_13/xbeta_2/read*
_output_shapes
: *
T0
c
Square_4Square*gradients/layer_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:d*
T0
D
mul_21Mulsub_13Square_4*
_output_shapes
:d*
T0
D
add_11AddV2mul_20mul_21*
_output_shapes
:d*
T0
A
mul_22Mulmul_2add_10*
_output_shapes
:d*
T0
M
Const_26Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_27Const*
_output_shapes
: *
valueB
 *  *
dtype0
Y
clip_by_value_4/MinimumMinimumadd_11Const_27*
T0*
_output_shapes
:d
b
clip_by_value_4Maximumclip_by_value_4/MinimumConst_26*
T0*
_output_shapes
:d
D
Sqrt_4Sqrtclip_by_value_4*
T0*
_output_shapes
:d
M
add_12/yConst*
dtype0*
valueB
 *wΜ+2*
_output_shapes
: 
F
add_12AddV2Sqrt_4add_12/y*
_output_shapes
:d*
T0
I
	truediv_5RealDivmul_22add_12*
T0*
_output_shapes
:d
P
sub_14Sublayer_2/bias/read	truediv_5*
T0*
_output_shapes
:d

Assign_9Assign
Variable_3add_10*
_output_shapes
:d*
_class
loc:@Variable_3*
validate_shape(*
use_locking(*
T0

	Assign_10AssignVariable_11add_11*
use_locking(*
T0*
validate_shape(*
_output_shapes
:d*
_class
loc:@Variable_11

	Assign_11Assignlayer_2/biassub_14*
use_locking(*
T0*
_output_shapes
:d*
validate_shape(*
_class
loc:@layer_2/bias
T
mul_23Mulbeta_1/readVariable_4/read*
_output_shapes

:d2*
T0
M
sub_15/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
E
sub_15Subsub_15/xbeta_1/read*
T0*
_output_shapes
: 
f
mul_24Mulsub_15&gradients/layer_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

:d2
H
add_13AddV2mul_23mul_24*
T0*
_output_shapes

:d2
U
mul_25Mulbeta_2/readVariable_12/read*
T0*
_output_shapes

:d2
M
sub_16/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
E
sub_16Subsub_16/xbeta_2/read*
_output_shapes
: *
T0
c
Square_5Square&gradients/layer_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

:d2
H
mul_26Mulsub_16Square_5*
_output_shapes

:d2*
T0
H
add_14AddV2mul_25mul_26*
_output_shapes

:d2*
T0
E
mul_27Mulmul_2add_13*
_output_shapes

:d2*
T0
M
Const_28Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_29Const*
dtype0*
valueB
 *  *
_output_shapes
: 
]
clip_by_value_5/MinimumMinimumadd_14Const_29*
T0*
_output_shapes

:d2
f
clip_by_value_5Maximumclip_by_value_5/MinimumConst_28*
T0*
_output_shapes

:d2
H
Sqrt_5Sqrtclip_by_value_5*
_output_shapes

:d2*
T0
M
add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+2
J
add_15AddV2Sqrt_5add_15/y*
_output_shapes

:d2*
T0
M
	truediv_6RealDivmul_27add_15*
_output_shapes

:d2*
T0
V
sub_17Sublayer_3/kernel/read	truediv_6*
_output_shapes

:d2*
T0

	Assign_12Assign
Variable_4add_13*
use_locking(*
_class
loc:@Variable_4*
T0*
_output_shapes

:d2*
validate_shape(

	Assign_13AssignVariable_12add_14*
_class
loc:@Variable_12*
T0*
_output_shapes

:d2*
validate_shape(*
use_locking(
 
	Assign_14Assignlayer_3/kernelsub_17*
use_locking(*
T0*
validate_shape(*
_output_shapes

:d2*!
_class
loc:@layer_3/kernel
P
mul_28Mulbeta_1/readVariable_5/read*
T0*
_output_shapes
:2
M
sub_18/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
E
sub_18Subsub_18/xbeta_1/read*
T0*
_output_shapes
: 
f
mul_29Mulsub_18*gradients/layer_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:2*
T0
D
add_16AddV2mul_28mul_29*
T0*
_output_shapes
:2
Q
mul_30Mulbeta_2/readVariable_13/read*
_output_shapes
:2*
T0
M
sub_19/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
E
sub_19Subsub_19/xbeta_2/read*
T0*
_output_shapes
: 
c
Square_6Square*gradients/layer_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:2
D
mul_31Mulsub_19Square_6*
T0*
_output_shapes
:2
D
add_17AddV2mul_30mul_31*
_output_shapes
:2*
T0
A
mul_32Mulmul_2add_16*
T0*
_output_shapes
:2
M
Const_30Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_31Const*
valueB
 *  *
dtype0*
_output_shapes
: 
Y
clip_by_value_6/MinimumMinimumadd_17Const_31*
T0*
_output_shapes
:2
b
clip_by_value_6Maximumclip_by_value_6/MinimumConst_30*
_output_shapes
:2*
T0
D
Sqrt_6Sqrtclip_by_value_6*
_output_shapes
:2*
T0
M
add_18/yConst*
dtype0*
valueB
 *wΜ+2*
_output_shapes
: 
F
add_18AddV2Sqrt_6add_18/y*
T0*
_output_shapes
:2
I
	truediv_7RealDivmul_32add_18*
T0*
_output_shapes
:2
P
sub_20Sublayer_3/bias/read	truediv_7*
_output_shapes
:2*
T0

	Assign_15Assign
Variable_5add_16*
_output_shapes
:2*
validate_shape(*
use_locking(*
_class
loc:@Variable_5*
T0

	Assign_16AssignVariable_13add_17*
T0*
validate_shape(*
_output_shapes
:2*
use_locking(*
_class
loc:@Variable_13

	Assign_17Assignlayer_3/biassub_20*
validate_shape(*
_class
loc:@layer_3/bias*
use_locking(*
_output_shapes
:2*
T0
T
mul_33Mulbeta_1/readVariable_6/read*
_output_shapes

:2*
T0
M
sub_21/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
E
sub_21Subsub_21/xbeta_1/read*
_output_shapes
: *
T0
k
mul_34Mulsub_21+gradients/output_layer/MatMul_grad/MatMul_1*
T0*
_output_shapes

:2
H
add_19AddV2mul_33mul_34*
T0*
_output_shapes

:2
U
mul_35Mulbeta_2/readVariable_14/read*
T0*
_output_shapes

:2
M
sub_22/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
E
sub_22Subsub_22/xbeta_2/read*
T0*
_output_shapes
: 
h
Square_7Square+gradients/output_layer/MatMul_grad/MatMul_1*
T0*
_output_shapes

:2
H
mul_36Mulsub_22Square_7*
T0*
_output_shapes

:2
H
add_20AddV2mul_35mul_36*
T0*
_output_shapes

:2
E
mul_37Mulmul_2add_19*
T0*
_output_shapes

:2
M
Const_32Const*
valueB
 *    *
dtype0*
_output_shapes
: 
M
Const_33Const*
valueB
 *  *
_output_shapes
: *
dtype0
]
clip_by_value_7/MinimumMinimumadd_20Const_33*
_output_shapes

:2*
T0
f
clip_by_value_7Maximumclip_by_value_7/MinimumConst_32*
T0*
_output_shapes

:2
H
Sqrt_7Sqrtclip_by_value_7*
T0*
_output_shapes

:2
M
add_21/yConst*
dtype0*
valueB
 *wΜ+2*
_output_shapes
: 
J
add_21AddV2Sqrt_7add_21/y*
T0*
_output_shapes

:2
M
	truediv_8RealDivmul_37add_21*
_output_shapes

:2*
T0
[
sub_23Suboutput_layer/kernel/read	truediv_8*
_output_shapes

:2*
T0

	Assign_18Assign
Variable_6add_19*
_output_shapes

:2*
T0*
_class
loc:@Variable_6*
validate_shape(*
use_locking(

	Assign_19AssignVariable_14add_20*
_class
loc:@Variable_14*
validate_shape(*
use_locking(*
_output_shapes

:2*
T0
ͺ
	Assign_20Assignoutput_layer/kernelsub_23*
T0*
validate_shape(*&
_class
loc:@output_layer/kernel*
use_locking(*
_output_shapes

:2
P
mul_38Mulbeta_1/readVariable_7/read*
T0*
_output_shapes
:
M
sub_24/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
E
sub_24Subsub_24/xbeta_1/read*
_output_shapes
: *
T0
k
mul_39Mulsub_24/gradients/output_layer/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
D
add_22AddV2mul_38mul_39*
_output_shapes
:*
T0
Q
mul_40Mulbeta_2/readVariable_15/read*
T0*
_output_shapes
:
M
sub_25/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
E
sub_25Subsub_25/xbeta_2/read*
T0*
_output_shapes
: 
h
Square_8Square/gradients/output_layer/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
D
mul_41Mulsub_25Square_8*
T0*
_output_shapes
:
D
add_23AddV2mul_40mul_41*
T0*
_output_shapes
:
A
mul_42Mulmul_2add_22*
_output_shapes
:*
T0
M
Const_34Const*
_output_shapes
: *
valueB
 *    *
dtype0
M
Const_35Const*
_output_shapes
: *
valueB
 *  *
dtype0
Y
clip_by_value_8/MinimumMinimumadd_23Const_35*
_output_shapes
:*
T0
b
clip_by_value_8Maximumclip_by_value_8/MinimumConst_34*
T0*
_output_shapes
:
D
Sqrt_8Sqrtclip_by_value_8*
T0*
_output_shapes
:
M
add_24/yConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+2
F
add_24AddV2Sqrt_8add_24/y*
T0*
_output_shapes
:
I
	truediv_9RealDivmul_42add_24*
T0*
_output_shapes
:
U
sub_26Suboutput_layer/bias/read	truediv_9*
_output_shapes
:*
T0

	Assign_21Assign
Variable_7add_22*
T0*
_class
loc:@Variable_7*
use_locking(*
_output_shapes
:*
validate_shape(

	Assign_22AssignVariable_15add_23*
_output_shapes
:*
T0*
_class
loc:@Variable_15*
validate_shape(*
use_locking(
’
	Assign_23Assignoutput_layer/biassub_26*
_output_shapes
:*$
_class
loc:@output_layer/bias*
validate_shape(*
T0*
use_locking(
Ί

group_depsNoOp^Assign
^AssignAdd	^Assign_1
^Assign_10
^Assign_11
^Assign_12
^Assign_13
^Assign_14
^Assign_15
^Assign_16
^Assign_17
^Assign_18
^Assign_19	^Assign_2
^Assign_20
^Assign_21
^Assign_22
^Assign_23	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8	^Assign_9^mul_1
α
initNoOp^Variable/Assign^Variable_1/Assign^Variable_10/Assign^Variable_11/Assign^Variable_12/Assign^Variable_13/Assign^Variable_14/Assign^Variable_15/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable_9/Assign^beta_1/Assign^beta_2/Assign^decay/Assign^iterations/Assign^layer_1/bias/Assign^layer_1/kernel/Assign^layer_2/bias/Assign^layer_2/kernel/Assign^layer_3/bias/Assign^layer_3/kernel/Assign
^lr/Assign^output_layer/bias/Assign^output_layer/kernel/Assign
e
layer_1/kernel_0/tagConst*
_output_shapes
: *!
valueB Blayer_1/kernel_0*
dtype0
p
layer_1/kernel_0HistogramSummarylayer_1/kernel_0/taglayer_1/kernel/read*
_output_shapes
: *
T0
a
layer_1/bias_0/tagConst*
dtype0*
valueB Blayer_1/bias_0*
_output_shapes
: 
j
layer_1/bias_0HistogramSummarylayer_1/bias_0/taglayer_1/bias/read*
T0*
_output_shapes
: 
[
layer_1_out/tagConst*
dtype0*
valueB Blayer_1_out*
_output_shapes
: 
_
layer_1_outHistogramSummarylayer_1_out/taglayer_1/Relu*
T0*
_output_shapes
: 
e
layer_2/kernel_0/tagConst*
dtype0*
_output_shapes
: *!
valueB Blayer_2/kernel_0
p
layer_2/kernel_0HistogramSummarylayer_2/kernel_0/taglayer_2/kernel/read*
_output_shapes
: *
T0
a
layer_2/bias_0/tagConst*
valueB Blayer_2/bias_0*
dtype0*
_output_shapes
: 
j
layer_2/bias_0HistogramSummarylayer_2/bias_0/taglayer_2/bias/read*
T0*
_output_shapes
: 
[
layer_2_out/tagConst*
_output_shapes
: *
dtype0*
valueB Blayer_2_out
_
layer_2_outHistogramSummarylayer_2_out/taglayer_2/Relu*
T0*
_output_shapes
: 
e
layer_3/kernel_0/tagConst*!
valueB Blayer_3/kernel_0*
_output_shapes
: *
dtype0
p
layer_3/kernel_0HistogramSummarylayer_3/kernel_0/taglayer_3/kernel/read*
T0*
_output_shapes
: 
a
layer_3/bias_0/tagConst*
dtype0*
valueB Blayer_3/bias_0*
_output_shapes
: 
j
layer_3/bias_0HistogramSummarylayer_3/bias_0/taglayer_3/bias/read*
_output_shapes
: *
T0
[
layer_3_out/tagConst*
valueB Blayer_3_out*
_output_shapes
: *
dtype0
_
layer_3_outHistogramSummarylayer_3_out/taglayer_3/Relu*
T0*
_output_shapes
: 
o
output_layer/kernel_0/tagConst*&
valueB Boutput_layer/kernel_0*
dtype0*
_output_shapes
: 

output_layer/kernel_0HistogramSummaryoutput_layer/kernel_0/tagoutput_layer/kernel/read*
_output_shapes
: *
T0
k
output_layer/bias_0/tagConst*$
valueB Boutput_layer/bias_0*
dtype0*
_output_shapes
: 
y
output_layer/bias_0HistogramSummaryoutput_layer/bias_0/tagoutput_layer/bias/read*
_output_shapes
: *
T0
e
output_layer_out/tagConst*
_output_shapes
: *!
valueB Boutput_layer_out*
dtype0
q
output_layer_outHistogramSummaryoutput_layer_out/tagoutput_layer/BiasAdd*
T0*
_output_shapes
: 

Merge/MergeSummaryMergeSummarylayer_1/kernel_0layer_1/bias_0layer_1_outlayer_2/kernel_0layer_2/bias_0layer_2_outlayer_3/kernel_0layer_3/bias_0layer_3_outoutput_layer/kernel_0output_layer/bias_0output_layer_out*
_output_shapes
: *
N

group_deps_1NoOp^mul_1
Y
save/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_f31fd5e62ca948908ae96b95d2fe2074/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
Q
save/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
k
save/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
value	B : *
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
ή
save/SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueψBυBVariableB
Variable_1BVariable_10BVariable_11BVariable_12BVariable_13BVariable_14BVariable_15B
Variable_2B
Variable_3B
Variable_4B
Variable_5B
Variable_6B
Variable_7B
Variable_8B
Variable_9Bbeta_1Bbeta_2BdecayB
iterationsBlayer_1/biasBlayer_1/kernelBlayer_2/biasBlayer_2/kernelBlayer_3/biasBlayer_3/kernelBlrBoutput_layer/biasBoutput_layer/kernel
¬
save/SaveV2/shape_and_slicesConst"/device:CPU:0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:

save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariable
Variable_1Variable_10Variable_11Variable_12Variable_13Variable_14Variable_15
Variable_2
Variable_3
Variable_4
Variable_5
Variable_6
Variable_7
Variable_8
Variable_9beta_1beta_2decay
iterationslayer_1/biaslayer_1/kernellayer_2/biaslayer_2/kernellayer_3/biaslayer_3/kernellroutput_layer/biasoutput_layer/kernel"/device:CPU:0*+
dtypes!
2
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename
¬
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*

axis *
_output_shapes
:*
T0

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
α
save/RestoreV2/tensor_namesConst"/device:CPU:0*
valueψBυBVariableB
Variable_1BVariable_10BVariable_11BVariable_12BVariable_13BVariable_14BVariable_15B
Variable_2B
Variable_3B
Variable_4B
Variable_5B
Variable_6B
Variable_7B
Variable_8B
Variable_9Bbeta_1Bbeta_2BdecayB
iterationsBlayer_1/biasBlayer_1/kernelBlayer_2/biasBlayer_2/kernelBlayer_3/biasBlayer_3/kernelBlrBoutput_layer/biasBoutput_layer/kernel*
dtype0*
_output_shapes
:
―
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
¬
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2

save/AssignAssignVariablesave/RestoreV2*
use_locking(*
_output_shapes

:	2*
validate_shape(*
_class
loc:@Variable*
T0
’
save/Assign_1Assign
Variable_1save/RestoreV2:1*
_output_shapes
:2*
_class
loc:@Variable_1*
T0*
validate_shape(*
use_locking(
¨
save/Assign_2AssignVariable_10save/RestoreV2:2*
validate_shape(*
T0*
_class
loc:@Variable_10*
_output_shapes

:2d*
use_locking(
€
save/Assign_3AssignVariable_11save/RestoreV2:3*
_output_shapes
:d*
T0*
validate_shape(*
_class
loc:@Variable_11*
use_locking(
¨
save/Assign_4AssignVariable_12save/RestoreV2:4*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_12*
_output_shapes

:d2
€
save/Assign_5AssignVariable_13save/RestoreV2:5*
use_locking(*
_output_shapes
:2*
validate_shape(*
_class
loc:@Variable_13*
T0
¨
save/Assign_6AssignVariable_14save/RestoreV2:6*
_output_shapes

:2*
validate_shape(*
use_locking(*
_class
loc:@Variable_14*
T0
€
save/Assign_7AssignVariable_15save/RestoreV2:7*
validate_shape(*
T0*
use_locking(*
_class
loc:@Variable_15*
_output_shapes
:
¦
save/Assign_8Assign
Variable_2save/RestoreV2:8*
_class
loc:@Variable_2*
use_locking(*
T0*
_output_shapes

:2d*
validate_shape(
’
save/Assign_9Assign
Variable_3save/RestoreV2:9*
validate_shape(*
_output_shapes
:d*
T0*
_class
loc:@Variable_3*
use_locking(
¨
save/Assign_10Assign
Variable_4save/RestoreV2:10*
_class
loc:@Variable_4*
use_locking(*
validate_shape(*
_output_shapes

:d2*
T0
€
save/Assign_11Assign
Variable_5save/RestoreV2:11*
T0*
_output_shapes
:2*
validate_shape(*
_class
loc:@Variable_5*
use_locking(
¨
save/Assign_12Assign
Variable_6save/RestoreV2:12*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_6*
_output_shapes

:2
€
save/Assign_13Assign
Variable_7save/RestoreV2:13*
validate_shape(*
T0*
use_locking(*
_class
loc:@Variable_7*
_output_shapes
:
¨
save/Assign_14Assign
Variable_8save/RestoreV2:14*
_output_shapes

:	2*
validate_shape(*
T0*
_class
loc:@Variable_8*
use_locking(
€
save/Assign_15Assign
Variable_9save/RestoreV2:15*
T0*
_output_shapes
:2*
use_locking(*
_class
loc:@Variable_9*
validate_shape(

save/Assign_16Assignbeta_1save/RestoreV2:16*
_class
loc:@beta_1*
validate_shape(*
T0*
use_locking(*
_output_shapes
: 

save/Assign_17Assignbeta_2save/RestoreV2:17*
validate_shape(*
_class
loc:@beta_2*
use_locking(*
_output_shapes
: *
T0

save/Assign_18Assigndecaysave/RestoreV2:18*
_class

loc:@decay*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
 
save/Assign_19Assign
iterationssave/RestoreV2:19*
use_locking(*
_class
loc:@iterations*
validate_shape(*
T0*
_output_shapes
: 
¨
save/Assign_20Assignlayer_1/biassave/RestoreV2:20*
_class
loc:@layer_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:2*
T0
°
save/Assign_21Assignlayer_1/kernelsave/RestoreV2:21*
_output_shapes

:	2*
T0*
validate_shape(*
use_locking(*!
_class
loc:@layer_1/kernel
¨
save/Assign_22Assignlayer_2/biassave/RestoreV2:22*
use_locking(*
_output_shapes
:d*
T0*
validate_shape(*
_class
loc:@layer_2/bias
°
save/Assign_23Assignlayer_2/kernelsave/RestoreV2:23*!
_class
loc:@layer_2/kernel*
_output_shapes

:2d*
use_locking(*
T0*
validate_shape(
¨
save/Assign_24Assignlayer_3/biassave/RestoreV2:24*
T0*
_class
loc:@layer_3/bias*
use_locking(*
_output_shapes
:2*
validate_shape(
°
save/Assign_25Assignlayer_3/kernelsave/RestoreV2:25*
_output_shapes

:d2*!
_class
loc:@layer_3/kernel*
use_locking(*
validate_shape(*
T0

save/Assign_26Assignlrsave/RestoreV2:26*
T0*
_class
	loc:@lr*
use_locking(*
_output_shapes
: *
validate_shape(
²
save/Assign_27Assignoutput_layer/biassave/RestoreV2:27*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*$
_class
loc:@output_layer/bias
Ί
save/Assign_28Assignoutput_layer/kernelsave/RestoreV2:28*
_output_shapes

:2*
T0*
validate_shape(*&
_class
loc:@output_layer/kernel*
use_locking(
ϋ
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"
	variablesυς
\
layer_1/kernel:0layer_1/kernel/Assignlayer_1/kernel/read:02layer_1/random_uniform:08
M
layer_1/bias:0layer_1/bias/Assignlayer_1/bias/read:02layer_1/Const:08
\
layer_2/kernel:0layer_2/kernel/Assignlayer_2/kernel/read:02layer_2/random_uniform:08
M
layer_2/bias:0layer_2/bias/Assignlayer_2/bias/read:02layer_2/Const:08
\
layer_3/kernel:0layer_3/kernel/Assignlayer_3/kernel/read:02layer_3/random_uniform:08
M
layer_3/bias:0layer_3/bias/Assignlayer_3/bias/read:02layer_3/Const:08
p
output_layer/kernel:0output_layer/kernel/Assignoutput_layer/kernel/read:02output_layer/random_uniform:08
a
output_layer/bias:0output_layer/bias/Assignoutput_layer/bias/read:02output_layer/Const:08
R
iterations:0iterations/Assigniterations/read:02iterations/initial_value:08
2
lr:0	lr/Assign	lr/read:02lr/initial_value:08
B
beta_1:0beta_1/Assignbeta_1/read:02beta_1/initial_value:08
B
beta_2:0beta_2/Assignbeta_2/read:02beta_2/initial_value:08
>
decay:0decay/Assigndecay/read:02decay/initial_value:08
;

Variable:0Variable/AssignVariable/read:02	Const_4:08
A
Variable_1:0Variable_1/AssignVariable_1/read:02	Const_5:08
A
Variable_2:0Variable_2/AssignVariable_2/read:02	Const_6:08
A
Variable_3:0Variable_3/AssignVariable_3/read:02	Const_7:08
A
Variable_4:0Variable_4/AssignVariable_4/read:02	Const_8:08
A
Variable_5:0Variable_5/AssignVariable_5/read:02	Const_9:08
B
Variable_6:0Variable_6/AssignVariable_6/read:02
Const_10:08
B
Variable_7:0Variable_7/AssignVariable_7/read:02
Const_11:08
B
Variable_8:0Variable_8/AssignVariable_8/read:02
Const_12:08
B
Variable_9:0Variable_9/AssignVariable_9/read:02
Const_13:08
E
Variable_10:0Variable_10/AssignVariable_10/read:02
Const_14:08
E
Variable_11:0Variable_11/AssignVariable_11/read:02
Const_15:08
E
Variable_12:0Variable_12/AssignVariable_12/read:02
Const_16:08
E
Variable_13:0Variable_13/AssignVariable_13/read:02
Const_17:08
E
Variable_14:0Variable_14/AssignVariable_14/read:02
Const_18:08
E
Variable_15:0Variable_15/AssignVariable_15/read:02
Const_19:08"τ
	summariesζ
γ
layer_1/kernel_0:0
layer_1/bias_0:0
layer_1_out:0
layer_2/kernel_0:0
layer_2/bias_0:0
layer_2_out:0
layer_3/kernel_0:0
layer_3/bias_0:0
layer_3_out:0
output_layer/kernel_0:0
output_layer/bias_0:0
output_layer_out:0"
trainable_variablesυς
\
layer_1/kernel:0layer_1/kernel/Assignlayer_1/kernel/read:02layer_1/random_uniform:08
M
layer_1/bias:0layer_1/bias/Assignlayer_1/bias/read:02layer_1/Const:08
\
layer_2/kernel:0layer_2/kernel/Assignlayer_2/kernel/read:02layer_2/random_uniform:08
M
layer_2/bias:0layer_2/bias/Assignlayer_2/bias/read:02layer_2/Const:08
\
layer_3/kernel:0layer_3/kernel/Assignlayer_3/kernel/read:02layer_3/random_uniform:08
M
layer_3/bias:0layer_3/bias/Assignlayer_3/bias/read:02layer_3/Const:08
p
output_layer/kernel:0output_layer/kernel/Assignoutput_layer/kernel/read:02output_layer/random_uniform:08
a
output_layer/bias:0output_layer/bias/Assignoutput_layer/bias/read:02output_layer/Const:08
R
iterations:0iterations/Assigniterations/read:02iterations/initial_value:08
2
lr:0	lr/Assign	lr/read:02lr/initial_value:08
B
beta_1:0beta_1/Assignbeta_1/read:02beta_1/initial_value:08
B
beta_2:0beta_2/Assignbeta_2/read:02beta_2/initial_value:08
>
decay:0decay/Assigndecay/read:02decay/initial_value:08
;

Variable:0Variable/AssignVariable/read:02	Const_4:08
A
Variable_1:0Variable_1/AssignVariable_1/read:02	Const_5:08
A
Variable_2:0Variable_2/AssignVariable_2/read:02	Const_6:08
A
Variable_3:0Variable_3/AssignVariable_3/read:02	Const_7:08
A
Variable_4:0Variable_4/AssignVariable_4/read:02	Const_8:08
A
Variable_5:0Variable_5/AssignVariable_5/read:02	Const_9:08
B
Variable_6:0Variable_6/AssignVariable_6/read:02
Const_10:08
B
Variable_7:0Variable_7/AssignVariable_7/read:02
Const_11:08
B
Variable_8:0Variable_8/AssignVariable_8/read:02
Const_12:08
B
Variable_9:0Variable_9/AssignVariable_9/read:02
Const_13:08
E
Variable_10:0Variable_10/AssignVariable_10/read:02
Const_14:08
E
Variable_11:0Variable_11/AssignVariable_11/read:02
Const_15:08
E
Variable_12:0Variable_12/AssignVariable_12/read:02
Const_16:08
E
Variable_13:0Variable_13/AssignVariable_13/read:02
Const_17:08
E
Variable_14:0Variable_14/AssignVariable_14/read:02
Const_18:08
E
Variable_15:0Variable_15/AssignVariable_15/read:02
Const_19:08*
serving_default
/
input&
layer_1_input:0?????????	9
earnings-
output_layer/BiasAdd:0?????????tensorflow/serving/predict