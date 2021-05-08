# Basics of TensorFlow

## Installation

* For GPU Based
	```
	pip install tensorflow-gpu==1.14.0
	```
* For CPU Based
	```
	pip install tensorflow-cpu==1.14.0
	```
or
	```
	pip install tensorflow==1.14.0
	```
	
## Important Modules
* tensor flow - Primary container, accessed as `tf`
* `tf.nn` - Classes related to neural networks
* `tf.training` - Functions needed to refine models
* `tf.contrib` - New and volatile features
* `tf.estimators` - Classes in the Estimator API

## Tensor
* N-dimensional arrays of a base datatype
	* May contain numbers, string, or Boolean values
	* Similar to NumPy's ndarrays
* Shape defines lengths of a tensor's dimensions
	* [3] - Vectors containing three values
	* [2,4] - Matrix containing 2 rows and four columns
* Tensor Data Types 
	* Boolean - `tf.bool`
	* Unsigned integer - `tf.uint8`/`tf.unit16`
	* Signed integer - `tf.int8`/`tf.int32`/`tf.int64`
	* Floating-point - `tf.float32`/`tf.float64`
	* Strings - `tf.string`

### Creating Constant Tensor
* Call `tf.constant` with a list of values
	```
	t1 = tf.constant([1.5,2.5,3.5])
	t2 = tf.constant([['a','b','c','d'],['e','f','g','h']])
	t3 = tf.constant([[True,False],[False,True]])
	t4 = tf.constant([[1,3],[5,7]],dtype=tf.unit8)
	``` 
* Normal Distribution
	```
	tf.random_normal(shape,mean=0.0,stddev=1.0) #
	tf.truncated_normal(shape,mean=0.0,stddev=1.0) # Values stand between 2 standard deviation of the mean
	``` 
* Uniform Distribution
	```
	tf.random_uniform(shape,minval=0,maxval=None)
	``` 

### Tensor Operations

#### Shape and Reshape

* `tf.shape` provides a tensor's shape
	```
	t1 = tf.constant([[1,2,4],[4,5,6]])
	tf.shape(t1)
	```
	Returns [2,3]

* `tf.reshape` updates a tensor's shape
	```
	t2 = tf.reshape(t1,[6])
	tf.shape(t2)
	```
	Returns [6]

#### Exacting Data from a Tensor
* `tf.slice(tensor,begin,size)`
	* extracts a subtensor or slice
	* begin - first index to extract
	* size - size of subtensor to extract
	```
	t1 = tf.constant([[1,2,4],[4,5,6],[7,8,9]])
	t2 = tf.slice(t1,[1,1],[2,2])
	```
	Returns [[5,6],[8,9]]

#### Math/Rounding Operations
* Arithmetic operations: two forms
	```
	tf.add(t1,t2) is equivalent to t1 + t2
	```
* Exponential/Logarithmic functions
	```
	tf.exp(t1)
	tf.log(t1)
	tf.erfc(t1) # Complimentary Error Function
	```
* Rounding Functions
	```
	tf.round(t1)
	tf.ceil(t1)
	tf.floor(t1)
	```
#### Comparison
* Maximum and Minimum of a Tensor
	```
	tf.maximum(t1,t2)
	tf.minimum(t1,t2)
	```
* Index of Maximum and Minimum of a Tensor
	`tf.argmax([0,-2,4,1])` - Returns 1
	`tf.argmin([0,-2,4,1])` - Returns 2

#### Vector/Matrix Operations
* Dot Product
	`tf.tensordot(t1,t2,axes)`

* Matrix Multiplication
	`tf.matmul(t1,t2)` - Returns t1.t2

* Matrix Solution
	`tf.matrix_solve(t1,t2)` - Returns x such that t1.x = t2

#### Reduction Operations
* `tf.reduce_sum(t1)` - Returns the sum of the elements
* `tf.reduce_mean(t1)` - Returns the average of the elements
* `tf.reduce_prod(t1)` - Returns the product of the elements
* `tf.reduce_max(t1)/tf.reduce_min(t1)` - Returns the maximum/minimum of the elements

#### Graphs
* `tf.get_default_graph()` - Returns current graph
* `graph = tf.Graph()` - Create new graph
* `graph.as_default()` - Changes the current graph

* Access graph elements:
	```
	graph.get_tensor_by_name(tensor_name)
	graph.get_operation_by_name(op_name)
	```
#### Sessions
* Create a session: `tf.Session()`
	* `tf.Session()` - makes session active
	* Accepts graph parameter to identify the graph
	* Session block: `with tf.Session() as sess:`
* Running a Session
	* First parameter of sess.run()
		* Accepts a tensor, operation, or tensor/operation list
		* For each tensor, sess.run returns a NumPy array
```
with tf.Session() as sess:
	result1,result2 = sess.run([t1,t2])
	print(result1)
```
Displays NumPy array

### Training
#### Variables
* Creating variables: 
	```
	tf.Variable()
	```
* Variables Needs to be Initialized
	```
	init_op1 = tf.variable_initializer([v1,v2])
	init_op2 = tf.local_variable_initializer()
	init_op3 = tf.global_variable_initializer()
	```
* Running 
	```
	sess.run(init_op3)
	```
#### Logging
* Five severity levels
	* debug
	* info
	* warn
	* error
	* fatal
* Enable logging
	```
	tf.logging.set_verbosity(tf.logging.INFO)
	```
* Print message to log
	```
	tf.logging.info('Output message')
	```
#### Optimizer
* Creating and Using an Optimizer
	```
	optimizer = tf.train.GradientDescentOptimizer(learn_rate)
	op = optimizer.minimize(loss)
	with tf.Session() as sess:
			for step in range(num_steps):
				loss = sess.run(op)
	```
* Optimizer Algorithm
	* Momentum algorithm (MomentumOptimizer)
		* Uses past values of the loss gradient
		* Accelerates/decelerates descent as needed
	* Adaptive gradient algorithm (AdagradOptimizer)
		* Adaptive learning rates for variable
		* Can be used with non-differentiable loss functions
	* Adam (AdagradOptimizer and MomentumOptimizer)
		* Both features of Adaptive gradient and Momentum algorithm


#### Batches
* Split training data set into subsets called batches
	* Each training step process one batch
	* Batch size is determined by trail and error
* Batch shuffling
	* Increases likelihood of finding global minimum
	* Stochastic Gradient Descent (SGD) algorithm

#### Placeholders
* Tensors that receive different batches of data
	* Created with `tf.placeholder(dtype,shape=None)` 
	* No initial values - values set by session
* Feeding data to placeholder
	* Second argument of `sess.run(feed_dict)` 
	* Dictionary that associates placeholder with data
```
holder = tf.placeholder(tf.float32,shape[100,10])
with tf.Session() as sess:
		for _ in range(num_batches):
				batch_data = np.array(...)
				sess.run(op,feed_dict={holder:batch_data})
```
#### TensorBoard
* Generating Summary Data
	* `tf.summary.scalar(name,tensor)` - Generates summary data for a single scalar value
	* `tf.summary.histogram(name,tensor)` - Generates summary data for a series of values
	* `tf.summary.image(name,tensor)` - Generates summary data for an image
	* `tf.summary.merge_all()` - Merge operations
* Print Summary Data
	* Create a FileWriter
		* `tf.summary.FileWriter(log_dir)` - log_dir identifies the directory to contain data
	* Print Data
		* `add_summary(data,global_step=None)` - global_step identifies the training step

### Datasets
* Sequence of elements in which each element contains one or more tensor objects
#### Creating Datasets
##### Range of Values
* `tf.data.Dataset.range` - Creates a dataset containing a range of values
	* `ds1 = tf.data.Dataset.range(5)` - Returns [1, 2, 3, 4]
	* `ds2 = tf.data.Dataset.range(10,13)` - Returns [10, 11, 12]
	* `ds3 = tf.data.Dataset.range(2,8,2)` - Returns [2, 4, 6]
##### Multiple Tensors
* `tf.data.Dataset.from_tensors` - Creates a dataset with a single element containing the tensors' element
	```
	t1 = tf.constant([1,2])
	t2 = tf.constant([3,4])
	ds = tf.data.Dataset.from_tensors([t1,t2])
	```
	Returns [[1,2],[3,4]]
##### Text Files
* `tf.data.TextLineDataset`
	* filenames - List of text files
	* compression_type - "","GZIP","ZLIB"
	* buffer_size - desired data size
```
ds = tf.data.TextLineDataset(["file1","file2"],"GZIP")
```
##### TFRecord Files
* `tf.data.TFRecordDataset`
	* filenames - List of text files
	* compression_type - "","GZIP","ZLIB"
	* buffer_size - desired data size
	* num_parallel_reads - Used to read remote data
#### Simple Operations
* `take(count)` - Returns a dataset with the first count elements  
* `skip(count)` - Returns a dataset with all but the first count elements
* `repeat(count=None)` - Repeats the dataset's element the given number of times

	`ds1 = tf.data.Dataset.range(5)` - Returns [0,1,2,3,4]

	`ds2 = ds1.take(2)` - Returns [0,1]

	`ds3 = ds1.skip(2)` - Returns [2,3,4]

	`ds4 = ds1.repeat()` - Returns [0,1,2,3,4,0,1,2,3,4]

* `concatenate(dataset)` - Returns a concatenation of the dataset with given dataset
* `batch(batch_size)` - Splits a dataset into elements of the given size
* `shuffle(buffer_size,seed=None)` - Shuffles the given number of values and returns them in the dataset
		
	`ds1 = tf.data.Dataset.range(3)` - Returns [0,1,2]

	`ds2 = tf.data.Dataset.range(3,6)` - Returns [3,4,5]

	`ds3 = ds1.concatenate(ds2)` - Returns [0,1,2,3,4,5]

	`ds4 = ds3.batch(3)` - Returns [[0,1,2],[3,4,5]]

	`ds5 = ds3.shuffle(6)` - Returns [2,3,4,1,0,5]
	
#### Transformations
* `filter(func)` - Function returns a bool that identifies if the element should be removed
* `map(func)` - Processes each element of input dataset and returns the values as an element in the output dataset

	`ds1 = tf.data.Dataset.range(6)` - Returns [0,1,2,3,4,5]

	`ds2 = ds1.filter(lambda x: x > 2)` - Returns [3,4,5]
	
	`def map_func(x):return x*x`
	
	`ds3 = ds1.map(map_func)` - Returns [0,1,4,9,16,25]

