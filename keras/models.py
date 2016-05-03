from __future__ import print_function
import warnings
import copy

from . import backend as K
from .engine.training import Model
from .engine.topology import get_source_inputs, Node
from .legacy.models import Graph


def model_from_config(config, custom_objects={}):
    from keras.utils.layer_utils import layer_from_config
    return layer_from_config(config, custom_objects=custom_objects)


def model_from_yaml(yaml_string, custom_objects={}):
    '''Parses a yaml model configuration file
    and returns a model instance.
    '''
    import yaml
    from keras.utils.layer_utils import layer_from_config
    config = yaml.load(yaml_string)
    return layer_from_config(config, custom_objects=custom_objects)


def model_from_json(json_string, custom_objects={}):
    '''Parses a JSON model configuration file
    and returns a model instance.
    '''
    import json
    from keras.utils.layer_utils import layer_from_config
    config = json.loads(json_string)
    return layer_from_config(config, custom_objects=custom_objects)


class Sequential(Model):
    '''Linear stack of layers.

    # Arguments
        layers: list of layers to add to the model.

    # Note
        The first layer passed to a Sequential model
        should have a defined input shape. What that
        means is that it should have received an `input_shape`
        or `batch_input_shape` argument,
        or for some type of layers (recurrent, Dense...)
        an `input_dim` argument.

    # Example

        ```python
            model = Sequential()
            # first layer must have a defined input shape
            model.add(Dense(32, input_dim=500))
            # afterwards, Keras does automatic shape inference
            model.add(Dense(32))

            # also possible (equivalent to the above):
            model = Sequential()
            model.add(Dense(32, input_shape=(500,)))
            model.add(Dense(32))

            # also possible (equivalent to the above):
            model = Sequential()
            # here the batch dimension is None,
            # which means any batch size will be accepted by the model.
            model.add(Dense(32, batch_input_shape=(None, 500)))
            model.add(Dense(32))
        ```
    '''
    def __init__(self, layers=[], name=None):
        self.layers = []  # stack of layers
        self.model = None  # internal Model instance
        self.inputs = []  # tensors
        self.outputs = []  # tensors (length 1)

        # model attributes
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.built = False

        if not name:
            prefix = 'sequential_'
            name = prefix + str(K.get_uid(prefix))
        self.name = name

        for layer in layers:
            self.add(layer)

    def add(self, layer):
        '''Adds a layer instance on top of the layer stack.

        # Arguments
            layer: layer instance.
        '''
        if not self.outputs:
            # first layer in model: check that it is an input layer
            if len(layer.inbound_nodes) == 0:
                # create an input layer
                if not hasattr(layer, 'batch_input_shape'):
                    raise Exception('The first layer in a Sequential model must '
                                    'get an `input_shape` or '
                                    '`batch_input_shape` argument.')
                batch_input_shape = layer.batch_input_shape
                if hasattr(layer, 'input_dtype'):
                    input_dtype = layer.input_dtype
                else:
                    input_dtype = None
                layer.create_input_layer(batch_input_shape, input_dtype)

            if len(layer.inbound_nodes) != 1:
                raise Exception('A layer added to a Sequential model must '
                                'not already be connected somewhere else. '
                                'Model received layer ' + layer.name +
                                ' which has ' + str(len(layer.inbound_nodes)) +
                                ' pre-existing inbound connections.')

            if len(layer.inbound_nodes[0].output_tensors) != 1:
                raise Exception('All layers in a Sequential model '
                                'should have a single output tensor. '
                                'For multi-output layers, '
                                'use the functional API.')

            self.outputs = [layer.inbound_nodes[0].output_tensors[0]]
            self.inputs = get_source_inputs(self.outputs[0])

            # We create an input node, which we will keep updated
            # as we add more layers
            Node(outbound_layer=self,
                 inbound_layers=[],
                 node_indices=[],
                 tensor_indices=[],
                 input_tensors=self.inputs,
                 output_tensors=self.outputs,
                 # no model-level masking for now
                 input_masks=[None for _ in self.inputs],
                 output_masks=[None],
                 input_shapes=[x._keras_shape for x in self.inputs],
                 output_shapes=[self.outputs[0]._keras_shape])
        else:
            output_tensor = layer(self.outputs[0])
            if type(output_tensor) is list:
                raise Exception('All layers in a Sequential model '
                                'should have a single output tensor. '
                                'For multi-output layers, '
                                'use the functional API.')
            self.outputs = [output_tensor]
            # update self.inbound_nodes
            self.inbound_nodes[0].output_tensors = self.outputs
            self.inbound_nodes[0].output_shapes = [self.outputs[0]._keras_shape]

        self.layers.append(layer)
        self.built = False

    def call(self, x, mask=None):
        if not self.built:
            self.build()
        return self.model.call(x, mask)

    def build(self, input_shape=None):
        if not self.inputs or not self.outputs:
            raise Exception('Sequential model cannot be built: model is empty.'
                            ' Add some layers first.')
        # actually create the model
        self.model = Model(self.inputs, self.outputs[0], name=self.name + '_model')

        # mirror model attributes
        self.supports_masking = self.model.supports_masking
        self._output_mask_cache = self.model._output_mask_cache
        self._output_tensor_cache = self.model._output_tensor_cache
        self._output_shape_cache = self.model._output_shape_cache
        self.input_layers = self.model.input_layers
        self.input_layers_node_indices = self.model.input_layers_node_indices
        self.input_layers_tensor_indices = self.model.input_layers_tensor_indices
        self.output_layers = self.model.output_layers
        self.output_layers_node_indices = self.model.output_layers_node_indices
        self.output_layers_tensor_indices = self.model.output_layers_tensor_indices
        self.nodes_by_depth = self.model.nodes_by_depth
        self.container_nodes = self.model.container_nodes
        self.output_names = self.model.output_names
        self.input_names = self.model.input_names

        # make sure child model callbacks will call the parent Sequential model:
        self.model.callback_model = self

        self.built = True

    @property
    def uses_learning_phase(self):
        if not self.built:
            self.build()
        return self.model.uses_learning_phase

    @property
    def flattened_layers(self):
        layers = []
        if self.layers[0].__class__.__name__ == 'Merge':
            merge = self.layers[0]
            for layer in merge.layers:
                if hasattr(layer, 'flattened_layers'):
                    for sublayer in layer.flattened_layers:
                        if sublayer not in layers:
                            layers.append(sublayer)
                elif hasattr(layer, 'layers'):
                    for sublayer in layer.layers:
                        if sublayer not in layers:
                            layers.append(sublayer)
                else:
                    if layer not in layers:
                        layers.append(layer)
        else:
            if self.layers[0] not in layers:
                layers.append(self.layers[0])
        for layer in self.layers[1:]:
            if layer not in layers:
                layers.append(layer)
        return layers

    def _gather_list_attr(self, attr):
        all_attrs = []
        for layer in self.flattened_layers:
            all_attrs += getattr(layer, attr, [])
        return all_attrs

    def _gather_dict_attr(self, attr):
        all_attrs = {}
        for layer in self.flattened_layers:
            layer_dict = getattr(layer, attr, {})
            all_attrs = dict(list(all_attrs.items()) +
                             list(layer_dict.items()))
        return all_attrs

    @property
    def trainable_weights(self):
        # support for legacy behavior
        return self._gather_list_attr('trainable_weights')

    @property
    def non_trainable_weights(self):
        # support for legacy behavior
        return self._gather_list_attr('non_trainable_weights')

    @property
    def updates(self):
        # support for legacy behavior
        return self._gather_list_attr('updates')

    @property
    def state_updates(self):
        # support for legacy behavior
        return self._gather_list_attr('state_updates')

    @property
    def regularizers(self):
        # support for legacy behavior
        return self._gather_list_attr('regularizers')

    @property
    def constraints(self):
        # support for legacy behavior
        return self._gather_dict_attr('constraints')

    def get_weights(self):
        '''Returns the weights of the model,
        as a flat list of Numpy arrays.
        '''
        # support for legacy behavior
        weights = []
        for layer in self.flattened_layers:
            weights += layer.get_weights()
        return weights

    def set_weights(self, weights):
        '''Sets the weights of the model.
        The `weights` argument should be a list
        of Numpy arrays with shapes and types matching
        the output of `model.get_weights()`.
        '''
        # support for legacy behavior
        for layer in self.flattened_layers:
            nb_param = len(layer.get_weights())
            layer.set_weights(weights[:nb_param])
            weights = weights[nb_param:]

    @property
    def validation_data(self):
        return self.model.validation_data

    @property
    def training_data(self):
        return self.model.training_data

    def compile(self, optimizer, loss,
                metrics=[],
                sample_weight_mode=None,
                **kwargs):
        '''Configures the learning process.

        # Arguments
            optimizer: str (name of optimizer) or optimizer object.
                See [optimizers](/optimizers).
            loss: str (name of objective function) or objective function.
                See [objectives](/objectives).
            metrics: list of metrics to be evaluated by the model
                during training and testing.
                Typically you will use `metrics=['accuracy']`.
            sample_weight_mode: if you need to do timestep-wise
                sample weighting (2D weights), set this to "temporal".
                "None" defaults to sample-wise weights (1D).
            kwargs: for Theano backend, these are passed into K.function.
                Ignored for Tensorflow backend.

        # Example
            ```python
                model = Sequential()
                model.add(Dense(32, input_shape=(500,)))
                model.add(Dense(10, activation='softmax'))
                model.compile(optimizer='rmsprop',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
            ```
        '''
        # create the underlying model
        self.build()
        # legacy kwarg support
        if 'class_mode' in kwargs:
            warnings.warn('"class_mode" argument is deprecated, '
                          'please remove it.')
            kwargs.pop('class_mode')
        # call compile method of Model class
        self.model.compile(optimizer, loss,
                           metrics=metrics,
                           sample_weight_mode=sample_weight_mode,
                           **kwargs)
        self.optimizer = self.model.optimizer
        self.loss = self.model.loss
        self.metrics_names = self.model.metrics_names
        self.sample_weight_mode = self.model.sample_weight_mode

    def fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[],
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, **kwargs):
        '''Trains the model for a fixed number of epochs.

        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).
            y: labels, as a Numpy array.
            batch_size: integer. Number of samples per gradient update.
            nb_epoch: integer, the number of epochs to train the model.
            verbose: 0 for no logging to stdout,
                1 for progress bar logging, 2 for one log line per epoch.
            callbacks: list of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See [callbacks](/callbacks).
            validation_split: float (0. < x < 1).
                Fraction of the data to use as held-out validation data.
            validation_data: tuple (X, y) to be used as held-out
                validation data. Will override validation_split.
            shuffle: boolean or str (for 'batch').
                Whether to shuffle the samples at each epoch.
                'batch' is a special option for dealing with the
                limitations of HDF5 data; it shuffles in batch-sized chunks.
            class_weight: dictionary mapping classes to a weight value,
                used for scaling the loss function (during training only).
            sample_weight: Numpy array of weights for
                the training samples, used for scaling the loss function
                (during training only). You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape (samples, sequence_length),
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                sample_weight_mode="temporal" in compile().

        # Returns
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).
        '''
        if self.model is None:
            raise Exception('The model needs to be compiled before being used.')
        if 'show_accuracy' in kwargs:
            kwargs.pop('show_accuracy')
            warnings.warn('The "show_accuracy" argument is deprecated, '
                          'instead you should pass the "accuracy" metric to '
                          'the model at compile time:\n'
                          '`model.compile(optimizer, loss, '
                          'metrics=["accuracy"])`')
        if kwargs:
            raise Exception('Received unknown keyword arguments: ' +
                            str(kwargs))
        return self.model.fit(x, y,
                              batch_size=batch_size,
                              nb_epoch=nb_epoch,
                              verbose=verbose,
                              callbacks=callbacks,
                              validation_split=validation_split,
                              validation_data=validation_data,
                              shuffle=shuffle,
                              class_weight=class_weight,
                              sample_weight=sample_weight)

    def evaluate(self, x, y, batch_size=32, verbose=1,
                 sample_weight=None, **kwargs):
        '''Computes the loss on some input data, batch by batch.

        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).
            y: labels, as a Numpy array.
            batch_size: integer. Number of samples per gradient update.
            verbose: verbosity mode, 0 or 1.
            sample_weight: sample weights, as a Numpy array.

        # Returns
            Scalar test loss (if the model has no metrics)
            or list of scalars (if the model computes other metrics).
            The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        '''
        if self.model is None:
            raise Exception('The model needs to be compiled before being used.')
        if 'show_accuracy' in kwargs:
            kwargs.pop('show_accuracy')
            warnings.warn('The "show_accuracy" argument is deprecated, '
                          'instead you should pass the "accuracy" metric to '
                          'the model at compile time:\n'
                          '`model.compile(optimizer, loss, '
                          'metrics=["accuracy"])`')
        if kwargs:
            raise Exception('Received unknown keyword arguments: ' +
                            str(kwargs))
        return self.model.evaluate(x, y,
                                   batch_size=batch_size,
                                   verbose=verbose,
                                   sample_weight=sample_weight)

    def predict(self, x, batch_size=32, verbose=0):
        '''Generates output predictions for the input samples,
        processing the samples in a batched way.

        # Arguments
            x: the input data, as a Numpy array.
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.

        # Returns
            A Numpy array of predictions.
        '''
        if self.model is None:
            raise Exception('The model needs to be compiled before being used.')
        return self.model.predict(x, batch_size=batch_size, verbose=verbose)

    def predict_on_batch(self, x):
        '''Returns predictions for a single batch of samples.
        '''
        return self.model.predict_on_batch(x)

    def train_on_batch(self, x, y, class_weight=None,
                       sample_weight=None, **kwargs):
        '''Single gradient update over one batch of samples.

        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).
            y: labels, as a Numpy array.
            class_weight: dictionary mapping classes to a weight value,
                used for scaling the loss function (during training only).
            sample_weight: sample weights, as a Numpy array.

        # Returns
            Scalar training loss (if the model has no metrics)
            or list of scalars (if the model computes other metrics).
            The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        '''
        if 'accuracy' in kwargs:
            kwargs.pop('accuracy')
            warnings.warn('The "accuracy" argument is deprecated, '
                          'instead you should pass the "accuracy" metric to '
                          'the model at compile time:\n'
                          '`model.compile(optimizer, loss, '
                          'metrics=["accuracy"])`')
        if kwargs:
            raise Exception('Received unknown keyword arguments: ' +
                            str(kwargs))
        return self.model.train_on_batch(x, y,
                                         sample_weight=sample_weight,
                                         class_weight=class_weight)

    def test_on_batch(self, x, y,
                      sample_weight=None, **kwargs):
        '''Evaluates the model over a single batch of samples.

        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).
            y: labels, as a Numpy array.
            sample_weight: sample weights, as a Numpy array.

        # Returns
            Scalar test loss (if the model has no metrics)
            or list of scalars (if the model computes other metrics).
            The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        '''
        if 'accuracy' in kwargs:
            kwargs.pop('accuracy')
            warnings.warn('The "accuracy" argument is deprecated, '
                          'instead you should pass the "accuracy" metric to '
                          'the model at compile time:\n'
                          '`model.compile(optimizer, loss, '
                          'metrics=["accuracy"])`')
        if kwargs:
            raise Exception('Received unknown keyword arguments: ' +
                            str(kwargs))
        return self.model.test_on_batch(x, y,
                                        sample_weight=sample_weight)

    def predict_proba(self, x, batch_size=32, verbose=1):
        '''Generates class probability predictions for the input samples
        batch by batch.

        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.

        # Returns
            A Numpy array of probability predictions.
        '''
        if self.model is None:
            raise Exception('The model needs to be compiled before being used.')
        preds = self.predict(x, batch_size, verbose)
        if preds.min() < 0. or preds.max() > 1.:
            warnings.warn('Network returning invalid probability values. '
                          'The last layer might not normalize predictions '
                          'into probabilities '
                          '(like softmax or sigmoid would).')
        return preds

    def predict_classes(self, x, batch_size=32, verbose=1):
        '''Generate class predictions for the input samples
        batch by batch.

        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.

        # Returns
            A numpy array of class predictions.
        '''
        if self.model is None:
            raise Exception('The model needs to be compiled before being used.')
        proba = self.predict(x, batch_size=batch_size, verbose=verbose)
        if proba.shape[-1] > 1:
            return proba.argmax(axis=-1)
        else:
            return (proba > 0.5).astype('int32')

    def fit_generator(self, generator, samples_per_epoch, nb_epoch,
                      verbose=1, callbacks=[],
                      validation_data=None, nb_val_samples=None,
                      class_weight=None, max_q_size=10, **kwargs):
        '''Fits the model on data generated batch-by-batch by
        a Python generator.
        The generator is run in parallel to the model, for efficiency.
        For instance, this allows you to do real-time data augmentation
        on images on CPU in parallel to training your model on GPU.

        # Arguments
            generator: a generator.
                The output of the generator must be either
                - a tuple (inputs, targets)
                - a tuple (inputs, targets, sample_weights).
                All arrays should contain the same number of samples.
                The generator is expected to loop over its data
                indefinitely. An epoch finishes when `samples_per_epoch`
                samples have been seen by the model.
            samples_per_epoch: integer, number of samples to process before
                going to the next epoch.
            nb_epoch: integer, total number of iterations on the data.
            verbose: verbosity mode, 0, 1, or 2.
            callbacks: list of callbacks to be called during training.
            validation_data: this can be either
                - a generator for the validation data
                - a tuple (inputs, targets)
                - a tuple (inputs, targets, sample_weights).
            nb_val_samples: only relevant if `validation_data` is a generator.
                number of samples to use from validation generator
                at the end of every epoch.
            class_weight: dictionary mapping class indices to a weight
                for the class.
            max_q_size: maximum size for the generator queue

        # Returns
            A `History` object.

        # Example

        ```python
            def generate_arrays_from_file(path):
                while 1:
                    f = open(path)
                    for line in f:
                        # create numpy arrays of input data
                        # and labels, from each line in the file
                        x, y = process_line(line)
                        yield (x, y)
                    f.close()

            model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                                samples_per_epoch=10000, nb_epoch=10)
        ```
        '''
        if self.model is None:
            raise Exception('The model needs to be compiled before being used.')
        if 'show_accuracy' in kwargs:
            kwargs.pop('show_accuracy')
            warnings.warn('The "show_accuracy" argument is deprecated, '
                          'instead you should pass the "accuracy" metric to '
                          'the model at compile time:\n'
                          '`model.compile(optimizer, loss, '
                          'metrics=["accuracy"])`')
        if 'nb_worker' in kwargs:
            kwargs.pop('nb_worker')
            warnings.warn('The "nb_worker" argument is deprecated, '
                          'please remove it from your code.')
        if 'nb_val_worker' in kwargs:
            kwargs.pop('nb_val_worker')
            warnings.warn('The "nb_val_worker" argument is deprecated, '
                          'please remove it from your code.')
        if kwargs:
            raise Exception('Received unknown keyword arguments: ' +
                            str(kwargs))
        return self.model.fit_generator(generator,
                                        samples_per_epoch,
                                        nb_epoch,
                                        verbose=verbose,
                                        callbacks=callbacks,
                                        validation_data=validation_data,
                                        nb_val_samples=nb_val_samples,
                                        class_weight=class_weight,
                                        max_q_size=max_q_size)

    def evaluate_generator(self, generator, val_samples, max_q_size=10, **kwargs):
        '''Evaluates the model on a data generator. The generator should
        return the same kind of data as accepted by `test_on_batch`.

        Arguments:
            generator:
                generator yielding tuples (inputs, targets)
                or (inputs, targets, sample_weights)
            val_samples:
                total number of samples to generate from `generator`
                before returning.
            max_q_size: maximum size for the generator queue
        '''
        if self.model is None:
            raise Exception('The model needs to be compiled before being used.')
        if 'show_accuracy' in kwargs:
            kwargs.pop('show_accuracy')
            warnings.warn('The "show_accuracy" argument is deprecated, '
                          'instead you should pass the "accuracy" metric to '
                          'the model at compile time:\n'
                          '`model.compile(optimizer, loss, '
                          'metrics=["accuracy"])`')
        if 'verbose' in kwargs:
            kwargs.pop('verbose')
            warnings.warn('The "verbose" argument is deprecated.')
        if kwargs:
            raise Exception('Received unknown keyword arguments: ' +
                            str(kwargs))
        return self.model.evaluate_generator(generator,
                                             val_samples,
                                             max_q_size=max_q_size)

    def predict_generator(self, generator, val_samples, max_q_size=10):
        '''Generates predictions for the input samples from a data generator.
        The generator should return the same kind of data as accepted by
        `predict_on_batch`.

        # Arguments
            generator: generator yielding batches of input samples.
            val_samples: total number of samples to generate from `generator`
                before returning.
            max_q_size: maximum size for the generator queue

        # Returns
            A Numpy array of predictions.
        '''
        if self.model is None:
            raise Exception('The model needs to be compiled before being used.')
        return self.model.predict_generator(generator, val_samples,
                                            max_q_size=max_q_size)

    def get_config(self):
        '''Returns the model configuration
        as a Python dictionary.
        '''
        config = []
        if self.layers[0].__class__.__name__ == 'Merge':
            assert hasattr(self.layers[0], 'layers')
            layers = []
            for layer in self.layers[0].layers:
                layer_config = {'class_name': layer.__class__.__name__,
                                'config': layer.get_config()}
                layers.append(layer_config)
            merge_config = self.layers[0].get_config()
            merge_config['layers'] = layers
            config.append({'class_name': 'Merge', 'config': merge_config})
        else:
            config.append({'class_name': self.layers[0].__class__.__name__,
                           'config': self.layers[0].get_config()})
        for layer in self.layers[1:]:
            config.append({'class_name': layer.__class__.__name__,
                           'config': layer.get_config()})
        return copy.deepcopy(config)

    @classmethod
    def from_config(cls, config):
        '''Supports legacy formats
        '''
        from keras.utils.layer_utils import layer_from_config
        from keras.layers import Merge
        assert type(config) is list

        def normalize_legacy_config(conf):
            if 'class_name' not in conf:
                class_name = conf['name']
                name = conf.get('custom_name')
                conf['name'] = name
                new_config = {
                    'class_name': class_name,
                    'config': conf,
                }
                return new_config
            return conf

        model = cls()

        first_layer = config[0]
        first_layer = normalize_legacy_config(first_layer)
        if first_layer['class_name'] == 'Merge':
            merge_inputs = []
            first_layer_config = first_layer['config']
            for merge_input_config in first_layer_config.pop('layers'):
                merge_input = layer_from_config(merge_input_config)
                merge_inputs.append(merge_input)
            first_layer_config['layers'] = merge_inputs
            merge = Merge.from_config(first_layer_config)
            model.add(merge)
        else:
#<<<<<<< HEAD
#            self.validation_data = None
#
#        self.stop_training = False
#        while epoch < nb_epoch:
#            callbacks.on_epoch_begin(epoch)
#            samples_seen = 0
#            batch_index = 0
#            while samples_seen < samples_per_epoch:
#                while not _data_stop.is_set():
#                    if not data_gen_queue.empty():
#                        generator_output = data_gen_queue.get()
#                        break
#                    else:
#                        time.sleep(wait_time)
#
#                data, sample_weight = self._check_generator_output(generator_output,
#                                                                   _data_stop)
#                batch_logs = {}
#                batch_size = len(data[list(data.keys())[0]])
#                batch_logs['batch'] = batch_index
#                batch_logs['size'] = batch_size
#                callbacks.on_batch_begin(batch_index, batch_logs)
#                outs = self.train_on_batch(data,
#                                           sample_weight=sample_weight,
#                                           class_weight=class_weight,
#                                           accuracy=show_accuracy)
#                if type(outs) != list:
#                    outs = [outs]
#                for l, o in zip(out_labels, outs):
#                    batch_logs[l] = o
#
#                callbacks.on_batch_end(batch_index, batch_logs)
#
#                # construct epoch logs
#                epoch_logs = {}
#                batch_index += 1
#                samples_seen += batch_size
#                
#                # epoch finished
#                if samples_seen >= samples_per_epoch and do_validation:
#                    if val_gen:
#                        val_outs = self.evaluate_generator(validation_data,
#                                                           nb_val_samples,
#                                                           verbose=0,
#                                                           show_accuracy=show_accuracy,
#                                                           nb_worker=nb_val_worker,
#                                                           wait_time=wait_time)
#                    else:
#                        val_outs = self.evaluate(data_val,
#                                                 sample_weight=sample_weight_val,
#                                                 show_accuracy=show_accuracy,
#                                                 verbose=0)
#                    if type(val_outs) != list:
#                        val_outs = [val_outs]
#                    # same labels assumed
#                    for l, o in zip(out_labels, val_outs):
#                        epoch_logs['val_' + l] = o
#
#            callbacks.on_epoch_end(epoch, epoch_logs)
#            epoch += 1
#            if self.stop_training:
#                break
#        _data_stop.set()
#        callbacks.on_train_end()
#        return self.history


#    def predict_generator(self, generator, nb_samples,
#                               verbose=0, **kwargs):
#            '''Generate output predictions for the input samples batch by batch that are drawn
#               from the `generator`. 
#
#            Arguments:
#                generator:
#                    generator yielding dictionaries of the kind accepted
#                    by `fit_generator`, or tuples of such dictionaries and
#                    associated dictionaries of sample weights.
#                nb_samples:
#                    total number of samples to be predicted
#                verbose, max_q_size, wait_time, nb_worker: the same as `fit_generator`
#
#            '''
#            done_samples = 0
#            all_outs = []
#            q, _stop = generator_queue(generator, **kwargs)
#
#            batch_index = 0
#            while done_samples < nb_samples:
#                data, _ = self._check_generator_output(q.get(), _stop)
#                do_samples = len(data[next(iter(data.keys()))])
#                outs_dct = self.predict(data, batch_size=do_samples, verbose=verbose)
#                if batch_index == 0:
#                    shape = (nb_samples,) + outs_dct[next(iter(outs_dct.keys()))].shape[1:]
#                    all_outs.append(np.zeros(shape))
#                all_outs[0][done_samples:(done_samples+do_samples)]
#
#                done_samples += do_samples
#                batch_index += 1
#
#            _stop.set()
#            return dict(zip(self.output_order, all_outs))


#    def fit_on_generator(self, data_generator, samples_per_epoch, nb_epoch, class_weight=None,
#                      callbacks=[], validation_generator=None, samples_per_epoch_valid=100,
#                      verbose=1, nb_worker=1):
#        '''Fit a model on data generated batch-by-batch by a Python generator.
#        The generator is run in parallel to the model, for efficiency,
#        and can be run by multiple workers at the same time.
#        For instance, this allows you to do real-time data augmentation
#        on images on CPU in parallel to training your model on GPU.
#
#        # Arguments
#            data_generator: a Python generator of generators. It yields 
#                `epoch_generator` that yields batch data in the form of
#                (X, y) or (X, y, sample_weight) or (X, y, sample_weight, ...)
#            samples_per_epoch: integer, number of samples to process before
#                going to the next epoch.
#            nb_epoch: integer, total number of iterations on the data.
#            verbose: verbosity mode, 0, 1, or 2.
#            callbacks: list of callbacks to be called during training.
#            validation_generator: held-out validation data. The form is the same
#                to `data_generator` 
#            samples_per_epoch_valid: integer, number of samples in valid data
#            class_weight: dictionary mapping class indices to a weight
#                for the class.
#            nb_worker: integer, number of workers to use for running
#                the generator (in parallel to model training).
#                If using multiple workers, the processing order of batches
#                generated by the model will be non-deterministic.
#                If using multiple workers, make sure to protect
#                any thread-unsafe operation done by the generator
#                using a Python mutex.
#
#        # Returns
#
#        A `History` object.
#
#        # Examples
#
#        ```python
#            def generate_arrays_from_file(path):
#                while 1:
#                    f = open(path)
#                    for line in f:
#                        # create numpy arrays of input data
#                        # and labels, from each line in the file
#                        x1, x2, y = process_line(line)
#                        yield {'input_1': x1, 'input_2': x2, 'output': y}
#                    f.close()
#
#            graph.fit_generator(generate_arrays_from_file('/my_file.txt'),
#                                samples_per_epoch=10000, nb_epoch=10)
#        ```
#        '''
#        max_queue_size = 10  # maximum number of batches in queue
#        wait_time = 0.05  # in seconds
#        epoch = 0
#        do_validation = bool(validation_generator)
#        out_labels = ['loss']
#        metrics = ['loss', 'val_loss']
#        if not class_weight:
#            class_weight = {}
#
#        # prepare callbacks
#        history = cbks.History()
#        if verbose:
#            callbacks = [history, cbks.BaseLogger()] + callbacks
#        else:
#            callbacks = [history] + callbacks
#        callbacks = cbks.CallbackList(callbacks)
#
#        callbacks._set_model(self)
#        callbacks._set_params({
#            'nb_epoch': nb_epoch,
#            'nb_sample': samples_per_epoch,
#            'verbose': verbose,
#            'do_validation': do_validation,
#            'metrics': metrics,
#        })
#        callbacks.on_train_begin()
#        
#        # # start generator thread storing batches into a queue
#        # generator_queue = queue.Queue()
#        # _stop = threading.Event()
#
#        # util function to validate the batches produced by the generator
#        def input_validation(generator_output):
#            if type(generator_output) in [list, tuple]:
#                if len(generator_output) == 2:
#                    data, sample_weight = generator_output
#                else:
#                    _stop.set()
#                    raise Exception('The generator output tuple must have '
#                                    '2 dictionary elements: '
#                                    '(data, sample_weight).')
#            elif type(generator_output) == dict:
#                data = generator_output
#                sample_weight = {}
#            else:
#                _stop.set()
#                raise Exception('The generator output must be '
#                                'a data dictionary or a tuple '
#                                '(data, sample_weight).')
#            assert type(data) == dict
#            assert type(sample_weight) == dict
#            if len(set([len(data[name]) for name in data.keys()] +
#                       [len(sample_weight[name]) for name in sample_weight.keys()])) != 1:
#                raise Exception('All input arrays and target arrays must have '
#                                'the same number of samples.')
#            sample_weight = {name: standardize_weights(data[name],
#                             sample_weight=sample_weight.get(name),
#                             sample_weight_mode=self.sample_weight_modes.get(name)) for name in self.output_order}
#            return data, sample_weight
#
#        def generator_task(generator):
#            i = 0
#            while not _stop.is_set():
#                try:
#                    if generator_queue.qsize() < max_queue_size:
#                        gen_output = next(generator)
#                        generator_queue.put(gen_output)
#                        i += 1
#                    else:
#                        time.sleep(wait_time)
#                except:
#                    # raise exception that shows generator is empty
#                    _stop.set()
#                    return
#
#        self.stop_training = False
#        while epoch < nb_epoch:
#            _stop.clear()
#            callbacks.on_epoch_begin(epoch)
#            samples_seen = 0
#            batch_index = 0
#            epoch_generator = next(data_generator)
#            generator_threads = [threading.Thread(target=generator_task, kwargs={'generator':epoch_generator}) for _ in range(nb_worker)]
#            for thread in generator_threads:
#                thread.daemon = True
#                thread.start()
#
#            while samples_seen < samples_per_epoch:
#                if not generator_queue.empty():
#                    generator_output = generator_queue.get()
#                elif not _stop.is_set():
#                    time.sleep(wait_time)
#                    continue
#                elif _stop.is_set():
#                    print('threading is stopped.')
#                    break
#
#                # while not _stop.is_set():
#                #     if not generator_queue.empty():
#                #         generator_output = generator_queue.get()
#                #         break
#                #     else:
#                #         time.sleep(wait_time)
#
#                data, sample_weight = input_validation(generator_output)
#
#                batch_logs = {}
#                batch_size = len(data[list(data.keys())[0]])
#                batch_logs['batch'] = batch_index
#                batch_logs['size'] = batch_size
#                callbacks.on_batch_begin(batch_index, batch_logs)
#                outs = self.train_on_batch(data,
#                                           sample_weight=sample_weight,
#                                           class_weight=class_weight)
#                if type(outs) != list:
#                    outs = [outs]
#                for l, o in zip(out_labels, outs):
#                    batch_logs[l] = o
#
#                callbacks.on_batch_end(batch_index, batch_logs)
#
#                # construct epoch logs
#                epoch_logs = {}
#                batch_index += 1
#                samples_seen += batch_size
#                if samples_seen >= samples_per_epoch and do_validation:
#                    validation_data = next(validation_generator)
#                    if hasattr(validation_data, 'next'):
#                        val_outs = self.evaluate_on_generator(validation_data,
#                                                        samples_per_epoch_valid,
#                                                        nb_worker)
#                    else:
#                        _stop.set()
#                        raise Exception('validation data is not a generator')
#                        # val_outs = self.evaluate(data_val,
#                        #                          sample_weight=sample_weight_val,
#                        #                          verbose=0)
#                    if type(val_outs) != list:
#                        val_outs = [val_outs]
#                    # same labels assumed
#                    for l, o in zip(out_labels, val_outs):
#                        epoch_logs['val_' + l] = o
#        # if do_validation:
#        #     data_val, sample_weight_val = input_validation(validation_data)
#        #     sample_weight_val_l = [sample_weight_val[name] for name in self.output_order]
#        #     y_val = [standardize_y(data_val[name]) for name in self.output_order]
#        #     self.validation_data = [data_val[name] for name in self.input_order] + y_val + sample_weight_val_l
#        # else:
#        #     self.validation_data = None
#
#            callbacks.on_epoch_end(epoch, epoch_logs)
#            epoch += 1
#            if self.stop_training:
#                break
#        _stop.set()
#        callbacks.on_train_end()
#        return history

#    def evaluate_on_generator(self, data_generator, samples_per_epoch_valid, verbose=0,nb_worker=1):
#        max_queue_size_eval = 10
#        wait_time_eval = 0.05
#        outs = []
#        if verbose == 1:
#            progbar = Progbar(target=samples_per_epoch_valid)
#
#        # start generator thread storing batches into a queue
#        generator_queue_eval = queue.Queue()
#        _stop_eval = threading.Event()
#        
#        # util function to validate the batches produced by the generator
#        def input_validation_eval(generator_output):
#            if type(generator_output) in [list, tuple]:
#                if len(generator_output) == 2:
#                    data, sample_weight = generator_output
#                else:
#                    _stop_eval.set()
#                    raise Exception('The generator output tuple must have '
#                                    '2 dictionary elements: '
#                                    '(data, sample_weight).')
#            elif type(generator_output) == dict:
#                data = generator_output
#                sample_weight = {}
#            else:
#                _stop.set()
#                raise Exception('The generator output must be '
#                                'a data dictionary or a tuple '
#                                '(data, sample_weight).')
#            assert type(data) == dict
#            assert type(sample_weight) == dict
#            if len(set([len(data[name]) for name in data.keys()] +
#                       [len(sample_weight[name]) for name in sample_weight.keys()])) != 1:
#                raise Exception('All input arrays and target arrays must have '
#                                'the same number of samples.')
#            sample_weight = {name: standardize_weights(data[name],
#                             sample_weight=sample_weight.get(name),
#                             sample_weight_mode=self.sample_weight_modes.get(name)) for name in self.output_order}
#            return data, sample_weight
#
#        def generator_task_eval(generator):
#            i = 0
#            while not _stop_eval.is_set():
#                try:
#                    if generator_queue_eval.qsize() < max_queue_size_eval:
#                        gen_output_eval = next(generator)
#                        generator_queue_eval.put(gen_output_eval)
#                        i += 1
#                    else:
#                        time.sleep(wait_time_eval)
#                except:
#                    # raise exception that shows generator is empty
#                    _stop_eval.set()
#                    return
#
#        generator_threads_eval = [threading.Thread(target=generator_task_eval, kwargs={'generator':data_generator}) for _ in range(nb_worker)]
#        for thread in generator_threads_eval:
#            thread.daemon = True
#            thread.start()
#
#        samples_seen_eval = 0
#        batch_index = 0
#        while samples_seen_eval < samples_per_epoch_valid:
#            if not generator_queue_eval.empty():
#                generator_output_eval = generator_queue_eval.get()
#            elif not _stop_eval.is_set():
#                time.sleep(wait_time_eval)
#                continue
#            elif _stop_eval.is_set():
#                print('threading is stopped.')
#                break
#            
#            data_eval, sample_weight_eval = input_validation_eval(generator_output_eval)
#            batch_size_eval = len(data_eval[list(data_eval.keys())[0]])
#            sample_weight = [standardize_weights(data_eval[name],
#                        sample_weight=sample_weight_eval.get(name),
#                        sample_weight_mode=self.sample_weight_modes.get(name)) for name in self.output_order]
#            ins = [data_eval[name] for name in self.input_order] + [standardize_y(data_eval[name]) for name in self.output_order] + sample_weight
#            if len(set([len(a) for a in ins])) != 1:
#                raise Exception('All input arrays and target arrays in validation must have '
#                                'the same number of samples.')
#            batch_outs = self._test(ins)
#            if type(batch_outs) == list:
#                if batch_index == 0:
#                    for batch_out in enumerate(batch_outs):
#                        outs.append(0.)
#                for i, batch_out in enumerate(batch_outs):
#                    outs[i] += batch_out * batch_size_eval
#            else:
#                if batch_index == 0:
#                    outs.append(0.)
#                outs[0] += batch_outs * batch_size_eval
#
#            samples_seen_eval += batch_size_eval
#            batch_index += 1
#            if verbose == 1:
#                progbar.update(samples_seen_eval)
#
#        for i, out in enumerate(outs):
#            outs[i] /= samples_per_epoch_valid
#        
#        _stop_eval.set()
#        return outs[0]

#    def predict_on_generator(self, data_generator, nb_sample, nb_worker=1, verbose=0):
#        '''Compute the loss on some input data generator, batch by batch.
#
#        Arguments: see `fit` method.
#        '''
#        max_queue_size_pred = 10
#        wait_time_pred = 0.05
#        outs = []
#        if verbose == 1:
#            progbar = Progbar(target=nb_sample)
#
#        # start generator thread storing batches into a queue
#        generator_queue_pred = queue.Queue()
#        _stop_pred = threading.Event()
#        
#        # util function to validate the batches produced by the generator
#        def input_validation_pred(generator_output):
#            if type(generator_output) in [list, tuple]:
#                if len(generator_output) == 2:
#                    data, sample_weight = generator_output
#                else:
#                    _stop_pred.set()
#                    raise Exception('The generator output tuple must have '
#                                    '2 dictionary elements: '
#                                    '(data, sample_weight).')
#            elif type(generator_output) == dict:
#                data = generator_output
#                sample_weight = {}
#            else:
#                _stop_pred.set()
#                raise Exception('The generator output must be '
#                                'a data dictionary or a tuple '
#                                '(data, sample_weight).')
#            assert type(data) == dict
#            assert type(sample_weight) == dict
#            if len(set([len(data[name]) for name in data.keys()] +
#                       [len(sample_weight[name]) for name in sample_weight.keys()])) != 1:
#                raise Exception('All input arrays and target arrays must have '
#                                'the same number of samples.')
#            sample_weight = {name: standardize_weights(data[name],
#                             sample_weight=sample_weight.get(name),
#                             sample_weight_mode=self.sample_weight_modes.get(name)) for name in self.output_order}
#            return data, sample_weight
#
#        def generator_task_pred(generator):
#            i = 0
#            while not _stop_pred.is_set():
#                try:
#                    if generator_queue_pred.qsize() < max_queue_size_pred:
#                        gen_output_pred = next(generator)
#                        generator_queue_pred.put(gen_output_pred)
#                        i += 1
#                    else:
#                        time.sleep(wait_time_pred)
#                except:
#                    # raise exception that shows generator is empty
#                    _stop_pred.set()
#                    return
#
#        generator_threads_pred = [threading.Thread(target=generator_task_pred, kwargs={'generator':data_generator}) for _ in range(nb_worker)]
#        for thread in generator_threads_pred:
#            thread.daemon = True
#            thread.start()
#
#        samples_seen_pred = 0
#        batch_index = 0
#        while samples_seen_pred < nb_sample:
#            if not generator_queue_pred.empty():
#                generator_output_pred = generator_queue_pred.get()
#            elif not _stop_pred.is_set():
#                time.sleep(wait_time_pred)
#                continue
#            elif _stop_pred.is_set():
#                print('threading is stopped.')
#                break
#            ins = [generator_output_pred[name] for name in self.input_order]
#            if len(set([len(a) for a in ins])) != 1:
#                raise Exception('All input arrays and target arrays in validation must have '
#                                'the same number of samples.')
#            batch_outs = self._predict(ins)
#            if type(batch_outs) != list:
#                batch_outs = [batch_outs]
#            if batch_index == 0:
#                for batch_out in batch_outs:
#                    shape = (nb_sample,) + batch_out.shape[1:]
#                    outs.append(np.zeros(shape))
#            batch_size = len(generator_output_pred[list(generator_output_pred.keys())[0]])
#            for i, batch_out in enumerate(batch_outs):
#                outs[i][samples_seen_pred:samples_seen_pred+batch_size] = batch_out
#
#            samples_seen_pred += batch_size
#            batch_index += 1
#
#            if verbose == 1:
#                progbar.update(samples_seen_pred)
#        
#        _stop_pred.set()
#        return dict(zip(self.output_order, outs))
#=======
            layer = layer_from_config(first_layer)
            model.add(layer)

        for conf in config[1:]:
            conf = normalize_legacy_config(conf)
            layer = layer_from_config(conf)
            model.add(layer)
        return model
