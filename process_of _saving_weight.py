def save_weights(self, filepath, overwrite=True, save_format=None):
    """Saves all layer weights.

    Either saves in HDF5 or in TensorFlow format based on the `save_format`
    argument.

    When saving in HDF5 format, the weight file has:
      - `layer_names` (attribute), a list of strings
          (ordered names of model layers).
      - For every layer, a `group` named `layer.name`
          - For every such layer group, a group attribute `weight_names`,
              a list of strings
              (ordered names of weights tensor of the layer).
          - For every weight in the layer, a dataset
              storing the weight value, named after the weight tensor.

    When saving in TensorFlow format, all objects referenced by the network are
    saved in the same format as `tf.train.Checkpoint`, including any `Layer`
    instances or `Optimizer` instances assigned to object attributes. For
    networks constructed from inputs and outputs using `tf.keras.Model(inputs,
    outputs)`, `Layer` instances used by the network are tracked/saved
    automatically. For user-defined classes which inherit from `tf.keras.Model`,
    `Layer` instances must be assigned to object attributes, typically in the
    constructor. See the documentation of `tf.train.Checkpoint` and
    `tf.keras.Model` for details.

    Arguments:
        filepath: String, path to the file to save the weights to. When saving
            in TensorFlow format, this is the prefix used for checkpoint files
            (multiple files are generated). Note that the '.h5' suffix causes
            weights to be saved in HDF5 format.
        overwrite: Whether to silently overwrite any existing file at the
            target location, or provide the user with a manual prompt.
        save_format: Either 'tf' or 'h5'. A `filepath` ending in '.h5' or
            '.keras' will default to HDF5 if `save_format` is `None`. Otherwise
            `None` defaults to 'tf'.

    Raises:
        ImportError: If h5py is not available when attempting to save in HDF5
            format.
        ValueError: For invalid/unknown format arguments.
    """
    filepath_is_h5 = _is_hdf5_filepath(filepath)
    if save_format is None:
      if filepath_is_h5:
        save_format = 'h5'
      else:
        save_format = 'tf'
    else:
      user_format = save_format.lower().strip()
      if user_format in ('tensorflow', 'tf'):
        save_format = 'tf'
      elif user_format in ('hdf5', 'h5', 'keras'):
        save_format = 'h5'
      else:
        raise ValueError(
            'Unknown format "%s". Was expecting one of {"tf", "h5"}.' % (
                save_format,))
    if save_format == 'tf' and filepath_is_h5:
      raise ValueError(
          ('save_weights got save_format="tf"/"tensorflow", but the '
           'filepath ("%s") looks like an HDF5 file. Omit the ".h5"/".keras" '
           'when saving in TensorFlow format.')
          % filepath)

    if save_format == 'h5' and h5py is None:
      raise ImportError(
          '`save_weights` requires h5py when saving in hdf5.')
    if save_format == 'tf':
      check_filepath = filepath + '.index'
    else:
      check_filepath = filepath
    # If file exists and should not be overwritten:
    if not overwrite and os.path.isfile(check_filepath):
      proceed = ask_to_proceed_with_overwrite(check_filepath)
      if not proceed:
        return
    if save_format == 'h5':
      with h5py.File(filepath, 'w') as f:
        saving.save_weights_to_hdf5_group(f, self.layers)
    else:
      if context.executing_eagerly():
        session = None
      else:
        session = backend.get_session()
      self._checkpointable_saver.save(filepath, session=session)
def save_weights_to_hdf5_group(f, layers):
  """Saves the weights of a list of layers to a HDF5 group.

  Arguments:
      f: HDF5 group.
      layers: List of layer instances.
  """
  from tensorflow.python.keras import __version__ as keras_version  # pylint: disable=g-import-not-at-top

  save_attributes_to_hdf5_group(
      f, 'layer_names', [layer.name.encode('utf8') for layer in layers])
  f.attrs['backend'] = K.backend().encode('utf8')
  f.attrs['keras_version'] = str(keras_version).encode('utf8')

  for layer in layers:
    g = f.create_group(layer.name)
    symbolic_weights = layer.weights
    weight_values = K.batch_get_value(symbolic_weights)
    weight_names = []
    for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
      if hasattr(w, 'name') and w.name:
        name = str(w.name)
      else:
        name = 'param_' + str(i)
      weight_names.append(name.encode('utf8'))
    save_attributes_to_hdf5_group(g, 'weight_names', weight_names)
    for name, val in zip(weight_names, weight_values):
      param_dset = g.create_dataset(name, val.shape, dtype=val.dtype)
      if not val.shape:
        # scalar
        param_dset[()] = val
      else:
        param_dset[:] = val
