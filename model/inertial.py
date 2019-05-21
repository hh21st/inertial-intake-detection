import os
import itertools
import numpy as np
import tensorflow as tf
import best_checkpoint_exporter
from tensorflow.python.platform import gfile


NUM_SHARDS = 10
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer(
    name='batch_size', default=64, help='Batch size used for training.')
tf.app.flags.DEFINE_string(
    name='eval_dir', default='../data/eval', help='Directory for eval data.')
tf.app.flags.DEFINE_enum(
    name='mode', default="train_and_evaluate", enum_values=["train_and_evaluate", "predict_and_export_csv"],
    help='What mode should tensorflow be started in')
tf.app.flags.DEFINE_string(
    name='model_dir', default='run',
    help='Output directory for model and training stats.')
tf.app.flags.DEFINE_integer(
    name='num_sequences', default=1185852, help='Number of training example steps.')
tf.app.flags.DEFINE_integer(
    name='seq_length', default=128,
    help='Number of sequence elements.')
tf.app.flags.DEFINE_integer(
    name='seq_pool', default=1, help='Factor of sequence pooling in the model.')
tf.app.flags.DEFINE_integer(
    name='seq_shift', default=8, help='Shift taken in sequence generation.')
tf.app.flags.DEFINE_string(
    name='train_dir', default='../data/train', help='Directory for training data.')
tf.app.flags.DEFINE_float(
    name='train_epochs', default=60, help='Number of training epochs.')
tf.app.flags.DEFINE_boolean(
    name='use_sequence_loss', default=True,
    help='Use sequence-to-sequence loss')


def run_experiment(arg=None):
    """Run the experiment."""

    steps_per_epoch = int(FLAGS.num_sequences / FLAGS.batch_size \
                        * FLAGS.seq_shift / FLAGS.seq_length)
    max_steps = steps_per_epoch * FLAGS.train_epochs

    # Model parameters
    params = tf.contrib.training.HParams(
        base_learning_rate=3e-3,
        batch_size=FLAGS.batch_size,
        decay_rate=0.9,
        dropout=0.5,
        gradient_clipping_norm=10.0,
        l2_lambda=1e-4,
        num_classes=2,
        num_lstm=64,
        seq_length=FLAGS.seq_length,
        steps_per_epoch=steps_per_epoch)

    # Run config
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        save_summary_steps=10,
        save_checkpoints_steps=25)

    # Define the estimator
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.model_dir,
        params=params,
        config=run_config)

    # Exporters
    best_exporter = best_checkpoint_exporter.BestCheckpointExporter(
        score_metric='metrics/accuracy',
        compare_fn=lambda x,y: x.score > y.score,
        sort_key_fn=lambda x: -x.score)

    # Training input_fn
    def train_input_fn():
        return input_fn(is_training=True, data_dir=FLAGS.train_dir)

    # Eval input_fn
    def eval_input_fn():
        return input_fn(is_training=False, data_dir=FLAGS.eval_dir)

    # Define the experiment
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=max_steps)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=None,
        exporters=best_exporter,
        start_delay_secs=30,
        throttle_secs=20)

    # Start the experiment
    if FLAGS.mode == "train_and_evaluate":
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    elif FLAGS.mode == "predict_and_export_csv":
        seq_skip = FLAGS.seq_length - 1
        predict_and_export_csv(estimator, eval_input_fn, FLAGS.eval_dir, seq_skip)


def model_fn(features, labels, mode, params):
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    is_predicting = mode == tf.estimator.ModeKeys.PREDICT

    # Set features to correct shape
    features = tf.reshape(features, [params.batch_size, params.seq_length, 12])

    # Model
    logits = model_pool_4(features, params)

    # If necessary, slice last sequence step for logits
    final_logits = logits[:,-1,:] if logits.get_shape().ndims == 3 else logits

    # Decode logits into predictions
    predictions = {
        'classes': tf.argmax(final_logits, axis=-1),
        'probabilities': tf.nn.softmax(final_logits, name='softmax_tensor')}

    if is_predicting:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })

    # If necessary, slice last sequence step for labels
    final_labels = labels[:,-1] if labels.get_shape().ndims == 2 else labels

    if logits.get_shape().ndims == 3:
        seq_length = int(FLAGS.seq_length / FLAGS.seq_pool)
        # If seq pooling performed in model, slice the labels as well
        if FLAGS.seq_pool > 1:
            labels = tf.strided_slice(input_=labels,
                begin=[0, FLAGS.seq_pool-1],
                end=[FLAGS.batch_size, FLAGS.seq_length],
                strides=[1, FLAGS.seq_pool])
        labels = tf.reshape(labels, [params.batch_size, seq_length])

    def _compute_balanced_sample_weight(labels):
        """Calculate the balanced sample weight for imbalanced data."""
        f_labels = tf.reshape(labels,[-1]) if labels.get_shape().ndims == 2 else labels
        y, idx, count = tf.unique_with_counts(f_labels)
        total_count = tf.size(f_labels)
        label_count = tf.size(y)
        calc_weight = lambda x: tf.divide(tf.divide(total_count, x),
            tf.cast(label_count, tf.float64))
        class_weights = tf.map_fn(fn=calc_weight, elems=count, dtype=tf.float64)
        sample_weights = tf.gather(class_weights, idx)
        sample_weights = tf.reshape(sample_weights, tf.shape(labels))
        return tf.cast(sample_weights, tf.float32)

    # Training with multiple labels per sequence
    if FLAGS.use_sequence_loss:

        # Calculate sample weights
        if is_training:
            sample_weights = _compute_balanced_sample_weight(labels)
        else:
            sample_weights = tf.ones_like(labels, dtype=tf.float32)

        # Calculate and scale cross entropy
        scaled_loss = tf.contrib.seq2seq.sequence_loss(
            logits=logits,
            targets=tf.cast(labels, tf.int32),
            weights=sample_weights)
        tf.identity(scaled_loss, name='seq2seq_loss')
        tf.summary.scalar('loss/seq2seq_loss', scaled_loss)

    # Training with one label per sequence
    else:

        # Calculate sample weights
        if is_training:
            sample_weights = _compute_balanced_sample_weight(final_labels)
        else:
            sample_weights = tf.ones_like(final_labels, dtype=tf.float32)

        # Calculate scaled cross entropy
        unscaled_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(final_labels, tf.int32),
            logits=final_logits)
        scaled_loss = tf.reduce_mean(tf.multiply(unscaled_loss, sample_weights))
        tf.summary.scalar('loss/scaled_loss', scaled_loss)

    # Compute loss with Weight decay
    l2_loss = params.l2_lambda * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
            if 'norm' not in v.name])
    tf.summary.scalar('loss/l2_loss', l2_loss)
    loss = scaled_loss + l2_loss

    if is_training:
        global_step = tf.train.get_or_create_global_step()

        def _decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(
                learning_rate=learning_rate, global_step=global_step,
                decay_steps=params.steps_per_epoch, decay_rate=params.decay_rate)

        # Learning rate
        learning_rate = _decay_fn(params.base_learning_rate, global_step)
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('training/learning_rate', learning_rate)

        # The optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grad_vars = optimizer.compute_gradients(loss)

        tf.summary.scalar("training/global_gradient_norm",
            tf.global_norm(list(zip(*grad_vars))[0]))

        # Clip gradients
        grads, vars = zip(*grad_vars)
        grads, _ = tf.clip_by_global_norm(grads, params.gradient_clipping_norm)
        grad_vars = list(zip(grads, vars))

        for grad, var in grad_vars:
            var_name = var.name.replace(":", "_")
            tf.summary.histogram("gradients/%s" % var_name, grad)
            tf.summary.scalar("gradient_norm/%s" % var_name, tf.global_norm([grad]))
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("training/clipped_global_gradient_norm",
            tf.global_norm(list(zip(*grad_vars))[0]))

        minimize_op = optimizer.apply_gradients(grad_vars, global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)

    else:
        train_op = None

    # Calculate accuracy metrics - always done with final labels
    final_labels = tf.cast(final_labels, tf.int64)
    accuracy = tf.metrics.accuracy(
        labels=final_labels, predictions=predictions['classes'])
    mean_per_class_accuracy = tf.metrics.mean_per_class_accuracy(
        labels=final_labels, predictions=predictions['classes'],
        num_classes=params.num_classes)
    tf.summary.scalar('metrics/accuracy', accuracy[1])
    tf.summary.scalar('metrics/mean_per_class_accuracy',
        tf.reduce_mean(mean_per_class_accuracy[1]))
    metrics = {
        'metrics/accuracy': accuracy,
        'metrics/mean_per_class_accuracy': mean_per_class_accuracy}

    # Calculate class-specific metrics
    for i in range(params.num_classes):
        class_precision = tf.metrics.precision_at_k(
            labels=final_labels, predictions=final_logits, k=1, class_id=i)
        class_recall = tf.metrics.recall_at_k(
            labels=final_labels, predictions=final_logits, k=1, class_id=i)
        tf.summary.scalar('metrics/class_%d_precision' % i, class_precision[1])
        tf.summary.scalar('metrics/class_%d_recall' % i, class_recall[1])
        metrics['metrics/class_%d_precision' % i] = class_precision
        metrics['metrics/class_%d_recall' % i] = class_recall

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def model_pool_4(inputs, params):
    inputs = tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=10,
        padding='same',
        activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(
        pool_size=2)(inputs)
    inputs = tf.keras.layers.Conv1D(
        filters=128,
        kernel_size=10,
        padding='same',
        activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(
        pool_size=2)(inputs)
    inputs = tf.keras.layers.Dropout(params.dropout)(inputs)
    inputs = tf.keras.layers.Dense(32)(inputs)
    inputs = tf.keras.layers.LSTM(
            params.num_lstm, return_sequences=True)(inputs)
    inputs = tf.keras.layers.LSTM(
            params.num_lstm, return_sequences=True)(inputs)
    inputs = tf.keras.layers.Dense(params.num_classes)(inputs)
    return inputs


def model(inputs, params):
    inputs = tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=10,
        padding='same',
        activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(
        filters=128,
        kernel_size=10,
        padding='same',
        activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Dropout(params.dropout)(inputs)
    inputs = tf.keras.layers.Dense(32)(inputs)
    inputs = tf.keras.layers.LSTM(
            params.num_lstm, return_sequences=True)(inputs)
    inputs = tf.keras.layers.LSTM(
            params.num_lstm, return_sequences=True)(inputs)
    inputs = tf.keras.layers.Dense(params.num_classes)(inputs)
    return inputs


def input_fn(is_training, data_dir):
    """Input pipeline"""
    # Scan for training files
    filenames = gfile.Glob(os.path.join(data_dir, "*.csv"))
    if not filenames:
        raise RuntimeError('No files found.')
    tf.logging.info("Found {0} files.".format(str(len(filenames))))
    # List files
    files = tf.data.Dataset.list_files(filenames)
    # Lookup table for Labels
    mapping_strings = tf.constant(["idle", "Intake"])
    table = tf.contrib.lookup.index_table_from_tensor(
        mapping=mapping_strings)
    # Initialize table
    with tf.Session() as sess:
        sess.run(table.init)
    # Shuffle files if needed
    if is_training:
        files = files.shuffle(NUM_SHARDS)
    select_cols = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
    record_defaults = [tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.string]
    shift = 1 if is_training else FLAGS.seq_length
    dataset = files.interleave(
        lambda filename:
            tf.data.experimental.CsvDataset(filenames=filename,
                record_defaults=record_defaults, select_cols=select_cols,
                header=True)
            .map(map_func=_get_input_parser(table))
            .window(size=FLAGS.seq_length, shift=shift, drop_remainder=True)
            .flat_map(lambda f_w, l_w: tf.data.Dataset.zip(
                (f_w.batch(FLAGS.seq_length), l_w.batch(FLAGS.seq_length)))),
            #.map(map_func=_get_transformation_parser(is_training)),
        cycle_length=1)
    if is_training:
        dataset = dataset.shuffle(100000).repeat()
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)

    return dataset


def _get_input_parser(table):

    """Return the input parser."""
    def input_parser(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, l):
        # Stack features
        features = tf.stack([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12], 0)
        features = tf.cast(features, tf.float32)
        # Map labels
        labels = table.lookup(l)
        return features, labels
    return input_parser


def _get_transformation_parser(is_training):

    def transformation_parser(features, labels):

        def _standardization(features):
            """Linearly scales feature data to have zero mean and unit variance."""
            num = tf.reduce_prod(tf.shape(features))
            mean = tf.reduce_mean(features)
            variance = tf.reduce_mean(tf.square(features)) - tf.square(mean)
            variance = tf.nn.relu(variance)
            stddev = tf.sqrt(variance)
            # Apply a minimum normalization
            min_stddev = tf.rsqrt(tf.cast(num, dtype=tf.float32))
            feature_value_scale = tf.maximum(stddev, min_stddev)
            feature_value_offset = mean
            features = tf.subtract(features, feature_value_offset)
            features = tf.divide(features, feature_value_scale)
            return features

        features = _standardization(features)

        return features, labels

    return transformation_parser


def predict_and_export_csv(estimator, eval_input_fn, eval_dir, seq_skip):
    tf.logging.info("Working on {0}".format(eval_dir))
    tf.logging.info("Starting prediction...")
    predictions = estimator.predict(input_fn=eval_input_fn)
    pred_list = list(itertools.islice(predictions, None))
    pred_probs_1 = list(map(lambda item: item["probabilities"][1], pred_list))
    num = len(pred_probs_1)
    # Get labels and ids
    filenames = gfile.Glob(os.path.join(eval_dir, ".csv"))
    select_cols = [0, 15]; record_defaults = [tf.int32, tf.string]
    mapping_strings = tf.constant(["idle", "Intake"])
    table = tf.contrib.lookup.index_table_from_tensor(
        mapping=mapping_strings)
    with tf.Session() as sess:
        sess.run(table.init)
    def input_parser(seqNo, label):
        label = table.lookup(label)
        return seqNo, label
    dataset = tf.data.experimental.CsvDataset(
        filenames=tf.data.Dataset.list_files(filenames),
            record_defaults=record_defaults, select_cols=select_cols, header=True)
    elem = dataset.map(input_parser).make_one_shot_iterator().get_next()
    labels = []; seq_no = []; sess = tf.Session()
    for i in range(0, num + seq_skip):
        val = sess.run(elem)
        seq_no.append(val[0])
        labels.append(val[1])
    seq_no = seq_no[seq_skip:]; labels = labels[seq_skip:]
    assert (len(labels)==num), "Lengths must match"
    name = os.path.normpath(eval_dir).split(os.sep)[-1]
    tf.logging.info("Writing {0} examples to {1}.csv...".format(num, name))
    pred_array = np.column_stack((seq_no, labels, pred_probs_1))
    np.savetxt("{0}.csv".format(name), pred_array, delimiter=",", fmt=['%i','%i','%f'])


# Run
if __name__ == "__main__":
    tf.app.run(
        main=run_experiment
    )
