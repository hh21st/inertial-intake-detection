import os
import itertools
import numpy as np
import tensorflow as tf
import math
import best_checkpoint_exporter
import resnet_cnn
import resnet_cnn_lstm
import small_cnn
import kyritsis
import cnn_lstm
import cnn_gru
import cnn_blstm
import fusion
from tensorflow.python.platform import gfile

tf.logging.set_verbosity(tf.logging.INFO)
NUM_SHARDS = 10
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer(
    name='batch_size', default=32, help='Batch size used for training.')
tf.app.flags.DEFINE_string(
    name='eval_dir', default='', help='Directory for eval data.')
tf.app.flags.DEFINE_string(
    name='train_dir', default='', help='Directory for training data.')
tf.app.flags.DEFINE_string(
    name='prob_dir', default='', help='Directory for eval data.')
tf.app.flags.DEFINE_enum(
    name='mode', default='train_and_evaluate', enum_values=['train_and_evaluate', 'predict_and_export_csv'],
    help='What mode should tensorflow be started in')
tf.app.flags.DEFINE_enum(
    name='fusion', default='earliest', enum_values=['none', 'earliest', 'accel_gyro', 'dom_ndom', 'accel_gyro_dom_ndom'],
    help='Select the model')
tf.app.flags.DEFINE_enum(
    name='f_strategy', default='earliest', enum_values=['earliest', 'early', 'early_merge_cnn', 'early_merge_rnn', 'late'],
    help='Select the fusion strategy')
tf.app.flags.DEFINE_string(
    name='f_mode', default='',
    help='Select the mode of the proposed fusion model')
tf.app.flags.DEFINE_enum(
    name='model', default='cnn_rnn', enum_values=['resnet_cnn', 'resnet_cnn_lstm', 'small_cnn', 'kyritsis', 'cnn_lstm', 'cnn_gru', 'cnn_blstm', 'cnn_rnn'],
    help='Select the model')
tf.app.flags.DEFINE_string(
    name='sub_mode', default='d:4;ks:1357;pad:valid|d:2;t:l',
    help='Select the mode of the proposed cnn_lstm, cnn_gru or cnn_blstm model')
tf.app.flags.DEFINE_string(
    name='model_dir', default='run',
    help='Output directory for model and training stats.')
tf.app.flags.DEFINE_integer(
    name='num_sequences', default=3175419, help='Number of training example steps.')
tf.app.flags.DEFINE_integer(
    name='seq_length', default=128,
    help='Number of sequence elements.')
tf.app.flags.DEFINE_integer(
    name='seq_pool', default=1, help='Factor of sequence pooling in the model.')
tf.app.flags.DEFINE_integer(
    name='seq_shift', default=1, help='Shift taken in sequence generation.')
tf.app.flags.DEFINE_float(
    name='train_epochs', default=60, help='Number of training epochs.')
tf.app.flags.DEFINE_boolean(
    name='use_sequence_loss', default=False,
    help='Use sequence-to-sequence loss')
tf.app.flags.DEFINE_float(
    name='base_learning_rate', default=3e-4, help='Base learning rate')
tf.app.flags.DEFINE_float(
    name='decay_rate', default=0.93, help='Decay rate of the learning rate.')
tf.app.flags.DEFINE_enum(
    name='hand', default='both', enum_values=['both', 'dom', 'nondom'],
    help='specified data from which hands are included in the input')
tf.app.flags.DEFINE_enum(
    name='modality', default='both', enum_values=['both', 'accel', 'gyro'],
    help='specified data from what modalities are included in the input')

def run_experiment(arg=None):
    """Run the experiment."""

    # Approximate steps per epoch
    steps_per_epoch = int(FLAGS.num_sequences / FLAGS.batch_size / FLAGS.seq_length)
    max_steps = steps_per_epoch * FLAGS.train_epochs

    # Model parameters
    params = tf.contrib.training.HParams(
        base_learning_rate=FLAGS.base_learning_rate,
        lowest_learning_rate = 2e-7,
        batch_size=FLAGS.batch_size,
        decay_rate=FLAGS.decay_rate,
        dropout=0.5,
        gradient_clipping_norm=10.0,
        l2_lambda=1e-4,
        num_classes=2,
        resnet_block_sizes=[3, 4, 6, 3],
        resnet_block_strides=[1, 2, 2, 2],
        resnet_conv_stride=2,
        resnet_first_pool_size=2,
        resnet_first_pool_stride=2,
        resnet_kernel_size=4,
        resnet_num_filters=64,
        small_kernel_size=10,
        small_num_filters=[64, 64, 128, 128, 256, 256, 512],
        fusion=FLAGS.fusion,
        f_mode=FLAGS.f_mode,
        f_strategy=FLAGS.f_strategy,
        model=FLAGS.model,
        sub_mode=FLAGS.sub_mode,
        small_pool_size=2,
        num_lstm=64,
        seq_length=FLAGS.seq_length,
        steps_per_epoch=steps_per_epoch)

    # Run config
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        save_summary_steps=2500,
        save_checkpoints_steps=5000)

    # Define the estimator
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.model_dir,
        params=params,
        config=run_config)

    # Exporters
    best_exporter = best_checkpoint_exporter.BestCheckpointExporter(
        score_metric='metrics/unweighted_average_recall',
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
        start_delay_secs=60,
        throttle_secs=60)

    # Start the experiment
    if FLAGS.mode == "train_and_evaluate":
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    elif FLAGS.mode == "predict_and_export_csv":
        seq_skip = FLAGS.seq_length - 1
        predict_and_export_csv(estimator, eval_input_fn, FLAGS.eval_dir, seq_skip)


def model_fn(features, labels, mode, params):
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    is_predicting = mode == tf.estimator.ModeKeys.PREDICT

    if FLAGS.hand != 'both' and FLAGS.modality != 'both':
        features_num = 3
    if FLAGS.hand == 'both' and FLAGS.modality != 'both':
        features_num = 6
    elif FLAGS.hand != 'both' and FLAGS.modality == 'both':
        features_num = 6
    elif FLAGS.hand == 'both' and FLAGS.modality == 'both':
        features_num = 12

    # Set features to correct shape
    features = tf.reshape(features, [params.batch_size, params.seq_length, features_num])
    padding_size = 0

    # Model
    if FLAGS.fusion != 'none':
        if FLAGS.mode == "train_and_evaluate":
            if FLAGS.f_strategy == 'earliest':
                assert FLAGS.fusion == 'earliest', "fusion strategy is not compatible with fusion model"
            elif FLAGS.f_strategy == 'early':
                assert FLAGS.fusion == 'accel_gyro' or FLAGS.fusion == 'dom_ndom' or FLAGS.fusion == 'accel_gyro_dom_ndom', "fusion strategy is not compatible with fusion model"
            elif FLAGS.f_strategy == 'early_merge_cnn':
                assert FLAGS.fusion == 'accel_gyro' or FLAGS.fusion == 'dom_ndom' or FLAGS.fusion == 'accel_gyro_dom_ndom', "fusion strategy is not compatible with fusion model"
            elif FLAGS.f_strategy == 'early_merge_rnn':
                assert FLAGS.fusion == 'accel_gyro' or FLAGS.fusion == 'dom_ndom' or FLAGS.fusion == 'accel_gyro_dom_ndom', "fusion strategy is not compatible with fusion model"
            elif FLAGS.f_strategy == 'late':
                assert FLAGS.fusion == 'accel_gyro' or FLAGS.fusion == 'dom_ndom' or FLAGS.fusion == 'accel_gyro_dom_ndom', "fusion strategy is not compatible with fusion model"
            assert FLAGS.model == 'cnn_rnn', "model is not compatible with modality"
        model = fusion.Model(params)
        FLAGS.seq_pool, padding_size, logits = model(features, is_training)
        # to see if the model is sequential
        sub_mode = params.sub_mode.split('|')[1]
        sub_mode_dict = dict(item.split(':') for item in sub_mode.split(';'))
        depth = int(sub_mode_dict['d']) if 'd' in sub_mode_dict else 2
        FLAGS.use_sequence_loss = False if depth == 0 else True
    elif FLAGS.model == 'resnet_cnn':
        if FLAGS.mode == "train_and_evaluate":
            assert not FLAGS.use_sequence_loss, "Cannot use sequence loss with this model"
        model = resnet_cnn.Model(params)
    elif FLAGS.model == 'resnet_cnn_lstm':
        if FLAGS.mode == "train_and_evaluate":
            assert FLAGS.use_sequence_loss, "Need sequence loss for this model"
            assert FLAGS.seq_pool == 16, "seq_pool should be 16"
        model = resnet_cnn_lstm.Model(params)
    elif FLAGS.model == 'small_cnn':
        if FLAGS.mode == "train_and_evaluate":
            assert not FLAGS.use_sequence_loss, "Cannot use sequence loss with this model"
        model = small_cnn.Model(params)
    elif FLAGS.model == 'kyritsis':
        if FLAGS.mode == "train_and_evaluate":
            assert FLAGS.use_sequence_loss, "Need sequence loss for this model"
            assert FLAGS.seq_pool == 4, "seq_pool should be 4"
        model = kyritsis.Model(params)
    elif FLAGS.model == 'cnn_lstm':
        FLAGS.use_sequence_loss = True
        model = cnn_lstm.Model(params)
        FLAGS.seq_pool, logits = model(features, is_training)
    elif FLAGS.model == 'cnn_gru':
        FLAGS.use_sequence_loss = True
        model = cnn_gru.Model(params)
        FLAGS.seq_pool, logits = model(features, is_training)
    elif FLAGS.model == 'cnn_blstm':
        FLAGS.use_sequence_loss = True
        model = cnn_blstm.Model(params)
        FLAGS.seq_pool, logits = model(features, is_training)

    if  FLAGS.fusion == 'none':
        if FLAGS.model != 'cnn_lstm' and FLAGS.model != 'cnn_gru' and FLAGS.model != 'cnn_blstm':
            logits = model(features, is_training)

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
    final_labels = labels

    if labels.get_shape().ndims == 2:
        seq_length = int((FLAGS.seq_length - padding_size) / FLAGS.seq_pool)
        # If the length of the sequence was reduced due to padding='valid' the labels has also to be trimmed respectively
        if padding_size > 0:
            labels = tf.strided_slice(input_=labels,
                begin=[0, math.ceil(padding_size/2)],
                end=[FLAGS.batch_size, FLAGS.seq_length-(math.floor(padding_size/2))],
                strides=[1, 1])
        # If necessary, slice last sequence step for labels
        final_labels = labels[:,-1]
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

        def _decay_fn(learning_rate, global_step, lowest_learning_rate):
            learning_rate = tf.train.exponential_decay(
                learning_rate=learning_rate, global_step=global_step,
                decay_steps=params.steps_per_epoch, decay_rate=params.decay_rate)
            return tf.cond(learning_rate >= lowest_learning_rate, lambda:learning_rate, lambda: lowest_learning_rate)

        # Learning rate
        learning_rate = _decay_fn(params.base_learning_rate, global_step, params.lowest_learning_rate)
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
    unweighted_average_recall = tf.metrics.mean_per_class_accuracy(
        labels=final_labels, predictions=predictions['classes'],
        num_classes=params.num_classes)
    tf.summary.scalar('metrics/accuracy', accuracy[1])
    tf.summary.scalar('metrics/unweighted_average_recall',
        tf.reduce_mean(unweighted_average_recall[1]))
    metrics = {
        'metrics/accuracy': accuracy,
        'metrics/unweighted_average_recall': unweighted_average_recall}

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
    mapping_strings = tf.constant(["Idle", "Intake"])
    table = tf.contrib.lookup.index_table_from_tensor(
        mapping=mapping_strings)
    # Initialize table
    with tf.Session() as sess:
        sess.run(table.init)
    # Shuffle files if needed
    if is_training:
        files = files.shuffle(NUM_SHARDS)
    select_cols = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16]
    record_defaults = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.string]
    dataset = files.interleave(
        lambda filename:
            tf.contrib.data.CsvDataset(filenames=filename,
                record_defaults=record_defaults, select_cols=select_cols,
                header=True)
            .map(map_func=_get_input_parser(table))
            .apply(_get_sequence_batch_fn(is_training))
            .map(map_func=_get_transformation_parser(is_training)),
        cycle_length=1)
    if is_training:
        dataset = dataset.shuffle(100000).repeat()
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)

    return dataset


def _get_input_parser(table):
    """Return the input parser."""
    def input_parser(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, l):

        # Stack features
        if FLAGS.hand == 'dom' and FLAGS.modality == 'accel' :
            features = tf.stack([f1, f2, f3], 0)
        elif FLAGS.hand == 'dom' and FLAGS.modality == 'gyro' :
            features = tf.stack([f4, f5, f6], 0)
        elif FLAGS.hand == 'dom' and FLAGS.modality == 'both' :
            features = tf.stack([f1, f2, f3, f4, f5, f6], 0)
        elif FLAGS.hand == 'nondom' and FLAGS.modality == 'accel' :
            features = tf.stack([f7, f8, f9], 0)
        elif FLAGS.hand == 'nondom' and FLAGS.modality == 'gyro' :
            features = tf.stack([f10, f11, f12], 0)
        elif FLAGS.hand == 'nondom' and FLAGS.modality == 'both' :
            features = tf.stack([f7, f8, f9, f10, f11, f12], 0)
        elif FLAGS.hand == 'both' and FLAGS.modality == 'accel' :
            features = tf.stack([f1, f2, f3, f7, f8, f9], 0)
        elif FLAGS.hand == 'both' and FLAGS.modality == 'gyro' :
            features = tf.stack([f4, f5, f6, f10, f11, f12], 0)
        else: # FLAGS.hand == 'both' and FLAGS.modality == 'both'
            features = tf.stack([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12], 0)
        features = tf.cast(features, tf.float32)
        # Map labels
        labels = table.lookup(l)
        return features, labels
    return input_parser


def _get_sequence_batch_fn(is_training):
    """Return sliding batched dataset or batched dataset."""
    if is_training:
        seq_shift = FLAGS.seq_shift
    else:
        seq_shift = 1
    if tf.__version__ < "1.13.1":
        return tf.contrib.data.sliding_window_batch(window_size=FLAGS.seq_length, window_shift=seq_shift)
    else:
        return lambda dataset: dataset.window(
            size=FLAGS.seq_length, shift=seq_shift, drop_remainder=True).flat_map(
                lambda f_w, l_w: tf.data.Dataset.zip(
                    (f_w.batch(FLAGS.seq_length), l_w.batch(FLAGS.seq_length))))


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
    filenames = gfile.Glob(os.path.join(eval_dir, "*.csv"))
    frame_id_index = 1
    label1_index = 16
    select_cols = [frame_id_index, label1_index]; record_defaults = [tf.int32, tf.string]
    mapping_strings = tf.constant(["Idle", "Intake"])
    table = tf.contrib.lookup.index_table_from_tensor(
        mapping=mapping_strings)
    def input_parser(seqNo, label):
        label = table.lookup(label)
        return seqNo, label
    if tf.__version__ < "1.13.1":
        dataset = tf.contrib.data.CsvDataset(filenames=filenames,
            record_defaults=record_defaults, select_cols=select_cols, header=True)
    else:
        dataset = tf.data.experimental.CsvDataset(filenames=filenames,
            record_defaults=record_defaults, select_cols=select_cols, header=True)
    iterator = dataset.map(input_parser).make_initializable_iterator()
    elem = iterator.get_next()
    labels = []; seq_no = []; sess = tf.Session()
    sess.run(iterator.initializer)
    sess.run(table.init)
    for i in range(0, num + seq_skip):
        val = sess.run(elem)
        seq_no.append(val[0])
        labels.append(val[1])
    seq_no = seq_no[seq_skip:]; labels = labels[seq_skip:]
    assert (len(labels)==num), "Lengths must match"
    name = os.path.normpath(eval_dir).split(os.sep)[-1]
    fullname = os.path.join(FLAGS.prob_dir,name)
    if os.path.isfile(fullname):
        tf.logging.info("{0} already exists! Skipping...".format(fullname))
    else:
        tf.logging.info("Writing {0} examples to {1}...".format(num, fullname))
        pred_array = np.column_stack((seq_no, labels, pred_probs_1))
        np.savetxt("{0}.csv".format(fullname), pred_array, delimiter=",", fmt=['%i','%i','%f'])


# Run
if __name__ == "__main__":
    tf.app.run(
        main=run_experiment
    )
