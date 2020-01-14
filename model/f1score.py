import tensorflow as tf
import os
import logging
import traceback
import utils
import main
import eval

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S', level=logging.INFO)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    name='root_dir', default='', help='root directory to find all .sh files')
tf.app.flags.DEFINE_boolean(
    name='overwrite', default=True, help='overwrite existing prob files and calculate the metrics if true')

def get_indexes_from_model_dir(model_dir):
    indexes = []
    for index_file in os.listdir(model_dir):
        if index_file.endswith('.index'):
            indexes.append(index_file[:-6].split('.')[1].split('-')[1].strip())
    return indexes

def move_index_files(index, source_dir, destination_dir):
    utils.move_file('model.ckpt-{0}.data-00000-of-00001'.format(index), source_dir, destination_dir)
    utils.move_file('model.ckpt-{0}.index'.format(index), source_dir, destination_dir)
    utils.move_file('model.ckpt-{0}.meta'.format(index), source_dir, destination_dir)
    
def create_checkpoint_file(checkpoint_file_fullname, index):
    if os.path.isfile(checkpoint_file_fullname):
        os.remove(checkpoint_file_fullname)
    checkpoint_file = open(checkpoint_file_fullname,"w")
    checkpoint_file.write('model_checkpoint_path: "model.ckpt-{0}"'.format(index))
    checkpoint_file.write('\n')
    checkpoint_file.write('all_model_checkpoint_paths: "model.ckpt-{0}"'.format(index))
    checkpoint_file.close()

def prepare_index_checkpoint(model_dir_bests):
    indexes = get_indexes_from_model_dir(model_dir_bests)
    for index in indexes:
        index_dir=os.path.join(model_dir_bests,index)
        if not os.path.exists(index_dir):
            os.mkdir(index_dir)
            logging.info("folder {} was created".format(index_dir))
        else:
            logging.info("folder {} has already been created".format(index_dir))
        move_index_files(index, model_dir_bests, index_dir)
    index_dirs = utils.get_immediate_subdirs(model_dir_bests)
    for index_dir in index_dirs:
        create_checkpoint_file(os.path.join(index_dir,'checkpoint'), utils.get_current_dir_name(index_dir))
    return indexes

def write_f1score_header(f1score_file_fullname):
    if os.path.isfile(f1score_file_fullname):
        return
    f1score_file_header = 'model,checkpoint index,F1,UAR,TP,FN,FP type 1,FP type 2,Precision,Recall,Best threshold,model path'
    f1score_file = open(f1score_file_fullname,'w')
    f1score_file.write(f1score_file_header)
    f1score_file.write('\n')
    f1score_file.close()
    logging.info("{} was created".format(f1score_file_fullname))

def write_f1score_line(f1score_file_fullname,model_desciption, index, f1, uar, tp, fn, fp_1, fp_2, precision, recall, best_threshold, model_dir):
    try:
        f1score_file = open(f1score_file_fullname,'a')
    except:
        file_nane, file_extension = os.path.splitext(f1score_file_fullname)
        f1score_file_fullname = file_nane+'1'+file_extension
        write_f1score_header(f1score_file_fullname)
        f1score_file = open(f1score_file_fullname)
    f1score_file_line = model_desciption
    f1score_file_line += ',' + str(index)
    f1score_file_line += ',' + str(f1)
    f1score_file_line += ',' + str(uar)
    f1score_file_line += ',' + str(tp)
    f1score_file_line += ',' + str(fn)
    f1score_file_line += ',' + str(fp_1)
    f1score_file_line += ',' + str(fp_2)
    f1score_file_line += ',' + str(precision)
    f1score_file_line += ',' + str(recall)
    f1score_file_line += ',' + str(best_threshold)
    f1score_file_line += ',' + model_dir
    f1score_file.write(f1score_file_line)
    f1score_file.write('\n')
    f1score_file.close()
    logging.info("values for model {} - index {} was added to the file".format(model_desciption, str(index)))

def calc_f1(eval_dir, model_dir, model, sub_mode, fusion, f_strategy, f_mode, f1score_file_fullname):
    FLAGS.model=model 
    FLAGS.sub_mode=sub_mode
    #FLAGS.decay_rate=.93 
    FLAGS.fusion=fusion
    FLAGS.f_strategy=f_strategy
    FLAGS.f_mode=f_mode
    #
    FLAGS.mode ='predict_and_export_csv'
    model_desciption = utils.get_current_dir_name(model_dir)
    model_dir_bests = os.path.join(model_dir,'best_checkpoints')
    prepare_index_checkpoint(model_dir_bests)
    #
    eval_dir_sub=eval_dir+'_sub'
    eval_dir_sub = eval_dir+'_sub' if FLAGS.eval_mode == 'estimate' else eval_dir.replace('eval','test',1)+'_sub' 
    logging.info("eval_dir_sub: {}".format(eval_dir_sub))
    eval_sub_dirs = utils.get_immediate_subdirs(eval_dir_sub)
    index_dirs = utils.get_immediate_subdirs(model_dir_bests)
    prob_suffix = 'prob' if FLAGS.eval_mode == 'estimate' else 'prob_test' 
    for index_dir in index_dirs:
        FLAGS.prob_dir=os.path.join(index_dir, prob_suffix)
        FLAGS.model_dir = index_dir
        try:
            if not os.path.exists(FLAGS.prob_dir):
                os.mkdir(FLAGS.prob_dir)
            for eval_sub_dir in eval_sub_dirs:
                if not FLAGS.overwrite and os.path.isfile(os.path.join(FLAGS.prob_dir,utils.get_current_dir_name(eval_sub_dir)+'.csv')):
                    continue
                FLAGS.eval_dir=eval_sub_dir
                main.run_experiment()
                logging.info("probabilities file for validation folder {} was added to folder {}".format(FLAGS.eval_dir, FLAGS.prob_dir))
        
            FLAGS.min_dist = 128
            FLAGS.col_label = 1
            FLAGS.col_prob = 2
            uar, tp, fn, fp_1, fp_2, precision, recall, f1, best_threshold = eval.main()
            if uar != -1:
                write_f1score_line(f1score_file_fullname,model_desciption,utils.get_current_dir_name(index_dir),f1,uar,tp,fn,fp_1,fp_2,precision,recall,best_threshold,model_dir)
        except Exception as e:
            logging.error(traceback.format_exc()) 
            write_f1score_line(f1score_file_fullname,model_desciption,utils.get_current_dir_name(index_dir),0,0,0,0,0,0,0,0,0,model_dir)

def read_sh_file(sh_file_fullname):
    def read_python_line(words):
        eval_dir, model_dir, model, sub_mode, fusion, f_strategy, f_mode = '','','','','','',''
        for word in words:
            if word.startswith('--model_dir'):
                model_dir = word.split('=')[1]
            elif word.startswith('--model'):
                model = word.split('=')[1]
            elif word.startswith('--sub_mode'):
                sub_mode = word.split('=')[1].strip('\"')
            elif word.startswith('--fusion'):
                fusion = word.split('=')[1]
            elif word.startswith('--f_strategy'):
                f_strategy = word.split('=')[1]
            elif word.startswith('--f_mode'):
                f_mode = word.split('=')[1]
            elif word.startswith('--eval_dir'):
                eval_dir = word.split('=')[1]
        if fusion == None or fusion.strip() == '' :
            fusion = 'none' 
        return eval_dir, model_dir, model, sub_mode, fusion, f_strategy, f_mode

    sh_file = open(sh_file_fullname,'r')
    content = sh_file.read()
    lines = content.splitlines()
    base_dir =''
    for line in lines:
        if line.startswith('cd'):
            base_dir = line.split()[1]
        elif line.startswith('python'):
            words=line.split()
            if words[0] == 'python' and words[1] == 'main.py':
                eval_dir, model_dir, model, sub_mode, fusion, f_strategy, f_mode = read_python_line(words)
                model_dir = os.path.join(base_dir,model_dir)
                return eval_dir, model_dir, model, sub_mode, fusion, f_strategy, f_mode
    return '___','___','___','___','___','___','___'


def calc_for_sh_file(sh_file_fullname, eval_dir, f1score_file_fullname):
    eval_dir_read, model_dir, model, sub_mode, fusion, f_strategy, f_mode = read_sh_file(sh_file_fullname)
    if eval_dir =='' or eval_dir == None:
        eval_dir = eval_dir_read
    if os.path.exists(model_dir):
        calc_f1(eval_dir, model_dir, model, sub_mode, fusion, f_strategy, f_mode, f1score_file_fullname)
    else:
        logging.info("path {} does not exists for file {}".format(model_dir, sh_file_fullname))

def calc(root_dir, eval_dir):
    dirs_info = [x for x in os.walk(root_dir)]
    f1score_file_name = 'f1scores.csv' if FLAGS.eval_mode == 'estimate' else 'f1scores_test.csv' 
    f1score_file_fullname = os.path.join(root_dir,f1score_file_name)
    write_f1score_header(f1score_file_fullname)
    for dir_info in dirs_info:
        dir=dir_info[0]
        files = dir_info[2]
        for file in files:
            try:
                if utils.get_file_extension(file) =='.sh':
                    sh_file_fullname = os.path.join(dir,file)
                    calc_for_sh_file(sh_file_fullname, eval_dir, f1score_file_fullname)
            except Exception as e:
                logging.error(traceback.format_exc())



if __name__ == '__main__':
    #if FLAGS.root_dir=='' or FLAGS.root_dir == None:
        #FLAGS.root_dir = r'\\10.2.224.9\c3140147\run'
        #FLAGS.root_dir = r'/home/c3140147/run'
    #if FLAGS.eval_dir=='' or FLAGS.eval_dir == None:
        #FLAGS.eval_dir = r'\\10.2.224.9\c3140147\input\OREBA\64_std_uni\smo_0\eval'
        #FLAGS.eval_dir = r'/home/c3140147/input/OREBA/64_std_uni/smo_0/eval'

    if utils.is_file(FLAGS.root_dir):
        assert utils.get_file_extension(FLAGS.root_dir) == '.sh', 'root_dir is a file therefore its extension has to be .sh, file name={0}'.format(FLAGS.root_dir)
        f1score_file_name_suffix = '_f1score.csv' if FLAGS.eval_mode == 'estimate' else '_f1score_test.csv' 
        f1score_file_fullname = utils.get_file_name_without_extension(FLAGS.root_dir)+ f1score_file_name_suffix
        calc_for_sh_file(FLAGS.root_dir, FLAGS.eval_dir, f1score_file_fullname)
    else:
        calc(FLAGS.root_dir,FLAGS.eval_dir)

    #f1score_file_fullname = os.path.join(FLAGS.root_dir,'f1scores.csv')
    #calc_for_sh_file(r'\\10.2.224.9\c3140147\run\20191002\20191017\est.d4ks1357d2tl.valid.cl.93.64_std_uni.smo_0.oldInput.sh',FLAGS.eval_dir, f1score_file_fullname)