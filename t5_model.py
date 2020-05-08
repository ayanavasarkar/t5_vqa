import gzip
import json

# Public directory of Natural Questions data on GCS.
NQ_JSONL_DIR = "gs://natural_questions/v1.0-simplified/"
NQ_SPLIT_FNAMES = {
    "train": "simplified-nq-train.jsonl.gz",
    "validation": "nq-dev-all.jsonl.gz"
}
nq_counts_path = os.path.join(DATA_DIR, "nq-counts.json")
nq_tsv_path = {
    "train": os.path.join(DATA_DIR, "nq-train.tsv"),
    "validation": os.path.join(DATA_DIR, "nq-validation.tsv")
}

def nq_jsonl_to_tsv(in_fname, out_fname):

  def extract_answer(tokens, span):
    """Reconstruct answer from token span and remove extra spaces."""
    start, end = span["start_token"], span["end_token"]  
    ans = " ".join(tokens[start:end])
    # Remove incorrect spacing around punctuation.
    ans = ans.replace(" ,", ",").replace(" .", ".").replace(" %", "%")
    ans = ans.replace(" - ", "-").replace(" : ", ":").replace(" / ", "/")
    ans = ans.replace("( ", "(").replace(" )", ")")
    ans = ans.replace("`` ", "\"").replace(" ''", "\"")
    ans = ans.replace(" 's", "'s").replace("s ' ", "s' ")
    return ans

  count = 0
  with tf.io.gfile.GFile(in_fname, "rb") as infile,\
       tf.io.gfile.GFile(out_fname, "w") as outfile:
    for line in gzip.open(infile):
      ex = json.loads(line)
      # Remove any examples with more than one answer.
      if len(ex['annotations'][0]['short_answers']) != 1:
        continue
      # Questions in NQ do not include a question mark.
      question = ex["question_text"] + "?"
      answer_span = ex['annotations'][0]['short_answers'][0]
      # Handle the two document formats in NQ (tokens or text).
      if "document_tokens" in ex:
        tokens = [t["token"] for t in ex["document_tokens"]]
      elif "document_text" in ex:
        tokens = ex["document_text"].split(" ")
      answer = extract_answer(tokens, answer_span)
      # Write this line as <question>\t<answer>
      outfile.write("%s\t%s\n" % (question, answer))
      count += 1
      tf.logging.log_every_n(
          tf.logging.INFO,
          "Wrote %d examples to %s." % (count, out_fname),
          1000)
    return count

if tf.io.gfile.exists(nq_counts_path):
  # Used cached data and counts.
  tf.logging.info("Loading NQ from cache.")
  num_nq_examples = json.load(tf.io.gfile.GFile(nq_counts_path))
else:
  # Create TSVs and get counts.
  tf.logging.info("Generating NQ TSVs.")
  num_nq_examples = {}
  for split, fname in NQ_SPLIT_FNAMES.items():
    num_nq_examples[split] = nq_jsonl_to_tsv(
        os.path.join(NQ_JSONL_DIR, fname), nq_tsv_path[split])
  json.dump(num_nq_examples, tf.io.gfile.GFile(nq_counts_path, "w"))


def nq_dataset_fn(split, shuffle_files=False):
  # We only have one file for each split.
  del shuffle_files

  # Load lines from the text file as examples.
  ds = tf.data.TextLineDataset(nq_tsv_path[split])
  # Split each "<question>\t<answer>" example into (question, answer) tuple.
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # Map each tuple to a {"question": ... "answer": ...} dict.
  ds = ds.map(lambda *ex: dict(zip(["question", "answer"], ex)))
  return ds

print("A few raw validation examples...")
for ex in tfds.as_numpy(nq_dataset_fn("validation").take(5)):
  print(ex)

def trivia_preprocessor(ds):
  def normalize_text(text):
    """Lowercase and remove quotes from a TensorFlow string."""
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
    return text

  def to_inputs_and_targets(ex):
    """Map {"question": ..., "answer": ...}->{"inputs": ..., "targets": ...}."""
    return {
        "inputs":
             tf.strings.join(
                 ["trivia question: ", normalize_text(ex["question"])]),
        "targets": normalize_text(ex["answer"])
    }
  return ds.map(to_inputs_and_targets, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

t5.data.TaskRegistry.add(
    "nq_context_free",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=nq_dataset_fn,
    splits=["train", "validation"],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=[trivia_preprocessor],
    # Use the same vocabulary that we used for pre-training.
    sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
    # Lowercase targets before computing metrics.
    postprocess_fn=t5.data.postprocessors.lower_text, 
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy],
    # Not required, but helps for mixing and auto-caching.
    num_input_examples=num_nq_examples
)

nq_task = t5.data.TaskRegistry.get("nq_context_free")
ds = nq_task.get_dataset(split="validation", sequence_length={"inputs": 128, "targets": 32})
print("A few preprocessed validation examples...")
for ex in tfds.as_numpy(ds.take(5)):
  print(ex)

ds = tfds.load(
    "trivia_qa/unfiltered.nocontext",
    data_dir=DATA_DIR,
    # Download data locally for preprocessing to avoid using GCS space.
    download_and_prepare_kwargs={"download_dir": "./downloads"})
print("A few raw validation examples...")
for ex in tfds.as_numpy(ds["validation"].take(2)):
  print(ex)

def tiviaqa_extract_qa(ds):
  def exract_qa(ex):
    return {
        "question": ex["question"],
        "answer": ex["answer"]["value"]
    }
  return ds.map(exract_qa, num_parallel_calls=tf.data.experimental.AUTOTUNE)

t5.data.TaskRegistry.add(
    "triviaqa_context_free",
    # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
    t5.data.TfdsTask,
    tfds_name="trivia_qa/unfiltered.nocontext:1.1.0",
    tfds_data_dir=DATA_DIR,
    sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
    text_preprocessor=[tiviaqa_extract_qa, trivia_preprocessor],
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy]
)

# Load and print a few examples.
triviaqa_task = t5.data.TaskRegistry.get("triviaqa_context_free")
ds = triviaqa_task.get_dataset(split="validation", sequence_length={"inputs": 128, "targets": 32})
print("A few preprocessed validation examples...")
for ex in tfds.as_numpy(ds.take(3)):
  print(ex)

t5.data.MixtureRegistry.remove("trivia_all")
t5.data.MixtureRegistry.add(
    "trivia_all",
    ["nq_context_free", "triviaqa_context_free"],
     default_rate=1.0
)

if ON_CLOUD:
  %reload_ext tensorboard
  import tensorboard as tb
tb.notebook.start("--logdir " + MODELS_DIR)

FINETUNE_STEPS = 25000 #@param {type: "integer"}

model.finetune(
    mixture_or_task_name="trivia_all",
    pretrained_model_dir=PRETRAINED_DIR,
    finetune_steps=FINETUNE_STEPS
)

# Use a larger batch size for evaluation, which requires less memory.
model.batch_size = train_batch_size * 4
model.eval(
    mixture_or_task_name="trivia_all",
    checkpoint_steps="all"
)

import random

def print_random_predictions(task_name, n=10):
  """Print n predictions from the validation split of a task."""
  # Grab the dataset for this task.
  ds = t5.data.TaskRegistry.get(task_name).get_dataset(
      split="validation",
      sequence_length={"inputs": 128, "targets": 32},
      shuffle=False)

  def _prediction_file_to_ckpt(path):
    """Extract the global step from a prediction filename."""
    return int(path.split("_")[-2])

  # Grab the paths of all logged predictions.
  prediction_files = tf.io.gfile.glob(
      os.path.join(
          MODEL_DIR,
          "validation_eval/%s_*_predictions" % task_name))
  # Get most recent prediction file by sorting by their step.
  latest_prediction_file = sorted(
      prediction_files, key=_prediction_file_to_ckpt)[-1]

  # Collect (inputs, targets, prediction) from the dataset and predictions file
  results = []
  with tf.io.gfile.GFile(latest_prediction_file) as preds:
    for ex, pred in zip(tfds.as_numpy(ds), preds):
      results.append((tf.compat.as_text(ex["inputs_plaintext"]),
                      tf.compat.as_text(ex["targets_plaintext"]),
                      pred.strip()))

  print("<== Random predictions for %s using checkpoint %s ==>\n" %
        (task_name, 
         _prediction_file_to_ckpt(latest_prediction_file)))

  for inp, tgt, pred in random.choices(results, k=10):
    print("Input:", inp)
    print("Target:", tgt)
    print("Prediction:", pred)
    print("Counted as Correct?", tgt == pred)
    print()

print_random_predictions("triviaqa_context_free")
print_random_predictions("nq_context_free")


