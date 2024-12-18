import argparse
from transformers import WhisperProcessor
from datasets import load_dataset, Dataset, load_from_disk
from functools import partial
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from pathlib import Path
import time
from contextlib import contextmanager
from jiwer import wer, wer_standardize
from tqdm.notebook import tqdm
import os
#################################################################
# Load model
#################################################################
print("Build model")

parser = argparse.ArgumentParser(description="""Run perf test for generative models""", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-d', '--device', help='Target device', type=str, required=False, default="GPU")
parser.add_argument('-m', '--model_dir', help='Model dir', type=str, required=False, default="C:\dev\models\whisper-base-nonstateful")
args = parser.parse_args()

if Path(args.model_dir).exists() == False:
    model = OVModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base", export=True, device=args.device, use_cache=True, local_files_only=True)
    model.save_pretrained(args.model_dir)

test_data_path = os.path.join(args.model_dir, 'dataset', 'librispeech_asr')
ov_model = OVModelForSpeechSeq2Seq.from_pretrained(args.model_dir, device=args.device, local_files_only=True)
processor = WhisperProcessor.from_pretrained(args.model_dir, local_files_only=True)
print("Model loaded")

TEST_DATASET_SIZE = 50
MEASURE_TIME = False

@contextmanager
def time_measurement():
    global MEASURE_TIME
    try:
        MEASURE_TIME = True
        yield
    finally:
        MEASURE_TIME = False

def time_fn(obj, fn_name, time_list):
    original_fn = getattr(obj, fn_name)

    def wrapper(*args, **kwargs):
        if not MEASURE_TIME:
            return original_fn(*args, **kwargs)
        start_time = time.perf_counter()
        result = original_fn(*args, **kwargs)
        end_time = time.perf_counter()
        time_list.append(end_time - start_time)
        return result

    setattr(obj, fn_name, wrapper)


def extract_input_features(sample):
    input_features = processor(
        sample["audio"]["array"],
        sampling_rate=sample["audio"]["sampling_rate"],
        return_tensors="pt",
    ).input_features
    return input_features

def calculate_transcription_time_and_accuracy(ov_model, test_samples):
    encoder_infer_times = []
    decoder_with_past_infer_times = []
    decoder_without_past_infer_times = []
    whole_infer_times = []
    time_fn(ov_model, "generate", whole_infer_times)
    time_fn(ov_model.encoder, "forward", encoder_infer_times)
    time_fn(ov_model.decoder_with_past, "forward", decoder_with_past_infer_times)
    time_fn(ov_model.decoder, "forward", decoder_without_past_infer_times)

    ground_truths = []
    predictions = []
    total_tokens = 0
    data_idx = 0
#    print("Start generation")
    for data_item in tqdm(test_samples, desc="Measuring performance and accuracy"):
#        print(f"Processing {data_idx}-th data")
        data_idx = data_idx + 1
        input_features = extract_input_features(data_item)

        with time_measurement():
            predicted_ids = ov_model.generate(input_features)
            total_tokens += predicted_ids.shape[1]
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        ground_truths.append(data_item["text"])
        predictions.append(transcription[0])
#    print("...Finished processing")
    word_accuracy = (1 - wer(ground_truths, predictions, reference_transform=wer_standardize,
                             hypothesis_transform=wer_standardize)) * 100
    whole_infer_time = sum(whole_infer_times)
    encoder_infer_time = sum(encoder_infer_times)
    decoder_with_time_infer_time = sum(decoder_with_past_infer_times) + sum(decoder_without_past_infer_times)
    tps = float(total_tokens) / whole_infer_time
    return word_accuracy, (whole_infer_time, encoder_infer_time, decoder_with_time_infer_time), tps, total_tokens


def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds

if Path(test_data_path).exists() == False:
    print(f"Test data dir not found. Download from repository")
    test_dataset = load_dataset("openslr/librispeech_asr", "clean", split="validation", streaming=True, trust_remote_code=True)
    test_dataset = test_dataset.take(TEST_DATASET_SIZE)
    ds = Dataset.from_generator(partial(gen_from_iterable_dataset, test_dataset), features=test_dataset.features)
    ds.save_to_disk(test_data_path)
else:
    print(f"Test data found from {test_data_path}. Load from existing path")
    test_dataset = load_from_disk(test_data_path)

test_samples = [sample for sample in test_dataset]

accuracy, times, tps, total_tokens = calculate_transcription_time_and_accuracy(ov_model, test_samples)
print(f"Encoder time: {times[1]:.3f}")
print(f"Decoder time: {times[2]:.3f}")
print(f"Whole pipeline time: {times[0]:.3f}")
print(f"Total generated tokens : {total_tokens}")
print(f"tps : {tps:.3f}")
print(f"Accuracy {accuracy:.2f}%")
