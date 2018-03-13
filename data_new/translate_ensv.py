from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf

_ENDE_TRAIN_DATASETS = [
    [
        "/data/abhinav/T2T/Swedish/en2sv",  # pylint: disable=line-too-long
        ("europarl-v7.train.sv-en.en",
         "europarl-v7.train.sv-en.sv")
    ],
]
_ENDE_TEST_DATASETS = [
    [
        "/data/abhinav/T2T/Swedish/en2sv",
        ("europarl-v7.test.sv-en.en",
         "europarl-v7.test.sv-en.sv")
    ],
]


@registry.register_problem
class TranslateEnsv(translate.TranslateProblem):
    """Problem spec for WMT En-De translation."""

    @property
    def approx_vocab_size(self):
        return 2 ** 15  # 32k

    @property
    def vocab_filename(self):
        return "vocab.ensv.%d" % self.approx_vocab_size

    def source_data_files(self, dataset_split):
        train = dataset_split == problem.DatasetSplit.TRAIN
        return _ENDE_TRAIN_DATASETS if train else _ENDE_TEST_DATASETS

    def vocab_data_files(self):
        """Files to be passed to get_or_generate_vocab."""
        return self.source_data_files(problem.DatasetSplit.TRAIN)

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        datasets = self.source_data_files(dataset_split)
        tag = "train" if dataset_split == problem.DatasetSplit.TRAIN else "dev"
        data_path = compile_data(tmp_dir, datasets, "%s-compiled-%s" % (self.name,
                                                                        tag))

        if self.vocab_type == text_problems.VocabType.SUBWORD:
            get_or_generate_vocab(
                data_dir, tmp_dir, self.vocab_filename, self.approx_vocab_size,
                self.vocab_data_files())

        return text_problems.text2text_txt_iterator(data_path + ".en",
                                                    data_path + ".sv")


def get_or_generate_vocab(data_dir, tmp_dir, vocab_filename, vocab_size,
                          sources, file_byte_budget=1e7):
    """Generate a vocabulary from the datasets in sources."""

    def generate():
        tf.logging.info("Generating vocab from: %s", str(sources))
        for source in sources:
            path = source[0]

            for lang_file in source[1]:
                tf.logging.info("Reading file: %s" % lang_file)
                filepath = os.path.join(path, lang_file)

                # Use Tokenizer to count the word occurrences.
                with tf.gfile.GFile(filepath, mode="r") as source_file:
                    file_byte_budget_ = file_byte_budget
                    counter = 0
                    countermax = int(source_file.size() / file_byte_budget_ / 2)
                    for line in source_file:
                        if counter < countermax:
                            counter += 1
                        else:
                            if file_byte_budget_ <= 0:
                                break
                            line = line.strip()
                            file_byte_budget_ -= len(line)
                            counter = 0
                            yield line

    return generator_utils.get_or_generate_vocab_inner(data_dir, vocab_filename, vocab_size,
                                       generate())


def compile_data(tmp_dir, datasets, filename):
    """Concatenate all `datasets` and save to `filename`."""
    filename = os.path.join(tmp_dir, filename)
    lang1_fname = filename + ".en"
    lang2_fname = filename + ".sv"
    if tf.gfile.Exists(lang1_fname) and tf.gfile.Exists(lang2_fname):
        tf.logging.info("Skipping compile data, found files:\n%s\n%s", lang1_fname,
                        lang2_fname)

    with tf.gfile.GFile(lang1_fname, mode="w") as lang1_resfile:
        with tf.gfile.GFile(lang2_fname, mode="w") as lang2_resfile:
            for dataset in datasets:
                path = dataset[0]

                lang1_filename, lang2_filename = dataset[1]
                lang1_filepath = os.path.join(path, lang1_filename)
                lang2_filepath = os.path.join(path, lang2_filename)

                is_sgm = (
                        lang1_filename.endswith("sgm") and lang2_filename.endswith("sgm"))

                for example in text_problems.text2text_txt_iterator(
                        lang1_filepath, lang2_filepath):
                    line1res = _preprocess_sgm(example["inputs"], is_sgm)
                    line2res = _preprocess_sgm(example["targets"], is_sgm)
                    if line1res and line2res:
                        lang1_resfile.write(line1res)
                        lang1_resfile.write("\n")
                        lang2_resfile.write(line2res)
                        lang2_resfile.write("\n")

    return filename


def _preprocess_sgm(line, is_sgm):
    """Preprocessing to strip tags in SGM files."""
    if not is_sgm:
        return line
    # In SGM files, remove <srcset ...>, <p>, <doc ...> lines.
    if line.startswith("<srcset") or line.startswith("</srcset"):
        return ""
    if line.startswith("<doc") or line.startswith("</doc"):
        return ""
    if line.startswith("<p>") or line.startswith("</p>"):
        return ""
    # Strip <seg> tags.
    line = line.strip()
    if line.startswith("<seg") and line.endswith("</seg>"):
        i = line.index(">")
        return line[i + 1:-6]  # Strip first <seg ...> and last </seg>.
