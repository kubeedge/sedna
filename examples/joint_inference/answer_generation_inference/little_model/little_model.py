import os
import time
import logging
from sedna.common.config import Context
from sedna.common.file_ops import FileOps
from sedna.core.joint_inference import JointInference
from interface import Estimator

LOG = logging.getLogger(__name__)

input_path = Context.get_parameters('input_text')
all_output_path = Context.get_parameters('all_examples_inference_output')
hard_example_edge_output_path = Context.get_parameters('hard_example_edge_inference_output')
hard_example_cloud_output_path = Context.get_parameters('hard_example_cloud_inference_output')

FileOps.clean_folder([
    all_output_path,
    hard_example_cloud_output_path,
    hard_example_edge_output_path
], clean=False)


def output_deal(final_result, is_hard_example, cloud_result, edge_result, filename, input_text):
    with open(os.path.join(all_output_path, f"{filename}.txt"), 'w') as f:
        f.write(f"Input: {input_text}\nOutput: {final_result}\n")

    if not is_hard_example:
        return
    if cloud_result is not None:
        with open(os.path.join(hard_example_cloud_output_path, f"{filename}.txt"), 'w') as f:
            f.write(f"Input: {input_text}\nCloud Output: {cloud_result}\n")
    if edge_result is not None:
        with open(os.path.join(hard_example_edge_output_path, f"{filename}.txt"), 'w') as f:
            f.write(f"Input: {input_text}\nEdge Output: {edge_result}\n")

def get_texts_from_file(file_path):
    texts = []
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                texts.append(line.strip())
    return texts

class textExtractor:
    def __init__(self, input_path):
        self.input_path = input_path
        self.processed_files = set()
    
    def get_new_texts(self):
        new_texts = []
        if os.path.isdir(self.input_path):
            for filename in os.listdir(self.input_path):
                file_path = os.path.join(self.input_path, filename)
                if file_path not in self.processed_files:
                    new_texts.extend(get_texts_from_file(file_path))
                    self.processed_files.add(file_path)
        else:
            if self.input_path not in self.processed_files:
                new_texts.extend(get_texts_from_file(self.input_path))
                self.processed_files.add(self.input_path)
        return new_texts

def main():
    hard_example_mining = JointInference.get_hem_algorithm_from_config()
    inference_instance = JointInference(
        estimator=Estimator,
        hard_example_mining=hard_example_mining
    )
    text_extractor = textExtractor(input_path)
    
    while True:
        input_texts = text_extractor.get_new_texts()
        for idx, input_text in enumerate(input_texts):
            is_hard_example, final_result, edge_result, cloud_result = (
                inference_instance.inference([input_text])
            )
            output_deal(
                final_result,
                is_hard_example,
                cloud_result,
                edge_result,
                str(idx),
                input_text
            )
        if not input_texts:
            time.sleep(2)

if __name__ == '__main__':
    main() 
