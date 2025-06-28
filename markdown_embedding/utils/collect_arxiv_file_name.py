import os
import re
import json
from collections import defaultdict


root_dir = "/RLS002/Public/arxiv/process_script/arxiv2markdown/output"
example_record_file = "/RLS002/Public/arxiv/process_script/markdown2embeddings/markdown_embedding/temp_files/each_name_example_files_records.json"
counter_file = "/RLS002/Public/arxiv/process_script/markdown2embeddings/markdown_embedding/temp_files/name_counter_records.json"
name_counter = defaultdict(int)
example_recorder = defaultdict(list)
with os.scandir(root_dir) as it:
    for entry in it:
        if entry.is_dir():
            arxiv_id = entry.path.rsplit("/", 1)[-1]
            name = re.sub(r"\d", "", arxiv_id)
            name_counter[name] += 1
            if name_counter[name] <= 3:
                example_recorder[name].append(entry.path)

with open(counter_file, "w") as f:
    json.dump(name_counter, f, indent=4)
with open(example_record_file, "w") as f:
    json.dump(example_recorder, f, indent=4)

example_dir = "/RLS002/Public/arxiv/process_script/markdown2embeddings/example_markdown"
for key in example_recorder:
    for example in example_recorder[key]:
        os.system(f"cp -r {example} {example_dir}")