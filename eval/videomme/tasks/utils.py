# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Video-MME long-video reasoning subset, served from local files.

Only the document selection and the video location are this task's own; the
prompt, the answer extraction and the scoring are lmms-eval's `videomme`.
"""

import os

from lmms_eval.tasks.videomme.utils import (videomme_aggregate_results,
                                            videomme_doc_to_text,
                                            videomme_process_results)

TASK_TYPES = frozenset({
    "Temporal Reasoning",
    "Information Synopsis",
    "Action Reasoning",
    "Object Reasoning",
})


def process_docs(dataset):
    return dataset.filter(lambda doc: doc["duration"] == "long" and doc[
        "task_type"] in TASK_TYPES)


def doc_to_visual(doc):
    path = os.path.join(os.environ["VIDEOMME_VIDEO_DIR"],
                        doc["videoID"] + ".mp4")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return [path]


doc_to_text = videomme_doc_to_text
process_results = videomme_process_results
aggregate_results = videomme_aggregate_results
