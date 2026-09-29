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
"""lmms-eval's `vllm_generate` backend, set up to time Video-MME on TPU.

The task, its prompt and its scoring stay lmms-eval's. This backend changes
only what a one-request, one-token timing run needs from the model side:

* Frames are sized for Qwen3.5's 16-pixel patch. `vllm_generate` leaves
  qwen_vl_utils at Qwen2.5-VL's 14, so a frame is resized to a multiple of 28
  on the host and again to a multiple of 32 by the HF processor, and
  qwen_vl_utils' default 128k-token clip budget caps a 2048-frame video far
  below the requested frame size.
* The chat template is rendered with thinking off. Qwen3.5's template opens a
  `<think>` block unless told otherwise, and a one-token answer would then be
  the first token of that block.
* Every request carries a fresh multimodal UUID, so the encoder runs for each
  question instead of serving a video an earlier question already encoded.
* Each video is decoded once, between requests, never alongside a timed
  step: the encoder phase has a host side (replay buffers, host-to-device
  copies) that a concurrent decode slowed from ~9 s to ~18 s at 2048 frames.
* The vision encoder runs as one compiled program per token budget rather
  than op by op: an eager ViT over a 1024-frame clip keeps every intermediate
  of ~450k patches alive and runs out of HBM. The budgets are the exact token
  counts of the videos to serve, so no clip is padded or left to the eager
  path.
* Each request is one engine step (batch 1, no chunked prefill, one output
  token), and its encoder and prefill times are read back from the record the
  runner writes for that step under STEP_TIMING_LOG_PATH. A request whose
  call traced or compiled anything is served again and timed on that pass.
"""

import dataclasses
import json
import os
import time
import uuid
from typing import Any

import cv2
import jax
from lmms_eval.api.instance import GenerationResult
from lmms_eval.models.chat.vllm_generate import VLLMGenerate
from lmms_eval.protocol import ChatMessages
from qwen_vl_utils import vision_process
from tqdm import tqdm
from vllm import SamplingParams

# Qwen3.5's ViT: 16-pixel patches, 2x2 spatial merge, 2-frame temporal patch.
IMAGE_PATCH_SIZE = 16
SPATIAL_MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
TOKEN_EDGE = IMAGE_PATCH_SIZE * SPATIAL_MERGE_SIZE

# Every timed request relies on these: one request per step, its whole prompt
# in that step, and nothing carried over from an earlier request.
REQUIRED_ENGINE_ARGS = {
    "max_num_seqs": 1,
    "enable_chunked_prefill": False,
    "enable_prefix_caching": False,
    "async_scheduling": False,
}
# These follow from the request shape, so the backend sets them itself.
OWNED_ENGINE_ARGS = ("limit_mm_per_prompt", "mm_processor_kwargs",
                     "mm_processor_cache_gb", "compilation_config")

# Tracing, compiling or loading a compiled program marks the call it happened
# in as unrepresentative of steady-state latency.
COMPILE_EVENTS = frozenset({
    "/jax/core/compile/jaxpr_trace_duration",
    "/jax/core/compile/backend_compile_duration",
    "/jax/compilation_cache/cache_retrieval_time_sec",
})

KERNEL_ENV = ("USE_FLYWHEEL_TPU_KERNEL", "USE_FLYWHEEL_TPU_LINEAR_KERNEL",
              "VIT_SDPA_SEGMENTS_AS_BATCH", "RPA_V3_MIXED_BLOCK_SIZES",
              "RPA_V3_PREFILL_BLOCK_SIZES", "RPA_V3_DECODE_BLOCK_SIZES",
              "VLLM_TPU_BUCKET_PADDING_GAP", "MODEL_IMPL_TYPE")


def _refuse_torchvision_reader(ele: dict) -> Any:
    # fetch_video retries a failed decord read with torchvision, which decodes
    # the whole video into host memory: ~290 GB for an hour of 720p.
    raise RuntimeError(f"decord could not read {ele['video']}")


def _visual_tokens(nframes: int, height: int, width: int) -> int:
    return (nframes // TEMPORAL_PATCH_SIZE) * (height // TOKEN_EDGE) * (
        width // TOKEN_EDGE)


def _encoder_token_budgets(video_dir: str, nframes: int, min_pixels: int,
                           max_pixels: int) -> list[int]:
    """The visual token count of every video in `video_dir`.

    Frames are sized by the same smart_resize fetch_video applies, from the
    container's frame size, so each video's clip lands on a budget exactly.
    """
    budgets = set()
    for name in sorted(os.listdir(video_dir)):
        capture = cv2.VideoCapture(os.path.join(video_dir, name))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        capture.release()
        if not height or not width:
            raise RuntimeError(f"cannot read the frame size of {name}")
        height, width = vision_process.smart_resize(height,
                                                    width,
                                                    factor=TOKEN_EDGE,
                                                    min_pixels=min_pixels,
                                                    max_pixels=max_pixels)
        budgets.add(_visual_tokens(nframes, height, width))
    return sorted(budgets)


@dataclasses.dataclass
class DecodedVideo:
    frames: Any  # uint8-valued float tensor [T, C, H, W]
    metadata: dict
    sample_fps: float
    decode_seconds: float


@dataclasses.dataclass
class RequestPlan:
    question_id: str
    doc: dict
    video_path: str
    prompt: str
    params: dict


class VLLMGenerateTPU(VLLMGenerate):

    def __init__(self,
                 request_log: str,
                 nframes: int,
                 max_pixels: int,
                 min_image_pixels: int = 28,
                 batch_size: int = 1,
                 max_new_tokens: int = 1,
                 fps: float | None = None,
                 **kwargs):
        if fps is not None:
            raise ValueError("Sample a fixed frame count with nframes; fps "
                             "sampling gives every video its own length.")
        if int(batch_size) != 1 or int(max_new_tokens) != 1:
            raise ValueError("One request per step and one output token make "
                             "each request exactly one timed engine step; got "
                             f"batch_size={batch_size}, "
                             f"max_new_tokens={max_new_tokens}.")
        for name, value in REQUIRED_ENGINE_ARGS.items():
            if kwargs.get(name, None) != value:
                raise ValueError(f"model_args must set {name}={value}; got "
                                 f"{kwargs.get(name)!r}.")
        for name in OWNED_ENGINE_ARGS:
            if name in kwargs:
                raise ValueError(f"{name} is set by this backend.")
        if kwargs.get("disable_log_stats"):
            raise ValueError("Per-request TTFT comes from vLLM's request "
                             "stats; leave disable_log_stats off.")
        self._step_log_path = os.environ.get("STEP_TIMING_LOG_PATH", "")
        if not self._step_log_path:
            raise ValueError("STEP_TIMING_LOG_PATH must name the log the "
                             "runner writes its per-step timing to.")
        if vision_process.get_video_reader_backend() != "decord":
            raise ValueError("Set FORCE_QWENVL_VIDEO_READER=decord.")
        vision_process.VIDEO_READER_BACKENDS[
            "torchvision"] = _refuse_torchvision_reader

        nframes = int(nframes)
        max_pixels = int(max_pixels)
        # The HF video processor caps the pixels of the whole clip, 25M by
        # default, which would shrink the frames again. Frames arrive sized to
        # a multiple of 32 and at most max_pixels each, so a cap of exactly
        # nframes * max_pixels leaves them as they are. `model` is the local
        # snapshot, whose own config supplies the rest of the size.
        with open(os.path.join(kwargs["model"],
                               "video_preprocessor_config.json"),
                  encoding="utf-8") as f:
            size = json.load(f)["size"]
        size["longest_edge"] = nframes * max_pixels
        encoder_budgets = _encoder_token_budgets(
            os.environ["VIDEOMME_VIDEO_DIR"], nframes, int(min_image_pixels),
            max_pixels)
        super().__init__(
            batch_size=1,
            max_new_tokens=1,
            nframes=nframes,
            max_pixels=max_pixels,
            min_image_pixels=int(min_image_pixels),
            limit_mm_per_prompt={
                "image": 0,
                "video": 1
            },
            mm_processor_kwargs={"size": size},
            # Every request's UUID is new, so a processed-input cache never
            # hits and only holds a ~3 GB entry per request.
            mm_processor_cache_gb=0,
            compilation_config={
                "cudagraph_mm_encoder": True,
                "encoder_cudagraph_token_budgets": encoder_budgets,
                "encoder_cudagraph_max_vision_items_per_batch": 1,
                # The ViT attends within each temporal patch, and this sizes
                # its per-sequence metadata: a clip holds nframes / 2 of them.
                "encoder_cudagraph_max_frames_per_batch":
                nframes // TEMPORAL_PATCH_SIZE,
            },
            **kwargs)
        if self._world_size != 1:
            raise ValueError("This backend serves from a single process.")

        self._encoder_budgets = frozenset(encoder_budgets)
        self._video_token_id = (self.client.llm_engine.vllm_config.
                                model_config.hf_config.video_token_id)
        self._request_log_path = request_log
        self._compile_seconds = 0.0
        jax.monitoring.register_event_duration_secs_listener(
            self._on_jax_event)
        self._check_run_config({
            "nframes": nframes,
            "max_pixels": max_pixels,
            "processor_size": size,
            "encoder_token_budgets": encoder_budgets,
            "engine_args": {
                name: value
                for name, value in kwargs.items()
            },
            "kernel_env": {
                name: os.environ.get(name, "")
                for name in KERNEL_ENV
            },
        })

    def _on_jax_event(self, event: str, duration_secs: float,
                      **kwargs) -> None:
        if event in COMPILE_EVENTS:
            self._compile_seconds += duration_secs

    def _check_run_config(self, config: dict) -> None:
        """Pins a resumed request log to the configuration that started it."""
        path = f"{self._request_log_path}.config.json"
        config = json.loads(json.dumps(config, default=str))
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                previous = json.load(f)
            if previous != config:
                raise ValueError(
                    f"{self._request_log_path} was started with a different "
                    f"configuration:\n{previous}\nnow:\n{config}")
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def _load_request_log(self) -> dict[str, dict]:
        records = {}
        if os.path.exists(self._request_log_path):
            with open(self._request_log_path, encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    if not record["warmup"]:
                        records[record["question_id"]] = record
        return records

    def _plan(self, request) -> RequestPlan:
        _, doc_to_messages, gen_kwargs, doc_id, task, split = request.arguments
        doc = self.task_dict[task][split][doc_id]
        chat = ChatMessages(messages=doc_to_messages(doc))
        images, videos, audios = chat.extract_media()
        if images or audios or len(videos) != 1:
            raise ValueError(f"{doc['question_id']} must carry exactly one "
                             "video and no other media.")
        gen = dict(gen_kwargs or {})
        gen["max_new_tokens"] = self._select_max_new_tokens(
            gen.get("max_new_tokens"))
        gen.setdefault("temperature", 0)
        gen.setdefault("top_p", 0.95)
        params = self._build_sampling_params_dict(gen)
        messages = chat.to_hf_messages(video_kwargs={
            "nframes": self.nframes,
            "max_pixels": self.max_pixels,
        })
        prompt = self.processor.apply_chat_template(messages,
                                                    tokenize=False,
                                                    add_generation_prompt=True,
                                                    enable_thinking=False)
        return RequestPlan(question_id=doc["question_id"],
                           doc=doc,
                           video_path=videos[0],
                           prompt=prompt,
                           params=params)

    def _decode(self, path: str) -> DecodedVideo:
        started = time.perf_counter()
        (frames, metadata), sample_fps = vision_process.fetch_video(
            {
                "type": "video",
                "video": path,
                "nframes": self.nframes,
                "min_pixels": self.min_image_pixels,
                "max_pixels": self.max_pixels,
                # Budget the clip at max_pixels per frame, so the per-frame cap
                # is the one that binds.
                "total_pixels": self.nframes * self.max_pixels,
            },
            image_patch_size=IMAGE_PATCH_SIZE,
            return_video_sample_fps=True,
            return_video_metadata=True)
        metadata["do_sample_frames"] = False
        return DecodedVideo(frames=frames,
                            metadata=metadata,
                            sample_fps=sample_fps,
                            decode_seconds=time.perf_counter() - started)

    def _read_steps(self, offset: int) -> list[dict]:
        with open(self._step_log_path, encoding="utf-8") as f:
            f.seek(offset)
            steps = [json.loads(line) for line in f]
        return [step for step in steps if step["total_num_scheduled_tokens"]]

    def _serve(self, plan: RequestPlan, video: DecodedVideo) -> dict:
        inputs = {
            "prompt": plan.prompt,
            "multi_modal_data": {
                "video": [(video.frames, video.metadata)]
            },
            "mm_processor_kwargs": {
                "fps": video.sample_fps,
                "do_sample_frames": False
            },
            "multi_modal_uuids": {
                "video": [f"{plan.question_id}-{uuid.uuid4().hex}"]
            },
        }
        offset = os.path.getsize(self._step_log_path)
        self._compile_seconds = 0.0
        started = time.perf_counter()
        (output, ) = self.client.generate([inputs],
                                          [SamplingParams(**plan.params)],
                                          use_tqdm=False)
        wall_seconds = time.perf_counter() - started
        compile_seconds = self._compile_seconds

        steps = self._read_steps(offset)
        if len(steps) != 1:
            raise RuntimeError(
                f"{plan.question_id} took {len(steps)} token-scheduling steps; "
                "a one-token request without chunked prefill takes one.")
        (step, ) = steps
        (prefill_tokens, ) = step["num_scheduled_tokens"].values()
        if step["encoder_device_s"] is None:
            raise RuntimeError(
                f"{plan.question_id} ran its encoder outside the compiled "
                "budgets, so its device time was not recorded.")

        prompt_ids = output.prompt_token_ids
        visual_tokens = sum(1 for token_id in prompt_ids
                            if token_id == self._video_token_id)
        num_frames, _, height, width = video.frames.shape
        if height % TOKEN_EDGE or width % TOKEN_EDGE:
            raise RuntimeError(f"{plan.video_path} frames are {height}x{width}"
                               f", not a multiple of {TOKEN_EDGE}.")
        expected_tokens = _visual_tokens(num_frames, height, width)
        # The processor must neither resize the frames nor lose any of them,
        # the clip must fill a compiled encoder budget exactly, and the step
        # must prefill the prompt the engine reports.
        if (visual_tokens != expected_tokens
                or visual_tokens not in self._encoder_budgets
                or step["encoder_output_tokens"] != visual_tokens
                or prefill_tokens != len(prompt_ids)):
            raise RuntimeError(
                f"{plan.question_id}: {num_frames} frames of {height}x{width} "
                f"should be {expected_tokens} visual tokens; the prompt holds "
                f"{visual_tokens} (budgets {sorted(self._encoder_budgets)}), "
                f"the encoder produced {step['encoder_output_tokens']}, and "
                f"the step prefilled {prefill_tokens} of {len(prompt_ids)} "
                "prompt tokens.")

        stats = output.metrics
        return {
            "warmup": False,
            "question_id": plan.question_id,
            "videoID": plan.doc["videoID"],
            "task_type": plan.doc["task_type"],
            "answer": plan.doc["answer"],
            "output_text": output.outputs[0].text,
            "sampled_frames": num_frames,
            "frame_height": height,
            "frame_width": width,
            "video_fps": video.metadata["fps"],
            "video_total_frames": video.metadata["total_num_frames"],
            "sample_fps": video.sample_fps,
            "visual_tokens": visual_tokens,
            "text_tokens": len(prompt_ids) - visual_tokens,
            "prefill_tokens": prefill_tokens,
            "prefill_padded_tokens": step["padded_num_tokens"],
            "vision_encoder_s": step["encoder_s"],
            # The compiled ViT from on-device inputs, and the rest of the
            # encoder phase: replay buffers and their host-to-device copies.
            "vision_encoder_device_s": step["encoder_device_s"],
            "vision_encoder_host_s":
            step["encoder_s"] - step["encoder_device_s"],
            # Scattering the encoder output into the text embeddings, between
            # the encoder and the prefill.
            "embed_merge_s": step["embed_s"],
            "llm_prefill_s": step["llm_forward_s"],
            # From request arrival, so it includes the host-side multimodal
            # processing; engine_ttft_s starts when the request is scheduled.
            "ttft_s": stats.first_token_latency,
            "engine_ttft_s": stats.first_token_ts - stats.scheduled_ts,
            "queue_s": stats.scheduled_ts - stats.queued_ts,
            "video_decode_s": video.decode_seconds,
            "generate_wall_s": wall_seconds,
            "compile_s": compile_seconds,
        }

    def generate_until(self, requests) -> list[GenerationResult]:
        done = self._load_request_log()
        plans = [self._plan(request) for request in requests]
        pending = [plan for plan in plans if plan.question_id not in done]
        # Requests arrive grouped by video, so keeping the last decoded video
        # decodes each one once. The decode finishes before the request it
        # serves is submitted, so it never overlaps a timed step.
        decoded: dict[str, DecodedVideo] = {}

        with open(self._request_log_path, "a", encoding="utf-8") as log:

            def video_for(plan: RequestPlan) -> DecodedVideo:
                if plan.video_path not in decoded:
                    decoded.clear()
                    decoded[plan.video_path] = self._decode(plan.video_path)
                return decoded[plan.video_path]

            def write(record: dict) -> None:
                log.write(json.dumps(record) + "\n")
                log.flush()

            if pending:
                # The first request compiles the encoder and the prefill for
                # the common frame shape; it is served once untimed.
                warmup = self._serve(pending[0], video_for(pending[0]))
                write({**warmup, "warmup": True})

            for plan in tqdm(pending, desc="Model Responding"):
                video = video_for(plan)
                record = self._serve(plan, video)
                if record["compile_s"] > 0:
                    retimed = self._serve(plan, video)
                    first_pass = {
                        key: record[key]
                        for key in ("vision_encoder_s",
                                    "vision_encoder_device_s",
                                    "vision_encoder_host_s", "embed_merge_s",
                                    "llm_prefill_s", "ttft_s", "engine_ttft_s",
                                    "queue_s", "generate_wall_s", "compile_s")
                    }
                    record = {
                        **retimed,
                        # The answer is the first pass's; a second pass that
                        # disagrees is kept for the record, not rescored.
                        "output_text": record["output_text"],
                        "retimed_output_text": retimed["output_text"],
                        "first_pass": first_pass,
                    }
                write(record)
                done[plan.question_id] = record

        return [
            GenerationResult(text=done[plan.question_id]["output_text"])
            for plan in plans
        ]
