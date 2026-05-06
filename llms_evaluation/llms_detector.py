#!/usr/bin/env python3

###########################################################################
#
#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###########################################################################

"""Module to evaluate and detect features using LLMs."""

import json
import time

from google import genai
from google.api_core.exceptions import ResourceExhausted
from google.genai import types
from google.genai.errors import APIError

from configuration import Configuration
from models import (
    VIDEO_METADATA_RESPONSE_SCHEMA,
    VIDEO_RESPONSE_SCHEMA,
    LLMParameters,
)
from prompts.prompt_generator import PromptConfig, prompt_generator


def clean_llm_response(response: str) -> str:
    """Cleans LLM response
    Args:
        response: llm response to clean
    Returns:
        reponse: without extra characters
    """
    if not response:
        return "[]"
    return response.replace("```", "").replace("json", "").strip()


def _get_modality_params(prompt: str, params: LLMParameters) -> list[any]:
    """Build the modality params based on the type of llm capability to use"""
    if params.modality["type"] == "video":
        mime_type = f"video/{params.modality['video_uri'].rsplit('.', 1)[-1]}"
        video = types.Part.from_uri(
            file_uri=params.modality["video_uri"], mime_type=mime_type
        )
        return [
            types.Content(role="user", parts=[video, types.Part.from_text(text=prompt)])
        ]
    elif params.modality["type"] == "text":
        return [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    return []


def execute_gemini(
    project_id: str, prompt_config: PromptConfig, llm_params: LLMParameters
) -> str:
    """Executes Gemini using the GenAI library with retries"""
    retries = 3
    for this_retry in range(retries):
        try:
            client = genai.Client(
                vertexai=True,
                project=project_id,
                location=llm_params.location,
            )
            # Build prompt parts
            contents = _get_modality_params(prompt_config.prompt, llm_params)
            generate_content_config = types.GenerateContentConfig(
                temperature=llm_params.generation_config.get("temperature"),
                top_p=llm_params.generation_config.get("top_p"),
                seed=0,
                max_output_tokens=llm_params.generation_config.get("max_output_tokens"),
                response_modalities=["TEXT"],
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT", threshold="OFF"
                    ),
                ],
                system_instruction=[
                    types.Part.from_text(text=prompt_config.system_instructions)
                ],
                response_mime_type="application/json",
            )

            # Check if response_schema is provided and add it
            schema = llm_params.generation_config.get("response_schema")
            if schema:
                generate_content_config.response_schema = schema

            # Get response from Gemini
            response = client.models.generate_content(
                model=llm_params.model_name,
                contents=contents,
                config=generate_content_config,
            )

            return response.text
        except ResourceExhausted as ex:
            print(f"QUOTA RETRY: {this_retry + 1}. ERROR {ex!s} ...")
            wait = 10 * 2**this_retry
            time.sleep(wait)
        except APIError as ex:
            print(f"API Error. {ex} ...")
            error_message = str(ex)
            if (
                "429" in error_message
                or "503" in error_message
                or "500" in error_message
            ):
                print(
                    f"Retrying {retries} times using exponential backoff. Retry number {this_retry + 1}...\n"
                )
                wait = 10 * 2**this_retry
                time.sleep(wait)
            else:
                raise
        except (ValueError, KeyError, Exception) as ex:
            print("GENERAL EXCEPTION...\n")
            error_message = str(ex)
            if (
                "429" in error_message
                or "503" in error_message
                or "500" in error_message
            ):
                print(
                    f"Error {error_message}. Retrying {retries} times using"
                    f" exponential backoff. Retry number {this_retry + 1}...\n"
                )
                wait = 10 * 2**this_retry
                time.sleep(wait)
            else:
                print(f"ERROR: the following issue can't be retried: {error_message}\n")
                raise
    return ""


def evaluate_features(
    config: Configuration,
    evaluation_details: dict,
) -> list[dict]:
    """Evaluates ABCD features using LLMs."""
    print(
        "Starting LLM evaluation for features grouped by"
        f" {evaluation_details.get('category')} and"
        f" {evaluation_details.get('group_by')}... \n"
    )
    prompt_config = prompt_generator.get_abcds_prompt_config(
        evaluation_details.get("feature_configs"),
        config,
    )
    # Create new object here to avoid race condition when executing
    llm_params = LLMParameters()
    llm_params.model_name = config.llm_params.model_name
    llm_params.location = config.llm_params.location
    llm_params.generation_config = config.llm_params.generation_config.copy()
    # Set modality for API
    llm_params.set_modality(
        {"type": "video", "video_uri": evaluation_details.get("video_uri")}
    )
    # Set the required schema for the LLM response
    llm_params.generation_config["response_schema"] = VIDEO_RESPONSE_SCHEMA

    response = execute_gemini(config.project_id, prompt_config, llm_params)

    try:
        evaluated_features = json.loads(clean_llm_response(response))
    except (json.JSONDecodeError, TypeError):
        evaluated_features = []

    if config.verbose and (not evaluated_features or len(evaluated_features) == 0):
        print(
            "WARNING: ABCD Detector was not able to process features for "
            f"video {evaluation_details.get('video_uri')}... \n"
        )

    return evaluated_features if isinstance(evaluated_features, list) else []


def get_video_metadata(config: Configuration, video_uri: str) -> dict:
    """Extracts metadata from video"""
    print(f"Extracting brand metadata for video {video_uri}... \n")
    prompt_config = prompt_generator.get_metadata_prompt_config()

    llm_params = LLMParameters()
    llm_params.model_name = config.llm_params.model_name
    llm_params.location = config.llm_params.location
    llm_params.generation_config = config.llm_params.generation_config.copy()

    # Set modality for API
    llm_params.set_modality({"type": "video", "video_uri": video_uri})
    # Set the required schema for the LLM response
    llm_params.generation_config["response_schema"] = VIDEO_METADATA_RESPONSE_SCHEMA

    response = execute_gemini(config.project_id, prompt_config, llm_params)

    try:
        metadata = json.loads(clean_llm_response(response))
    except (json.JSONDecodeError, TypeError):
        metadata = {}

    return metadata if isinstance(metadata, dict) else {}
