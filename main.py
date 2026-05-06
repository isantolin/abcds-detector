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

"""Module to execute the ABCD Detector Assessment"""

import time
import traceback

from google.cloud import storage

import models
import utils
from annotations_evaluation import annotations_generation
from configuration import Configuration
from evaluation_services import video_evaluation_service
from helpers import generic_helpers
from helpers.logger import get_logger

logger = get_logger(__name__)


def get_creative_uris(config: Configuration) -> list[str]:
  """Retrieves creative URIs based on the provider type."""
  if config.creative_provider_type == models.CreativeProviderType.YOUTUBE:
    return config.video_uris

  uris = []
  client = storage.Client()
  for uri in config.video_uris:
    if uri.endswith("/"):
      print(f"EXPANDING URI: {uri} \n")
      bucket, prefix = uri.replace("gs://", "").split("/", 1)
      for blob in client.get_bucket(bucket).list_blobs(
        prefix=prefix, delimiter="/"
      ):
        if not blob.name.endswith("/"):
          uris.append(f"gs://{bucket}/{blob.name}")
    else:
      uris.append(uri)
  return uris


def execute_abcd_assessment_for_videos(config: Configuration):
  """Execute ABCD Assessment for all brand videos retrieved."""

  video_uris = get_creative_uris(config)

  for video_uri in video_uris:
    # Validate that creative provides match the video uris
    if (
      config.creative_provider_type == models.CreativeProviderType.GCS
      and "gs://" not in video_uri
    ):
      logger.error(
        "The creative provider GCS does not match with the video uri"
        " %s. Stopping execution. Please check.",
        video_uri,
      )
      break

    if (
      config.creative_provider_type == models.CreativeProviderType.YOUTUBE
      and "https://www.youtube.com" not in video_uri
    ):
      logger.error(
        "The creative provider YOUTUBE does not match with the video "
        "uri %s. Stopping execution. Please check.",
        video_uri,
      )
      break

    print(f"\n\nProcessing ABCD Assessment for video {video_uri}... \n")

    # Generate video annotations for custom features.
    # Annotations are supported only for GCS providers
    if (
      config.use_annotations
      and config.creative_provider_type == models.CreativeProviderType.GCS
    ):
      annotations_generation.generate_video_annotations(config, video_uri)

    # Full ABCD features require 1st_5_secs videos only for GCS providers
    if (
      config.run_long_form_abcd
      and config.creative_provider_type == models.CreativeProviderType.GCS
    ):
      generic_helpers.trim_video(config, video_uri)

    # Execute ABCD Assessment
    long_form_abcd_evaluated_features: models.FeatureEvaluation = []
    shorts_evaluated_features: models.FeatureEvaluation = []

    if config.run_long_form_abcd:
      long_form_abcd_evaluated_features = (
        video_evaluation_service.video_evaluation_service.evaluate_features(
          config=config,
          video_uri=video_uri,
          features_category=models.VideoFeatureCategory.LONG_FORM_ABCD,
        )
      )

    if config.run_shorts:
      shorts_evaluated_features = (
        video_evaluation_service.video_evaluation_service.evaluate_features(
          config=config,
          video_uri=video_uri,
          features_category=models.VideoFeatureCategory.SHORTS,
        )
      )

    video_assessment: models.VideoAssessment = models.VideoAssessment(
      brand_name=config.brand_name,
      video_uri=video_uri,
      long_form_abcd_evaluated_features=long_form_abcd_evaluated_features,
      shorts_evaluated_features=shorts_evaluated_features,
      config=config,
    )

    # Print assessments for Full ABCD and Shorts and store results
    if len(long_form_abcd_evaluated_features) > 0:
      generic_helpers.print_abcd_assessment(
        video_assessment.brand_name,
        video_assessment.video_uri,
        long_form_abcd_evaluated_features,
      )
    else:
      logger.info(
        "There are not Full ABCD evaluated features results to display."
      )
    if len(shorts_evaluated_features) > 0:
      generic_helpers.print_abcd_assessment(
        video_assessment.brand_name,
        video_assessment.video_uri,
        shorts_evaluated_features,
      )
    else:
      logger.info("There are not Shorts evaluated features results to display.")

    if config.bq_table_name:
      generic_helpers.store_in_bq(config, video_assessment)

    # Remove local version of video files
    generic_helpers.remove_local_video_files()


def main(arg_list: list[str] | None = None) -> None:
  """Main ABCD Assessment execution. See docstring and args.

  Args:
    arg_list: A list of command line arguments

  """

  try:
    args = utils.parse_args(arg_list)

    config = utils.build_abcd_params_config(args)

    if utils.invalid_brand_metadata(config):
      logger.error(
        "The Extract Brand Metadata option is disabled and no brand "
        "details were defined. \n"
      )
      logger.error("Please enable the option or define brand details. \n")
      return

    start_time = time.time()
    logger.info("Starting ABCD assessment... \n")

    if config.video_uris:
      execute_abcd_assessment_for_videos(config)
      logger.info("Finished ABCD assessment. \n")
    else:
      logger.info("There are no videos to process. \n")

    duration = (time.time() - start_time) / 60
    logger.info("ABCD assessment took - %s mins. - \n", duration)
  except (ValueError, KeyError) as ex:
    logger.error("ERROR: %s", ex)
    traceback.print_exc()
  except Exception:
      logger.exception("Unexpected ERROR")

if __name__ == "__main__":
  main()
