"""Service that handles video evaluations using AI (LLMs and/or Annotations)"""

import functools

import configuration
import models
from custom_evaluation import custom_detector
from features_repository import feature_configs_handler
from helpers import gcs_utils, generic_helpers
from helpers.logger import get_logger
from llms_evaluation import llms_detector

logger = get_logger(__name__)


class VideoEvaluationService:
    """Service that handles video evaluations using AI"""

    def __init__(self):
        pass

    def evaluate_features(
        self,
        config: configuration.Configuration,
        video_uri: str,
        features_category: models.VideoFeatureCategory,
    ):
        """Run ABCD evaluation on videos for Full ABCD features or Shorts"""

        if config.extract_brand_metadata:
            metadata = llms_detector.get_video_metadata(config, video_uri)
            config.brand_name = metadata.get("brand_name")
            config.brand_variations = metadata.get("brand_variations")
            config.branded_products = metadata.get("branded_products")
            config.branded_products_categories = metadata.get(
                "branded_products_categories"
            )
            config.branded_call_to_actions = metadata.get(
                "branded_call_to_actions"
            )

        feature_evaluations: list[models.FeatureEvaluation] = []
        tasks = []
        handler = feature_configs_handler.features_configs_handler
        feature_groups = handler.get_features_by_category_by_group_config(
            features_category
        )
        uri = video_uri  # use full video uri by default

        for group_key in feature_groups:
            feature_configs: list[models.VideoFeature] = feature_groups.get(
                group_key
            )

            # Use LLM evaluation method only
            if config.use_llms and not config.use_annotations:
                handler.change_evaluation_method_to_llms_only(feature_configs)

            # Process the features that are not grouped individually
            # meaning, each will be a separate request to the LLM
            if (
                group_key == "NO_GROUPING"
                and config.use_annotations
                and config.creative_provider_type
                == models.CreativeProviderType.GCS
            ):
                for f_config in feature_configs:
                    if (
                        f_config.video_segment.value
                        == models.VideoSegment.FIRST_5_SECS_VIDEO.value
                    ):
                        uri = gcs_utils.get_reduced_uri(config, video_uri)
                    else:
                        uri = video_uri

                    # Build function to execute in parallel
                    # If custom detector was not defined, default to LLMs
                    if self.is_custom_evaluation(f_config.evaluation_function):
                        func = functools.partial(
                            custom_detector.custom_detector.evaluate_features,
                            config,
                            f_config,
                            uri,
                        )
                    else:
                        func = functools.partial(
                            llms_detector.evaluate_features,
                            config,
                            {
                                "category": features_category,
                                "group_by": f"{group_key}-{f_config.id}",
                                "video_uri": uri,
                                "feature_configs": feature_configs,
                            },
                        )
                    # Add task to be process
                    tasks.append(func)
            else:
                # Use full video for Public URL videos
                if (
                    group_key == models.VideoSegment.FIRST_5_SECS_VIDEO.value
                    and config.creative_provider_type
                    == models.CreativeProviderType.GCS
                ):
                    uri = gcs_utils.get_reduced_uri(config, video_uri)
                else:
                    uri = video_uri

                # Build function to execute in parallel
                func = functools.partial(
                    llms_detector.evaluate_features,
                    config,
                    {
                        "category": features_category,
                        "group_by": f"{group_key} for video {uri}",
                        "video_uri": uri,
                        "feature_configs": feature_configs,
                    },
                )
                # Add task to be process
                tasks.append(func)

        logger.info("Starting ABCD evaluation for features... \n")

        llm_evals = generic_helpers.execute_tasks_in_parallel(tasks)

        # Process LLM results and create feature objs in the required format
        for evals in llm_evals:
            for evaluated_feature in evals:
                feature: models.VideoFeature = handler.get_feature_by_id(
                    evaluated_feature.get("id")
                )
                if feature:
                    feature_evaluations.append(
                        models.FeatureEvaluation(
                            feature=feature,
                            detected=evaluated_feature.get("detected"),
                            confidence_score=evaluated_feature.get(
                                "confidence_score"
                            ),
                            rationale=evaluated_feature.get("rationale"),
                            evidence=evaluated_feature.get("evidence"),
                            strengths=evaluated_feature.get("strengths"),
                            weaknesses=evaluated_feature.get("weaknesses"),
                        )
                    )
                else:
                    logger.warning(
                        "Feature %s not found. Feature was not added to"
                        " feature_evaluations.",
                        evaluated_feature.get("id"),
                    )

        # Sort features by category and id for presentation
        if features_category == models.VideoFeatureCategory.LONG_FORM_ABCD:
            feature_evaluations = sorted(
                feature_evaluations,
                key=lambda feature_eval: (
                    feature_eval.feature.category.value,
                    feature_eval.feature.id,
                ),
                reverse=False,
            )

        return feature_evaluations

    def is_custom_evaluation(self, function_name):
        return function_name != ""


video_evaluation_service = VideoEvaluationService()
