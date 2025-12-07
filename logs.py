# logs.py

import json
import uuid
import datetime
import os


class ExperimentLogger:
    def __init__(self, save_dir="logs", schema_version="urbanist_experiment_log_v0.1"):
        self.save_dir = save_dir
        self.schema_version = schema_version
        os.makedirs(save_dir, exist_ok=True)

    # ---------- default templates so missing fields are "empty" ----------
    def _empty_metrics(self):
        return {
            # Parity metrics (Week 2)
            "mae": None,
            "rmse": None,
            "iou_flooded": None,
            "iou_non_flooded": None,
            "percent_area_misclassified": None,

            # QC stats
            "qc_min": None,
            "qc_max": None,
            "qc_mean": None,
            "qc_std": None,
            "qc_p1": None,
            "qc_p5": None,
            "qc_p50": None,
            "qc_p95": None,
            "qc_p99": None,
            "qc_skewness": None,
            "qc_kurtosis": None,
            "qc_mode": None,
            "qc_unique_values": None,
            "qc_outlier_count": None,
            "qc_outlier_percent": None,

            # QC general info
            "qc_total_pixels": None,
            "qc_nodata_pixels": None,
            "qc_nodata_percent": None,
        }

    def _empty_metadata(self):
        return {
            "experiment_name": "",
            "dataset_id": "",
            "user": "",
            "script_name": "",
            "aoi_used": "",
            "model": {
                "name": "",
                "version": "",
                "commit_hash": ""
            }
        }

    def _empty_data_sources(self):
        return {
            "raster_files": [],
            "vector_files": []
        }

    def _empty_parameters(self):
        return {
            "model_type": "",
            "learning_rate": None,
            "batch_size": None,
            "epochs": None,
            "loss_function": "",
            "optimizer": "",
            "input_features": [],
            "extra": {}
        }

    # ---------- main logging method ----------
    def log(self,
            run_type,
            metadata=None,
            parameters=None,
            data_sources=None,
            metrics=None,
            artifacts=None):

        run_id = str(uuid.uuid4())
        timestamp_utc = datetime.datetime.utcnow().isoformat() + "Z"

        base_metadata = self._empty_metadata()
        if metadata:
            # shallow update; nested "model" can be partially provided
            for k, v in metadata.items():
                if k == "model" and isinstance(v, dict):
                    base_metadata["model"].update(v)
                else:
                    base_metadata[k] = v

        base_parameters = self._empty_parameters()
        if parameters:
            for k, v in parameters.items():
                if k in base_parameters:
                    base_parameters[k] = v
                else:
                    base_parameters["extra"][k] = v

        base_data_sources = self._empty_data_sources()
        if data_sources:
            # We accept "raster_files" and "vector_files" as lists
            if "raster_files" in data_sources:
                base_data_sources["raster_files"] = data_sources["raster_files"]
            if "vector_files" in data_sources:
                base_data_sources["vector_files"] = data_sources["vector_files"]

        base_metrics = self._empty_metrics()
        if metrics:
            for k, v in metrics.items():
                # If it's a known metric key, override default; if not, ignore
                if k in base_metrics:
                    base_metrics[k] = v

        log_obj = {
            "schema_version": self.schema_version,
            "run_id": run_id,
            "timestamp_utc": timestamp_utc,
            "run_type": run_type,

            "metadata": base_metadata,
            "data_sources": base_data_sources,
            "parameters": base_parameters,

            "results": {
                "metrics": base_metrics,
                "artifacts_saved": artifacts or []
            }
        }

        filename = f"{run_type}_{timestamp_utc.replace(':', '').replace('-', '')}_{run_id[:8]}.json"
        filepath = os.path.join(self.save_dir, filename)

        with open(filepath, "w") as f:
            json.dump(log_obj, f, indent=4)

        return filepath
