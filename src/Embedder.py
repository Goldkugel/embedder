import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from sentence_transformers      import SentenceTransformer
from typing                     import List, Union, Dict, Optional
from EmbedderConfig             import EmbedderConfig
from Logger                     import Logger
import numpy                    as np
import umap
import yaml

# Key under which embedder settings are expected to live in the YAML config file.
configuration_section: str  = "embedder"

# Default path to the config file, used if no path is explicitly passed in.
standard_directory: str     = "../config/config.yaml"

class Embedder:
    """
    Core embedding engine responsible for encoding textual terms into 
    dense vector representations and optionally applying dimensionality reduction.
    """

    models: dict = {}

    reducer = None

    def __init__(self, config: str = standard_directory):
        """
        Initialize the Embedder with a configuration object.

        Args:
            config: the file path to the configuration, if not provided 
                standard_directory is used.
        """
        data            = None
        self.models     = {}
        self.reducer    = None
        
        with open(config, "r") as f:
            data = yaml.safe_load(f)
        self.config = EmbedderConfig.model_validate(data[configuration_section])

        l = Logger()
        l.log(f"Loading model(s)...")
        if len(self.config.model_id) > 0:
            for model_id in self.config.model_id:
                l.log(f"Loading embedding model '{model_id}' on device '{self.config.device}'...")
                model = SentenceTransformer(model_id, device=self.config.device)
                model.max_seq_length = self.config.max_seq_length
                self.models[model_id] = model
                l.log(f"Loading embedding model '{model_id}' on device '{self.config.device}' completed.")

            if self.config.dimensionality_reduction:
                l.log("Loading reducer...")
                self.reducer = umap.UMAP(
                    n_components    = self.config.n_components,
                    n_neighbors     = self.config.n_neighbors,
                    min_dist        = self.config.min_dist,
                    metric          = self.config.metric,
                    random_state    = self.config.random_state,
                )
                l.log("Loading reducer completed.")
        else:
            l.log("No model specified.")
        l.log(f"Loading model(s) completed.")

    def embed(
        self, 
        terms: list
    ) -> dict:
        """
        Encode a list of text terms into vector embeddings.

        Args:
            terms: List of text strings to embed.

        Returns:
            A dictionary having as keys the model names and as values a list
            of embedding in the same order as the terms.
        """
        ret = {}
        if len(self.models.keys()) > 0:
            for model_name in self.models.keys():
                model = self.models[model_name]

                # Generate embeddings
                embeddings = model.encode(
                    terms,
                    batch_size              = self.config.batch_size,
                    show_progress_bar       = False,
                    normalize_embeddings    = self.config.normalization,
                    convert_to_numpy        = True
                )

                # Apply dimensionality reduction if enabled
                if self.config.dimensionality_reduction and self.reducer is not None and len(embeddings) > self.config.n_components:
                    embeddings = self.reducer.fit_transform(embeddings)

                ret[model_name] = embeddings

        return ret