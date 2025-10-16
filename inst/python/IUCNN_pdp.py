#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Octuber 2025

@author: Torsten hauffe (torsten.hauffe8@gmail.com)
"""

import os, sys
# use only one thread
try:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
except:
    pass

import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# disable progress bars globally (instead of model.predict(..., verbose=0), which does not supress progress output in R)
tf.keras.utils.disable_interactive_logging()

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disable tf compilation warning
except:
    pass

# declare location of the python files make functions of other python files importable
sys.path.append(os.path.dirname(__file__))
from IUCNN_predict import rescale_labels, turn_reg_output_into_softmax


def iucnn_pdp(input_features,
              focal_features,
              model_dir,
              iucnn_mode,
              cv_fold,
              dropout,
              dropout_reps,
              rescale_factor,
              min_max_label,
              stretch_factor_rescaled_labels
             ):

    if cv_fold == 1:
        model = [tf.keras.models.load_model(model_dir)]
    else:
        model = [tf.keras.models.load_model(model_dir[i]) for i in range(cv_fold)]

    if not isinstance(focal_features, list):
        focal_features = [focal_features]

    pdp_features = make_pdp_features(input_features, focal_features)
    num_pdp_steps = pdp_features.shape[0]

    if iucnn_mode == 'nn-reg':
        num_iucn_cat = int(rescale_factor + 1)
        label_cats = np.arange(num_iucn_cat)
    else:
        num_iucn_cat = min_max_label[1] + 1
    pdp = np.zeros((num_pdp_steps, num_iucn_cat))

    if dropout:
        pdp_lwr = np.zeros((num_pdp_steps, num_iucn_cat))
        pdp_upr = np.zeros((num_pdp_steps, num_iucn_cat))

    if iucnn_mode == 'cnn':
        sys.exit('No partial dependence probabilities possible for CNN')

    else:
        for i in range(num_pdp_steps):
            tmp_features = np.copy(input_features)
            tmp_features[:, focal_features] = pdp_features[i, :]

            if dropout:
                predictions_raw = []
                for j in range(cv_fold):
                    predictions_raw.append(np.array([model[j].predict(tmp_features, verbose=0) for i in np.arange(dropout_reps)]))
                predictions_raw = np.vstack(predictions_raw)

                if iucnn_mode == 'nn-reg':
                    predictions_rescaled = np.array([rescale_labels(j, rescale_factor, min_max_label, stretch_factor_rescaled_labels, reverse=True) for j in predictions_raw])
                    s = predictions_rescaled.shape
                    predictions_raw = np.zeros((s[0], s[1], num_iucn_cat))
                    for j in range(s[0]):
                        predictions_raw[j, :, :] = turn_reg_output_into_softmax(predictions_rescaled[j, :, :].T, label_cats)

                # cumulative probs for each taxon and each dropout reps
                pred_reps = np.cumsum(predictions_raw, axis=2)
                # mean prob across taxa for each dropout reps
                pred_reps = np.mean(pred_reps, axis=1)

                # uncertainty in cumulative prob across dropout reps
                pred_quantiles = np.quantile(pred_reps, q=(0.025, 0.975), axis=0)
                pdp_lwr[i, :] = pred_quantiles[0, :]
                pdp_lwr[i, :] = pdp_lwr[i, :]
                pdp_upr[i, :] = pred_quantiles[1, :]
                pdp_upr[i, :] = pdp_upr[i, :]

                pred_mean = np.mean(pred_reps, axis=0)

            else:
                predictions_raw = []
                for j in range(cv_fold):
                    predictions_raw.append(model[j].predict(tmp_features, verbose=0))
                predictions_raw = np.vstack(predictions_raw)

                if iucnn_mode == 'nn-reg':
                    predictions_raw = np.array([rescale_labels(j, rescale_factor, min_max_label, stretch_factor_rescaled_labels, reverse=True) for j in predictions_raw])
                    predictions_raw = turn_reg_output_into_softmax(predictions_raw.T, label_cats)

                # cumulative probs for each taxon
                pred = np.cumsum(predictions_raw, axis=1)
                # mean prob across taxa
                pred_mean = np.mean(pred, axis=0)

            pdp[i, :] = pred_mean


    out_dict = {
        'feature': pdp_features,
        'pdp': pdp
    }
    if dropout:
        out_dict.update({'lwr': pdp_lwr, 'upr': pdp_upr})

    return  out_dict



def get_focal_summary(input_features, focal_features):
    """Get whether features are ohe/binary/ordinal/continuous and their min and max values"""
    num_features = len(focal_features)
    focal_summary = np.zeros((3, num_features))

    for i in range(num_features):
        ff = input_features[:, focal_features[i]]
        values, counts = np.unique(ff, return_counts=True)
        focal_summary[1, i] = np.nanmin(values)
        focal_summary[2, i] = np.nanmax(values)
        values_range = np.arange(focal_summary[1, i], focal_summary[2, i] + 1)
        focal_summary[0, i] = np.all(np.isin(values, values_range))

    return focal_summary


def make_pdp_features(input_features, focal_features):
    """Get the features for which we calculate the PDP"""
    focal_summary = get_focal_summary(input_features, focal_features)

    if np.sum(focal_summary[0, :] == 0) and len(focal_features) == 1:
        # Single continuous feature
        pdp_feat = np.linspace(focal_summary[1, 0], focal_summary[2, 0], num=100).reshape(100, 1)
    elif focal_summary[0, 0] == 1 and len(focal_features) == 1:
        # ordinal or binary
        M = focal_summary[2, 0]
        pdp_feat = np.linspace(focal_summary[1, 0], M, num=M + 1).reshape((M + 1, 1))
    else:
        # One-hot-encoded
        pdp_feat = np.eye(focal_summary.shape[1])

    return pdp_feat
